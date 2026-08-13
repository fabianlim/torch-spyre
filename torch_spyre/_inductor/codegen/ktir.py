# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpSpec -> KTIR emitter.

``generate_ktir`` is an OpSpec consumer: it consumes the finished
``list[OpSpec | LoopSpec]`` kernel contract (the same contract the SDSC bundle
emitter ``generate_bundle`` consumes) and emits **KTDP-dialect MLIR** directly.
The module is built with the ``mlir_ktdp`` Python builders, so the returned
``str(module)`` is canonical, verifier-checked MLIR that the golden snapshot
test consumes without drift.

It uses the OpSpec-reading helpers from ``opspec_utils`` to adapt the OpSpec
information to generate_ktir.

Base addresses are emitted either as func arguments or as baked
``arith.constant``s, selected by ``config.bundle_symbolic_args``.  The baked
form is a temporary dataflow-scheduler#65 workaround, to be reverted when the
backend accepts symbolic addresses.

Structure
---------

``generate_ktir`` is four steps, in this order:

1. ``validate(specs)`` -- a **pure** recursive walk that raises every
   ``NotImplementedError`` the emitter can raise and returns a ``BufferTable``.
   It imports nothing from ``mlir_ktdp``, so every rejection is reachable and
   testable where the dialect build is absent.
2. ``address_source(table)`` -- picks ``BakedConstants`` or
   ``FuncArgAddresses``.  The single seam for reverting #65.
3. ``KtirBuilder.create(...)`` -- the single ``mlir_ktdp`` import site; owns the
   context, the dialect handles and the per-module state.
4. ``emit_specs(b, specs)`` -- a recursive walk over the same spec tree,
   dispatching each ``OpSpec`` through the module-level ``REGISTRY``.

Adding a pointwise op is one ``REGISTRY`` entry.  Enabling counted loops is
dropping the ``LoopSpec`` rejection in ``_validate_list`` and filling in
``emit_loop``; the walk already recurses.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.codegen.opspec_utils import (
    _align_reshape_plan,
    _buf_id,
    _row_major_strides,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

# The dialect handles.  Modules are singletons in sys.modules, so they are held
# here rather than threaded through KtirBuilder: one name per dialect, bound once
# by _load_dialects().
#
# Under TYPE_CHECKING these are the real imports, which is what gives `ir.Module`
# and `arith.AddFOp` their types.  At runtime the block does not execute, so
# importing this module needs no dialect build -- `validate` stays usable without
# one -- and the names are None until _load_dialects() runs.
if TYPE_CHECKING:
    from mlir_ktdp import ir
    from mlir_ktdp.dialects import arith, func, ktdp, linalg, scf, tensor
else:
    ir = arith = func = ktdp = linalg = scf = tensor = None


def _load_dialects() -> None:
    """Bind the dialect handles into this module, once.  The only import site."""
    global ir, arith, func, ktdp, linalg, scf, tensor
    if ir is not None:
        return
    from mlir_ktdp import ir as _ir
    from mlir_ktdp.dialects import arith as _arith
    from mlir_ktdp.dialects import func as _func
    from mlir_ktdp.dialects import ktdp as _ktdp
    from mlir_ktdp.dialects import linalg as _linalg
    from mlir_ktdp.dialects import scf as _scf
    from mlir_ktdp.dialects import tensor as _tensor

    ir, arith, func, ktdp, linalg, scf, tensor = (
        _ir,
        _arith,
        _func,
        _ktdp,
        _linalg,
        _scf,
        _tensor,
    )


def dialect_available() -> bool:
    """True when the bindings the emitter needs are importable.

    Asked of this module rather than reimplemented by callers, so the import list
    exists in exactly one place and cannot drift.
    """
    try:
        _load_dialects()
    except ImportError:
        return False
    return True


# Supported device dtype -> the *name* of the ``mlir_ktdp.ir`` type builder for
# it.  Names, not builder references, so this table stays importable without the
# dialect: ``validate`` uses it as the supported-dtype predicate and
# ``KtirBuilder.elt_type`` resolves the name against the imported ``ir``.  The
# two fp16 device formats both map to ``f16``; extend this map (never fall
# through silently) as new dtypes are supported.
_MLIR_ELT_TYPE_NAMES: dict[DataFormats, str] = {
    DataFormats.IEEE_FP16: "F16Type",
    DataFormats.SEN169_FP16: "F16Type",
    DataFormats.IEEE_FP32: "F32Type",
    DataFormats.BFLOAT16: "BF16Type",
}


# ---------------------------------------------------------------------------
# BufferTable: the one record the walk needs up front
# ---------------------------------------------------------------------------
#
# The func signature must be known before the body is emitted, and the address
# source needs the buffer list to choose zero-arg versus N-arg, so the buffers
# are collected by the validation walk into a flat table keyed by ``_buf_id``.
# Everything else (strides, block shapes, coordinate bounds) is a pure function
# of ``sizes`` and is recomputed per access by a single helper apiece, rather
# than resolved into records.


@dataclasses.dataclass(frozen=True)
class BufferEntry:
    """One unique buffer referenced by the kernel."""

    buf_id: str  # opspec_utils._buf_id(arg)
    arg_index: int  # >= 0 external; -1 fused intermediate (rejected today)
    sizes: list[int]  # arg.device_size
    dtype: DataFormats
    base_elements: int | None  # ELEMENTS for the baked form; None => func arg


class BufferTable:
    """Unique buffers in first-seen order, keyed by ``_buf_id``."""

    def __init__(self) -> None:
        self.buffers: dict[str, BufferEntry] = {}

    def add(self, arg: TensorArg) -> None:
        """Register ``arg``'s buffer, rejecting what the emitter cannot address."""
        buf_id = _buf_id(arg)
        if buf_id in self.buffers:
            return
        # Only real external buffers become func parameters; register-threaded
        # fused intermediates carry the -1 sentinel and would have to be threaded
        # as SSA values instead.
        if arg.arg_index < 0:
            raise NotImplementedError(
                "OpSpec->KTIR: fused intermediates (register threading) "
                "not supported yet"
            )
        if arg.device_dtype not in _MLIR_ELT_TYPE_NAMES:
            raise NotImplementedError(
                f"OpSpec->KTIR: unsupported device dtype {arg.device_dtype!r}"
            )
        self.buffers[buf_id] = BufferEntry(
            buf_id=buf_id,
            arg_index=arg.arg_index,
            sizes=[int(s) for s in arg.device_size],
            dtype=arg.device_dtype,
            # Resolved only for the baked form: the symbolic form takes its bases
            # from func arguments and never reads ``allocation["hbm"]``, whose
            # units differ between the two forms.
            base_elements=(
                _base_address_elements(arg) if _addresses_are_baked() else None
            ),
        )

    @property
    def param_entries(self) -> list[BufferEntry]:
        """External buffers in ascending ``arg_index``.

        Ascending ``arg_index`` matches the positional order ``call_kernel``
        passes to ``.run(...)``, so the emitted func signature lines up with
        that binding.
        """
        return sorted(
            (e for e in self.buffers.values() if e.arg_index >= 0),
            key=lambda e: e.arg_index,
        )


def _addresses_are_baked() -> bool:
    """True when base addresses are baked ``arith.constant``s, not func args.

    TENTATIVE (dataflow-scheduler#65): the backend rejects symbolic addresses.
    """
    return not _spyre_config.bundle_symbolic_args


def _base_address_elements(arg: TensorArg) -> int:
    """``arg``'s buffer base address in ELEMENTS, for the baked form only.

    Read from ``allocation["hbm"]``, the same field the SDSC path resolves into
    the bundle start address (``superdsc.py:774`` -> ``startAddressCoreCorelet_``).
    Its units follow ``config.bundle_symbolic_args``: baked gives a byte address
    (arg 1 -> ``{'hbm': 17179869184}``), symbolic a bare sentinel ``arg_index``
    (arg 1 -> ``{'hbm': 1}``).  A memref offset indexes the *element* type, so
    the byte address is scaled down by the element size.
    """
    allocation = arg.allocation or {}
    # Key presence, not truthiness: a legitimate 'hbm' address of 0 exists.
    if "hbm" not in allocation:
        space = next(iter(allocation), None)
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} is not HBM-allocated "
            f"(allocation={allocation!r}); the emitter only emits HBM memory "
            f"views, so {space!r} allocations are out of scope"
        )
    byte_offset = allocation["hbm"]
    if byte_offset is None:
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} has an unassigned 'hbm' "
            "address (None); memory planning must run before KTIR emission"
        )
    return int(byte_offset) // num_bytes(arg.device_dtype)


# ---------------------------------------------------------------------------
# validate: every rejection, with no mlir_ktdp
# ---------------------------------------------------------------------------


def validate(
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
) -> BufferTable:
    """Pure. Raises every ``NotImplementedError`` the emitter can raise.

    Imports nothing from ``mlir_ktdp`` (the dialect import is lazy, inside
    ``KtirBuilder.create``), so it is testable wherever ``import ktir`` works --
    which is everywhere.  Afterwards the emission path holds no ``raise`` but
    the ``AssertionError`` in ``emit_specs``, which only fires on a validation
    bug.
    """
    # Multi-core work division is future work; the emitted grid is hard-coded to
    # a single core, so reject anything else rather than silently emitting a
    # single-core grid on a multi-core request.
    if _spyre_config.sencores != 1:
        raise NotImplementedError(
            "OpSpec->KTIR: multi-core work division is not supported yet "
            f"(SENCORES={_spyre_config.sencores}, only 1 is supported)"
        )
    table = BufferTable()
    _validate_list(specs, table)
    if not table.buffers:
        raise NotImplementedError("OpSpec->KTIR: no OpSpec to emit")
    return table


def _validate_list(specs, table: BufferTable) -> None:
    """Recursive: validate one spec list. Mirrors ``emit_specs``' structure."""
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            raise NotImplementedError(f"OpSpec->KTIR: unimplemented op {entry.op!r}")
        if isinstance(entry, LoopSpec):
            raise NotImplementedError(
                "OpSpec->KTIR: counted loops (LoopSpec) are not supported yet"
            )
            # When loops land, this becomes: _validate_list(entry.body, table)
        if not isinstance(entry, OpSpec):
            raise NotImplementedError(
                f"OpSpec->KTIR: unexpected spec entry {type(entry).__name__}"
            )
        if entry.is_reduction:
            raise NotImplementedError("OpSpec->KTIR: reductions are not supported yet")
        if entry.op not in REGISTRY:
            raise NotImplementedError(
                f"OpSpec->KTIR: op {entry.op!r} is not supported yet "
                "(only pointwise 'add')"
            )
        _validate_op(entry, table)


def _validate_op(spec: OpSpec, table: BufferTable) -> None:
    """Per-op checks: roles/arity, in-place aliasing, operand alignment, buffers."""
    out, inputs = validated_roles(spec)
    out_extents = [int(s) for s in out.device_size]
    for arg in inputs:
        # In-place (input buffer aliases the output) is not supported yet.
        if _buf_id(arg) == _buf_id(out):
            raise NotImplementedError(
                "OpSpec->KTIR: in-place ops (input aliases output) not supported"
            )
        # Reject broadcast / transpose operands: only operands whose device
        # axes already match the output tile exactly are supported.
        plan = _align_reshape_plan(
            list(arg.device_coordinates),
            [int(s) for s in arg.device_size],
            list(out.device_coordinates),
            out_extents,
        )
        if plan is not None:
            raise NotImplementedError(
                "OpSpec->KTIR: broadcast / reshape operands not supported yet"
            )
    # Buffer-level rejections (fused intermediate, dtype, base address) live in
    # BufferTable.add, so registration and validation cannot disagree.
    for arg in spec.args:
        table.add(arg)


def validated_roles(spec: OpSpec) -> tuple[TensorArg, list[TensorArg]]:
    """``(output, inputs)`` for ``spec``, or raise. Pure; shared with ``validate``.

    Handlers call this instead of re-deriving the roles, so the arity and
    single-output rejections have exactly one implementation.
    """
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(outputs) != 1:
        raise NotImplementedError(
            f"OpSpec->KTIR: expected exactly one output, got {len(outputs)}"
        )
    arity = REGISTRY[spec.op].arity
    if len(inputs) != arity:
        raise NotImplementedError(
            f"OpSpec->KTIR: {spec.op!r} expects {arity} inputs, got {len(inputs)}"
        )
    return outputs[0], inputs


# ---------------------------------------------------------------------------
# Op registration: one module-level map
# ---------------------------------------------------------------------------


def _emit_add(b: KtirBuilder, spec: OpSpec) -> None:
    """Load / compute / store for one pointwise ``OpSpec``."""
    out, inputs = validated_roles(spec)
    loaded = [b.operand(a) for a in inputs]
    result = b.pointwise(spec.op, loaded, out)
    b.store_to(out, result)


@dataclasses.dataclass(frozen=True)
class OpEntry:
    """Everything the emitter knows about one supported op name.

    One record per op, so adding an op is one ``REGISTRY`` entry rather than an
    entry in each of several maps keyed by the same op name.

    ``arith_builder`` / ``linalg_builder`` are the two compute spellings the two
    address forms need (see ``AddressSource``); they are builder *names*, not
    references, so this table needs no dialect import.
    """

    emit: Callable[[KtirBuilder, OpSpec], None]
    arity: int
    arith_builder: str  # ``arith`` op class, for the symbolic form
    linalg_builder: str  # ``linalg`` named op, for the baked form


# A map, not a decorator: registration is explicit, greppable, has no
# import-order dependency, and is enumerable by ``validate`` and by tests.
REGISTRY: dict[str, OpEntry] = {
    "add": OpEntry(
        emit=_emit_add, arity=2, arith_builder="AddFOp", linalg_builder="add"
    ),
}


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def emit_specs(b: KtirBuilder, specs) -> None:
    """Emit a spec list into the current insertion point. Recursive."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            emit_loop(b, entry)
        elif isinstance(entry, OpSpec):
            REGISTRY[entry.op].emit(b, entry)
        else:
            # validate() already rejected UnimplementedOp and anything else, so
            # reaching here is a validation bug, not an unsupported request.
            # AssertionError (not TypeError) says exactly that.
            raise AssertionError(  # noqa: TRY004
                f"unvalidated spec entry {type(entry).__name__}"
            )


def emit_loop(b: KtirBuilder, loop: LoopSpec) -> None:
    """``scf.for`` over ``loop.count``, body emitted inside the loop's region.

    Not supported yet -- ``_validate_list`` rejects ``LoopSpec``, so this is
    unreachable.  The shape is here so that enabling loops is a local change
    rather than a restructure: the nesting in the emitted IR comes from
    ``ir.InsertionPoint`` plus ``b.env.scope(...)``, both scoped by ``with``, so
    the walk needs no parallel structure to know where it is.
    """
    lo, step = b.icst_index(0), b.icst_index(1)
    hi = b.trip_count(loop.count)  # int today; sympy later
    for_op = scf.ForOp(lo, hi, step)
    with (
        ir.InsertionPoint(for_op.body),
        b.env.scope(iv=for_op.induction_variable),
    ):
        emit_specs(b, loop.body)  # the recursion point


# ---------------------------------------------------------------------------
# What the walk carries in scope
# ---------------------------------------------------------------------------


class ScopeStack:
    """Builder-owned lexical scope: loop induction variables and live values.

    Pushed and popped by the walk via ``with``.  A base frame is always present
    so values produced at function level have somewhere to live.
    """

    def __init__(self) -> None:
        # (induction variable or None, {buf_id: Value}) innermost last.
        self._frames: list[tuple[Any, dict[str, Any]]] = [(None, {})]

    @contextlib.contextmanager
    def scope(self, iv: Any = None) -> Iterator[None]:
        self._frames.append((iv, {}))
        try:
            yield
        finally:
            self._frames.pop()

    def produced(self, buf_id: str):
        """The ``Value`` a live node produced for ``buf_id``, else ``None``."""
        for _, produced in reversed(self._frames):
            if buf_id in produced:
                return produced[buf_id]
        return None

    def bind_produced(self, buf_id: str, value) -> None:
        self._frames[-1][1][buf_id] = value

    def ivs(self) -> list:
        """Enclosing induction variables, innermost last."""
        return [iv for iv, _ in self._frames if iv is not None]


# ---------------------------------------------------------------------------
# KtirBuilder
# ---------------------------------------------------------------------------


class KtirBuilder:
    """Owns the MLIR context, the dialect handles and per-module state.

    Knows nothing about ``OpSpec`` beyond what handlers pass in.  Every ktdp
    shape method returns an SSA ``Value``, so ``val()`` does not appear at call
    sites.
    """

    def __init__(self, addrs, stack):
        self.addrs = addrs
        self._stack = stack
        self.env = ScopeStack()
        # Requires the live context entered by create().
        self.index_t = ir.IndexType.get()
        self.block_args: list = []
        self.views: dict[str, Any] = {}
        self.c0 = None
        self._text: str | None = None

    @classmethod
    def create(cls, addrs: AddressSource) -> KtirBuilder:
        """THE single lazy-import site, and the owner of the MLIR context.

        Module level stays ``mlir_ktdp``-free, so ``import ktir`` -- and
        therefore ``validate`` -- works where the dialect build is absent.

        The context is entered here rather than in ``module()`` because
        ``AddressSource.func_param_types`` builds ``ir`` types and is called
        before the module is opened; ``module()`` closes it on the way out.
        """
        _load_dialects()

        stack = contextlib.ExitStack()
        try:
            ctx = stack.enter_context(ir.Context())
            stack.enter_context(ir.Location.unknown())
            ktdp.register_dialects(ctx)
            return cls(addrs, stack)
        except BaseException:
            stack.close()
            raise

    # -- generic helpers ---------------------------------------------------

    def val(self, x):
        """The SSA ``Value`` of a builder result (builders return ``OpView`` or ``Value``)."""
        return x.result if hasattr(x, "result") else x

    def elt_type(self, dtype: DataFormats):
        """The ``ir`` element type for a Spyre device dtype (validated already)."""
        return getattr(ir, _MLIR_ELT_TYPE_NAMES[dtype]).get()

    def icst_index(self, value: int):
        """A fresh ``arith.constant <value> : index``."""
        return self.val(arith.ConstantOp(self.index_t, int(value)))

    def trip_count(self, count):
        """``count`` as an index SSA value. ``int`` today; sympy later."""
        return self.icst_index(int(count))

    def zero_offsets(self, rank: int) -> list:
        """``rank`` copies of the function-entry ``%c0``.

        Access-tile offsets are all-zero while multi-core work division is
        rejected.  The *same* ``%c0`` value, not one constant per axis: when work
        division lands, an axis start becomes that core's slice index times the
        per-core extent, replacing this list element-wise.
        """
        return [self.c0] * rank

    # -- module scaffolding ------------------------------------------------

    @contextlib.contextmanager
    def module(self, kernel_name: str, grid: list[int], params: list) -> Iterator[None]:
        """Open ``module { func.func @kernel_name(params) }`` and emit into it."""
        try:
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                fn = func.FuncOp(kernel_name, ir.FunctionType.get(params, []))
                i64 = ir.IntegerType.get_signless(64)
                fn.attributes["grid"] = ir.ArrayAttr.get(
                    [ir.IntegerAttr.get(i64, int(g)) for g in grid]
                )
                block = fn.add_entry_block()
                self.block_args = list(block.arguments)
                with ir.InsertionPoint(block):
                    self.c0 = self.icst_index(0)
                    yield
                    func.ReturnOp([])
            # Printed while the context is still alive.
            self._text = str(module)
        finally:
            self._stack.close()

    def finish(self) -> str:
        """The canonical MLIR text of the module built by ``module()``."""
        if self._text is None:
            raise AssertionError("KtirBuilder.finish() before module() completed")
        return self._text

    # -- ktdp shapes -------------------------------------------------------

    def memory_view(self, base, entry: BufferEntry):
        """``ktdp.construct_memory_view`` for one buffer, at base address ``base``."""
        sizes = list(entry.sizes)
        strides = _row_major_strides(sizes)
        memref_t = ir.MemRefType.get(sizes, self.elt_type(entry.dtype))
        # No Python builder is exposed for the ``spyre_memory_space`` enum
        # attribute (only the ktdp *types* have getters), so this small enum
        # literal is the one unavoidable textual attribute.
        memory_space = ir.Attribute.parse("#ktdp.spyre_memory_space<HBM>")
        # All extents are static -> empty dynamic size/stride operand lists.
        return self.val(
            ktdp.construct_memory_view(
                memref_t,
                base,
                [],
                [],
                sizes,
                strides,
                memory_space,
                self.coord_set(sizes),
            )
        )

    def bind_view(self, buf_id: str, view) -> None:
        self.views[buf_id] = view

    def view(self, arg: TensorArg):
        return self.views[_buf_id(arg)]

    def access_tile(self, view, sizes: list[int], offsets: list):
        """``ktdp.construct_access_tile`` at ``offsets`` into ``view``."""
        rank = len(sizes)
        at_t = ktdp.AccessTileType.get(sizes, ir.IndexType.get())
        identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))
        return self.val(
            ktdp.construct_access_tile(
                at_t,
                view,
                identity,
                offsets,
                [],
                self.coord_set(sizes),
                identity,
            )
        )

    def operand(self, arg: TensorArg):
        """The value of an input operand: a live produced value, or a fresh load.

        Reusing a produced value is what register-threaded fused intermediates
        will need; they are rejected today, so ``produced`` is always ``None``
        and this always loads.
        """
        produced = self.env.produced(_buf_id(arg))
        return produced if produced is not None else self.load(arg)

    def load(self, arg: TensorArg):
        """An access tile + ``ktdp.load`` for an input ``arg``."""
        sizes = [int(s) for s in arg.device_size]
        tensor_t = ir.RankedTensorType.get(sizes, self.elt_type(arg.device_dtype))
        tile = self.access_tile(self.view(arg), sizes, self.zero_offsets(len(sizes)))
        return self.val(ktdp.load(tensor_t, tile))

    def store_to(self, arg: TensorArg, value) -> None:
        """An access tile + ``ktdp.store`` of ``value`` into output ``arg``."""
        sizes = [int(s) for s in arg.device_size]
        tile = self.access_tile(self.view(arg), sizes, self.zero_offsets(len(sizes)))
        ktdp.store(value, tile)

    def pointwise(self, op: str, ins, out: TensorArg):
        """The compute op for ``op``. Spelled by the address form (see #65)."""
        return self.addrs.pointwise(self, op, ins, out)

    # -- attributes --------------------------------------------------------

    def coord_set(self, sizes: list[int]):
        """Per-dim bounding integer set ``(0 <= d_i <= size_i - 1)`` as an attribute.

        Built with ``ir.IntegerSet`` from ``AffineExpr`` constraints (no textual
        round-trip): for each dim ``i`` two inequalities ``d_i >= 0`` and
        ``-d_i + (size_i - 1) >= 0``, matching the ``affine_set`` MLIR prints.
        """
        exprs = []
        eq_flags: list[bool] = []
        for i, s in enumerate(sizes):
            dim = ir.AffineExpr.get_dim(i)
            # d_i >= 0
            exprs.append(dim)
            eq_flags.append(False)
            # -d_i + (size_i - 1) >= 0
            neg_dim = ir.AffineExpr.get_mul(ir.AffineExpr.get_constant(-1), dim)
            upper = ir.AffineExpr.get_add(
                neg_dim, ir.AffineExpr.get_constant(int(s) - 1)
            )
            exprs.append(upper)
            eq_flags.append(False)
        integer_set = ir.IntegerSet.get(len(sizes), 0, exprs, eq_flags)
        return ir.IntegerSetAttr.get(integer_set)


# ---------------------------------------------------------------------------
# AddressSource: the one seam for dataflow-scheduler#65
# ---------------------------------------------------------------------------
#
# Two members, not four: constant addresses and ``linalg`` named compute must
# revert together.  ``ktdp.load`` needs a *static* memref offset; ``offset: ?``
# persists even with an ``arith.constant`` base and folds to static only when the
# consumer is a ``linalg`` op, and ``linalg`` alone cannot help while the base is
# a func argument.  Splitting the halves would admit configurations that do not
# work, so ``pointwise`` lives here with the address spelling rather than on the
# op registry.


class BakedConstants:
    """TENTATIVE: dataflow-scheduler#65. Zero-arg func, ``arith.constant`` bases
    in elements, ``linalg`` named compute. Delete this class to revert."""

    def __init__(self, table: BufferTable) -> None:
        self.table = table

    def func_param_types(self, b: KtirBuilder) -> list:
        return []

    def bind_views(self, b: KtirBuilder, table: BufferTable) -> None:
        for entry in table.param_entries:
            base = b.icst_index(entry.base_elements)
            b.bind_view(entry.buf_id, b.memory_view(base, entry))

    def pointwise(self, b: KtirBuilder, op: str, ins, out: TensorArg):
        """``linalg`` named op over an uninitialised ``tensor.empty`` out."""
        out_extents = [int(s) for s in out.device_size]
        elt_t = b.elt_type(out.device_dtype)
        tensor_t = ir.RankedTensorType.get(out_extents, elt_t)
        empty = b.val(tensor.EmptyOp(out_extents, elt_t))
        builder = getattr(linalg, REGISTRY[op].linalg_builder)
        return b.val(builder(*ins, outs=[empty], result_tensors=[tensor_t]))


class FuncArgAddresses:
    """The revert target: one ``index`` func arg per buffer, ``arith`` compute."""

    def __init__(self, table: BufferTable) -> None:
        self.table = table

    def func_param_types(self, b: KtirBuilder) -> list:
        return [b.index_t] * len(self.table.param_entries)

    def bind_views(self, b: KtirBuilder, table: BufferTable) -> None:
        for position, entry in enumerate(table.param_entries):
            base = b.block_args[position]
            b.bind_view(entry.buf_id, b.memory_view(base, entry))

    def pointwise(self, b: KtirBuilder, op: str, ins, out: TensorArg):
        return b.val(getattr(arith, REGISTRY[op].arith_builder)(*ins))


AddressSource = BakedConstants | FuncArgAddresses


def address_source(table: BufferTable) -> AddressSource:
    """The address form for this kernel: the single #65 revert seam."""
    return BakedConstants(table) if _addresses_are_baked() else FuncArgAddresses(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_ktir(
    kernel_name: str,
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
) -> str:
    """Build a KTDP-dialect MLIR module for ``specs`` and return ``str(module)``.

    ``specs`` is the finished OpSpec kernel contract (the same value
    ``call_kernel`` passes positionally to ``.run(...)``).  Func parameters are
    the unique operand buffers in ascending ``arg_index`` order so the emitted
    signature matches that positional binding (or, in the baked form, no
    parameters at all and one ``arith.constant`` base address per buffer).
    """
    # Every rejection lives in validate(), and validate() completes before
    # KtirBuilder.create(), so an unsupported request fails fast -- and is
    # testable -- whether or not mlir_ktdp is installed.
    table = validate(specs)
    addrs = address_source(table)
    b = KtirBuilder.create(addrs)
    # Single-core (SENCORES=1) grid; work-division scaling is future work.
    with b.module(kernel_name, grid=[1], params=addrs.func_param_types(b)):
        addrs.bind_views(b, table)
        emit_specs(b, specs)
    return b.finish()
