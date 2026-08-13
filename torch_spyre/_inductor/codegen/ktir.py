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

``generate_ktir`` is three steps, in this order:

1. ``validate(specs)`` -- a **pure** recursive walk that raises every
   ``NotImplementedError`` the emitter can raise and returns a ``BufferTable``.
   It imports nothing from ``mlir_ktdp``, so every rejection is reachable and
   testable where the dialect build is absent.
2. ``KtirBuilder.create()`` -- the single ``mlir_ktdp`` import site; owns the
   context and the per-module state.
3. ``emit_specs(b, specs)`` -- a recursive walk over the same spec tree,
   dispatching each ``OpSpec`` through ``KtirBuilder.RECIPES``.

Adding a pointwise op is one ``RECIPES`` entry.  Enabling counted loops is
dropping the ``LoopSpec`` rejection in ``_validate_list`` and filling in
``emit_loop``; the walk already recurses.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.codegen.opspec_utils import (
    _align_reshape_plan,
    _buf_id,
    _row_major_strides,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

# The dialect handles: one module-level name each, None until _load_dialects()
# binds them.  Under TYPE_CHECKING they are the real imports, so `ir.Module` and
# `linalg.add` carry types; at runtime the block does not execute, so importing
# this module requires no dialect build.
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
    """True when the bindings ``_load_dialects`` needs are importable."""
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
# The func signature must be known before the body is emitted, and its parameter
# count depends on the address form, so the buffers
# are collected by the validation walk into a flat table keyed by ``_buf_id``.
# Everything else (strides, block shapes, coordinate bounds) is a pure function
# of ``sizes`` and is recomputed per access by a single helper apiece, rather
# than resolved into records.


def is_internal(arg: TensorArg) -> bool:
    """Whether the kernel produces this buffer only for its own later ops.

    An internal buffer is threaded as an SSA value: not stored, and read back as
    a value rather than loaded.  SDSC says the same thing by giving the buffer an
    LX or HBM-pool allocation, which KTIR does not want -- the scheduler owns
    buffering -- so the KTIR form needs a spec-level flag instead.

    OpSpec does not carry one yet, so this reads a field that does not exist and
    nothing is internal.  When ``TensorArg`` grows the flag, the body becomes
    ``return arg.is_internal`` and no caller changes.
    """
    return bool(getattr(arg, "is_internal", False))


@dataclasses.dataclass(frozen=True)
class BufferEntry:
    """One unique buffer referenced by the kernel."""

    buf_id: str  # opspec_utils._buf_id(arg)
    arg_index: int  # position in the kernel call; -1 => not a kernel argument
    sizes: list[int]  # arg.device_size
    dtype: DataFormats
    base_elements: int | None  # ELEMENTS for the baked form; None => func arg


class BufferTable:
    """Unique buffers in first-seen order, keyed by ``_buf_id``.

    Carries the address form, because it is what resolves each buffer's base.
    """

    def __init__(self, *, bake_addresses: bool = False) -> None:
        self.buffers: dict[str, BufferEntry] = {}
        self.bake_addresses = bake_addresses

    def add(self, arg: TensorArg) -> None:
        """Register ``arg``'s buffer, rejecting what the emitter cannot address."""
        buf_id = _buf_id(arg)
        if buf_id in self.buffers:
            return
        # An internal buffer never reaches memory: no func parameter, no memory
        # view, no address.  It is threaded as a value instead.
        if is_internal(arg):
            return
        # ``arg_index`` stays -1 for buffers the frontend does not pass to the
        # kernel, which today means an LX or HBM-pool allocation.  This emitter
        # constructs HBM memory views only.
        if arg.arg_index < 0:
            raise NotImplementedError(
                f"OpSpec->KTIR: buffer {arg.name!r} is not a kernel argument "
                f"(allocation={arg.allocation!r}); only HBM buffers are supported"
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
                _base_address_elements(arg) if self.bake_addresses else None
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
    *,
    bake_addresses: bool = False,
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
    table = BufferTable(bake_addresses=bake_addresses)
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
        if entry.op not in KtirBuilder.RECIPES:
            raise NotImplementedError(
                f"OpSpec->KTIR: op {entry.op!r} is not supported yet "
                f"(registered: {sorted(KtirBuilder.RECIPES)})"
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
    arity = KtirBuilder.RECIPES[spec.op].arity
    if len(inputs) != arity:
        raise NotImplementedError(
            f"OpSpec->KTIR: {spec.op!r} expects {arity} inputs, got {len(inputs)}"
        )
    return outputs[0], inputs


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
#
# A recipe declares one op: how many inputs it takes, which emission family it
# belongs to, and the dialect builder that implements it.  The recipes live on
# ``KtirBuilder`` beside the family methods that execute them.
#
# ``binding`` returns the builder rather than being it.  The call defers the
# dialect reference to emit time, keeping this module importable without a
# dialect build, and keeps the reference a literal that tooling can resolve.


class Family(enum.Enum):
    """An emission shape.  A recipe declares the one its binding implements;
    ``Family.of`` reads the one a spec asks for."""

    ELEMENTWISE = enum.auto()
    REDUCTION = enum.auto()

    @classmethod
    def of(cls, spec: OpSpec) -> Family:
        """The family ``spec`` asks for, read from the spec rather than the op
        name: the same binding can be wanted in more than one shape."""
        return cls.REDUCTION if spec.is_reduction else cls.ELEMENTWISE


@dataclasses.dataclass(frozen=True)
class Recipe:
    arity: int
    family: Family
    binding: Callable[[], Any]

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError(f"OpSpec->KTIR: arity must be >= 1, got {self.arity}")


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def emit_specs(b: KtirBuilder, specs) -> None:
    """Emit a spec list into the current insertion point. Recursive."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            emit_loop(b, entry)
        elif isinstance(entry, OpSpec):
            _emit_op(b, entry)
        else:
            # validate() already rejected UnimplementedOp and anything else, so
            # reaching here is a validation bug, not an unsupported request.
            # AssertionError (not TypeError) says exactly that.
            raise AssertionError(  # noqa: TRY004
                f"unvalidated spec entry {type(entry).__name__}"
            )


def _emit_op(b: KtirBuilder, spec: OpSpec) -> None:
    """Dispatch one ``OpSpec`` to the builder method for its family."""
    recipe = KtirBuilder.RECIPES[spec.op]
    family = Family.of(spec)
    if family is Family.ELEMENTWISE:
        b.elementwise(spec, recipe)
    else:
        # validate() rejects every family without a builder method, so reaching
        # here is a validation bug rather than an unsupported request.
        raise AssertionError(f"no emission for family {family} of {spec.op!r}")


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

    def __init__(self, stack):
        self._stack = stack
        self.env = ScopeStack()
        # Requires the live context entered by create().
        self.index_t = ir.IndexType.get()
        self.block_args: list = []
        self.views: dict[str, Any] = {}
        self.c0 = None
        self._text: str | None = None

    @classmethod
    def create(cls) -> KtirBuilder:
        """THE single lazy-import site, and the owner of the MLIR context.

        Module level stays ``mlir_ktdp``-free, so ``import ktir`` -- and
        therefore ``validate`` -- works where the dialect build is absent.

        The context is entered here rather than in ``module()`` because
        ``_func_param_types`` builds ``ir`` types and is called before the module
        is opened; ``module()`` closes it on the way out.
        """
        _load_dialects()

        stack = contextlib.ExitStack()
        try:
            ctx = stack.enter_context(ir.Context())
            stack.enter_context(ir.Location.unknown())
            ktdp.register_dialects(ctx)
            return cls(stack)
        except BaseException:
            stack.close()
            raise

    # -- generic helpers ---------------------------------------------------

    @staticmethod
    def val(x):
        """The SSA ``Value`` of a builder result (builders return ``OpView`` or ``Value``)."""
        return x.result if hasattr(x, "result") else x

    @staticmethod
    def elt_type(dtype: DataFormats):
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
                # [] is the result list: a KTIR kernel returns nothing.
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
                    func.ReturnOp([])  # no operands, matching the signature
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
        return self.val(
            ktdp.construct_memory_view(
                result=memref_t,
                offset=base,
                # SSA operands for dynamic extents, of which there are none:
                # every size and stride is a literal, passed as static_* below.
                sizes=[],
                strides=[],
                static_sizes=sizes,
                static_strides=strides,
                memory_space=memory_space,
                coordinate_set=self.coord_set(sizes),
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
                result=at_t,
                base=view,
                # How the view is indexed, and the order of the tile's own axes.
                # Both identity: the tile covers the view one-to-one.
                base_map=identity,
                access_tile_order=identity,
                indices=offsets,
                # SSA operands for symbols in base_map; it uses none.
                symbol_operands=[],
                access_tile_set=self.coord_set(sizes),
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
        return self.val(ktdp.load(result=tensor_t, access_tile=tile))

    def result(self, arg: TensorArg, value) -> None:
        """Dispose of an op's result: thread it, or materialise it.

        The mirror of ``operand`` on the way out.
        """
        if is_internal(arg):
            self.env.bind_produced(_buf_id(arg), value)
        else:
            self.store_to(arg, value)

    def store_to(self, arg: TensorArg, value) -> None:
        """An access tile + ``ktdp.store`` of ``value`` into output ``arg``."""
        sizes = [int(s) for s in arg.device_size]
        tile = self.access_tile(self.view(arg), sizes, self.zero_offsets(len(sizes)))
        ktdp.store(data_tile=value, access_tile=tile)

    # -- compute -----------------------------------------------------------
    #
    # One method per emission family, not per op: the op contributes only its
    # dialect builder, via ``recipe.binding()``.
    #
    # Bindings are dialect *functions*, not OpView classes: ``linalg.AddOp``
    # constructed directly leaves the named op's body region empty and fails
    # verification, while the OpDSL function generates that body.
    #
    # A repeated key here is ruff F601, so an op cannot be declared twice.
    RECIPES: ClassVar[dict[str, Recipe]] = {
        "add": Recipe(arity=2, family=Family.ELEMENTWISE, binding=lambda: linalg.add),
        "mul": Recipe(arity=2, family=Family.ELEMENTWISE, binding=lambda: linalg.mul),
    }

    def elementwise(self, spec: OpSpec, recipe: Recipe) -> None:
        """Load the operands, apply ``recipe``'s builder, store the result.

        Every operand and the result share the output's extents.

        The destination is an uninitialised ``tensor.empty``, valid because an
        elementwise op writes every element of it.  An accumulating op reads its
        destination and needs it filled with an identity instead.
        """
        out, inputs = validated_roles(spec)
        ins = [self.operand(a) for a in inputs]
        extents = [int(x) for x in out.device_size]
        elt_t = self.elt_type(out.device_dtype)
        dest = self.val(tensor.EmptyOp(extents, elt_t))
        result = recipe.binding()(
            *ins,
            outs=[dest],
            result_tensors=[ir.RankedTensorType.get(extents, elt_t)],
        )
        self.result(out, result)

    # -- attributes --------------------------------------------------------

    @staticmethod
    def coord_set(sizes: list[int]):
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
# The two bundle_symbolic_args forms
# ---------------------------------------------------------------------------
#
# ``config.bundle_symbolic_args`` selects where a buffer's base address comes
# from: a func argument (symbolic) or an ``arith.constant`` in elements (baked).
# The SDSC path reads the same flag.
#
# The baked form exists because ``ktdp.load`` requires a static memref offset,
# which a constant base gives only when the consumer is a ``linalg`` op.  It is
# the dataflow-scheduler#65 workaround: deleting the baked arm of the two
# functions below reverts it.


def _func_param_types(b: KtirBuilder, table: BufferTable) -> list:
    """Baked bases need no func arguments; symbolic ones need an index each."""
    if table.bake_addresses:
        return []
    return [b.index_t] * len(table.param_entries)


def _bind_views(b: KtirBuilder, table: BufferTable) -> None:
    """One memory view per buffer, based on a constant or on a func argument."""
    baked = table.bake_addresses
    for position, entry in enumerate(table.param_entries):
        base = b.icst_index(entry.base_elements) if baked else b.block_args[position]
        b.bind_view(entry.buf_id, b.memory_view(base, entry))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_ktir(
    kernel_name: str,
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
    **options,
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
    # Emission options, defaulted here so callers pass only what they need.
    #
    # bake_addresses: emit each base as an arith.constant instead of a func
    # argument.  Canonical KTIR is symbolic; baking is what dbo-opt requires
    # (dataflow-scheduler#65), so the caller that runs dbo-opt asks for it.
    bake_addresses = bool(options.pop("bake_addresses", False))
    if options:
        raise TypeError(f"generate_ktir: unknown option(s) {sorted(options)}")

    table = validate(specs, bake_addresses=bake_addresses)
    b = KtirBuilder.create()
    # Single-core (SENCORES=1) grid; work-division scaling is future work.
    with b.module(kernel_name, grid=[1], params=_func_param_types(b, table)):
        _bind_views(b, table)
        emit_specs(b, specs)
    return b.finish()
