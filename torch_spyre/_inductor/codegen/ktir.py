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
"""

from __future__ import annotations

from collections.abc import Sequence

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.codegen.opspec_utils import (
    _align_reshape_plan,
    _buf_id,
    _row_major_strides,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

# Pointwise op name -> the ``linalg`` named-op builder that implements it.  Only
# ``add`` is wired up so far; other ops raise before reaching here.
#
# These must be ``linalg`` *named ops*, not ``arith`` scalar-on-tensor ops:
# the backend's ``construct-three-stage-pipeline`` only rewrites a ``ktdp.load``
# into a ``ktdf.read_from_fifo`` when the load's consumer is a linalg op.  With
# ``arith.addf`` on tensors the load survives that rewrite, and its
# ``memref<..., offset: ?>`` operand then fails the verifier.
_LINALG_FLOAT_OP = {"add": "add"}


def _val(x):
    """The SSA ``Value`` of a builder result (builders return ``OpView`` or ``Value``)."""
    return x.result if hasattr(x, "result") else x


def _base_address_elements(arg: TensorArg) -> int:
    """``arg``'s buffer base address in ELEMENTS, for ``ktdp.construct_memory_view``.

    The emitter bakes base addresses as ``arith.constant`` rather than taking
    them as func arguments because the backend's address-assignment pass
    requires compile-time-constant HBM addresses (dataflow-scheduler#65): a
    symbolic base makes ``construct_memory_view`` produce
    ``memref<..., offset: ?>``, which ``ktdp.load``'s verifier rejects.

    The address is read from ``allocation["hbm"]`` -- the same field the SDSC
    path resolves into the bundle's start address (``superdsc.py:774`` ->
    ``startAddressCoreCorelet_`` at ``compute_ops.py:1306``).  It is NOT
    recomputed as ``SEGMENT_OFFSETS[arg.arg_index]``: an HBM pool takes slot 0
    and shifts every tensor arg up by one (``spyre_kernel.py:1235``), so
    ``arg_index`` would address one slot low whenever a pool exists.

    Three things this must get right, each of them silent if it does not:

    * ``allocation`` is a tagged union (see ``TensorArg.allocation``), so an
      ``lx`` / ``hbm_pool`` buffer is rejected outright -- every view this
      emitter builds hardcodes ``memory_space = HBM``.
    * ``config.bundle_symbolic_args`` decides the *units* of the field, so it is
      rejected here; this is the flag's only semantic consumer in the emitter.
      For a 3-buffer add the two paths give (measured)::

          literal   {'hbm': 0}  {'hbm': 17179869184}  {'hbm': 34359738368}
          symbolic  {'hbm': 0}  {'hbm': 1}            {'hbm': 2}

      The literal values are byte offsets in the device virtual address space,
      one segment slot per argument (``slot << 34``).  The symbolic ones are a
      bare sentinel ``arg_index`` rebound at launch; baking those would emit
      base addresses 0/0/1 and read garbage.
    * a memref base offset indexes the *element* type, so a byte offset must be
      scaled down by the element size.  Emitting the raw byte offset compiles
      and runs, but addresses 2x too high for fp16, with no diagnostic anywhere.
    """
    # Units, hence correctness, depend on the flag -- see above.  config.py
    # forces the flag (and the BUNDLE_SYMBOLIC_ARGS env var) off whenever the
    # KTIR emitter is selected; this is the safety net for any other route in.
    if _spyre_config.bundle_symbolic_args:
        raise NotImplementedError(
            "OpSpec->KTIR: requires the literal address path, but "
            "bundle_symbolic_args is set, so allocation['hbm'] holds a "
            "sentinel arg_index rather than an address. Set "
            "BUNDLE_SYMBOLIC_ARGS=0 (prepare_kernel.cpp reads that env var "
            "directly, so the env var itself must be 0, not just the config "
            "flag)."
        )

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

    elem_bytes = num_bytes(arg.device_dtype)
    if int(byte_offset) % elem_bytes:
        raise NotImplementedError(
            f"OpSpec->KTIR: HBM offset {int(byte_offset):#x} for buffer "
            f"{arg.name!r} is not a multiple of the {elem_bytes}-byte element size"
        )
    return int(byte_offset) // elem_bytes


def _mlir_elt_type(ir, device_dtype: DataFormats):
    """The ``mlir_ktdp.ir`` element type for a Spyre device dtype.

    ``ir`` is the lazily-imported ``mlir_ktdp.ir`` module (the file never
    imports it at top level, so it stays importable where the dialect build is
    absent).  The two fp16 device formats both map to ``f16``; extend this map
    (never fall through silently) as new dtypes are supported.
    """
    # Direct type-builder references (not name strings) resolved here, where
    # ``ir`` is in scope.
    mapping = {
        DataFormats.IEEE_FP16: ir.F16Type,
        DataFormats.SEN169_FP16: ir.F16Type,
        DataFormats.IEEE_FP32: ir.F32Type,
        DataFormats.BFLOAT16: ir.BF16Type,
    }
    builder = mapping.get(device_dtype)
    if builder is None:
        raise NotImplementedError(
            f"OpSpec->KTIR: unsupported device dtype {device_dtype!r}"
        )
    return builder.get()


def generate_ktir(
    kernel_name: str,
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
) -> str:
    """Build a KTDP-dialect MLIR module for ``specs`` and return ``str(module)``.

    ``specs`` is the finished OpSpec kernel contract (the same value
    ``call_kernel`` passes positionally to ``.run(...)``).  The emitted
    ``func.func`` takes no arguments: every buffer base address is materialised
    as a constant from its own allocation (see ``_base_address_elements``).
    Buffers are still walked in ascending ``arg_index`` order, so the emitted
    memory views appear in the same order the runtime binds arguments.
    """
    # Pure capability checks first, before the mlir_ktdp import: they need no
    # dialect build, so an unsupported request fails fast (and is testable)
    # whether or not mlir_ktdp is installed.
    #
    # Multi-core work division is future work; the grid below is hard-coded to a
    # single core, so reject anything else rather than silently emitting a
    # single-core grid on a multi-core request.
    if _spyre_config.sencores != 1:
        raise NotImplementedError(
            "OpSpec->KTIR: multi-core work division is not supported yet "
            f"(SENCORES={_spyre_config.sencores}, only 1 is supported)"
        )
    op_specs = _collect_pointwise_op_specs(specs)

    # ``mlir_ktdp`` is imported lazily so the module stays importable (and the
    # golden test can skip) where the dialect-packaged mlir_ktdp is not built.
    from mlir_ktdp import ir
    from mlir_ktdp.dialects import arith, func, ktdp, linalg, tensor

    # Ordered unique operand buffers.  Ascending arg_index matches the
    # positional order call_kernel passes to .run(...), which keeps the emitted
    # memory views in argument order.  Only real external buffers
    # (arg_index >= 0) get a view; register-threaded fused intermediates carry
    # the -1 sentinel and are rejected below as unsupported.
    ordered_args: dict[object, TensorArg] = {}
    for spec in op_specs:
        for arg in spec.args:
            ordered_args.setdefault(_buf_id(arg), arg)
    buffer_args = sorted(
        (a for a in ordered_args.values() if a.arg_index >= 0),
        key=lambda a: a.arg_index,
    )

    with ir.Context() as ctx, ir.Location.unknown():
        ktdp.register_dialects(ctx)
        index_t = ir.IndexType.get()

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            # Zero-arg func: base addresses are baked constants, not parameters,
            # because the backend requires constant HBM addresses
            # (dataflow-scheduler#65).  See _base_address_elements.
            fn_type = ir.FunctionType.get([], [])
            fn = func.FuncOp(kernel_name, fn_type)
            # Single-core (SENCORES=1) grid; work-division scaling is future work.
            i64 = ir.IntegerType.get_signless(64)
            fn.attributes["grid"] = ir.ArrayAttr.get([ir.IntegerAttr.get(i64, 1)])
            block = fn.add_entry_block()

            with ir.InsertionPoint(block):
                c0 = arith.ConstantOp(index_t, 0)

                # One memory view per unique buffer, in argument order, each
                # rooted at a constant base address resolved from that buffer's
                # own allocation.
                memory_views: dict[object, ir.Value] = {}
                for arg in buffer_args:
                    base = arith.ConstantOp(index_t, _base_address_elements(arg))
                    memory_views[_buf_id(arg)] = _emit_memory_view(
                        ir, ktdp, arg, _val(base)
                    )

                for spec in op_specs:
                    _emit_pointwise_op(ir, ktdp, linalg, tensor, spec, memory_views, c0)

                func.ReturnOp([])

        return str(module)


def _collect_pointwise_op_specs(
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
) -> list[OpSpec]:
    """Validate ``specs`` and return the flat list of pointwise ``OpSpec``s.

    Rejects everything outside the supported scope with an explicit
    ``NotImplementedError``.
    """
    op_specs: list[OpSpec] = []
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            raise NotImplementedError(f"OpSpec->KTIR: unimplemented op {entry.op!r}")
        if isinstance(entry, LoopSpec):
            raise NotImplementedError(
                "OpSpec->KTIR: counted loops (LoopSpec) are not supported yet"
            )
        if not isinstance(entry, OpSpec):
            raise NotImplementedError(
                f"OpSpec->KTIR: unexpected spec entry {type(entry).__name__}"
            )
        if entry.is_reduction:
            raise NotImplementedError("OpSpec->KTIR: reductions are not supported yet")
        if entry.op not in _LINALG_FLOAT_OP:
            raise NotImplementedError(
                f"OpSpec->KTIR: op {entry.op!r} is not supported yet "
                "(only pointwise 'add')"
            )
        op_specs.append(entry)
    if not op_specs:
        raise NotImplementedError("OpSpec->KTIR: no OpSpec to emit")
    return op_specs


def _emit_memory_view(ir, ktdp, arg: TensorArg, offset):
    """Emit ``ktdp.construct_memory_view`` for one buffer, return its SSA value."""
    sizes = [int(s) for s in arg.device_size]
    strides = _row_major_strides(sizes)
    memref_t = ir.MemRefType.get(sizes, _mlir_elt_type(ir, arg.device_dtype))
    coord_set = _coordinate_set_attr(ir, sizes)
    # No Python builder is exposed for the ``spyre_memory_space`` enum attribute
    # (only the ktdp *types* have getters), so this small enum literal is the one
    # unavoidable textual attribute.
    memory_space = ir.Attribute.parse("#ktdp.spyre_memory_space<HBM>")
    # All extents are static -> empty dynamic size/stride operand lists.
    return ktdp.construct_memory_view(
        memref_t,
        offset,
        [],
        [],
        sizes,
        strides,
        memory_space,
        coord_set,
    )


def _emit_pointwise_op(ir, ktdp, linalg, tensor, spec: OpSpec, memory_views, c0):
    """Emit the load / compute / store sequence for one pointwise ``OpSpec``."""
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(outputs) != 1:
        raise NotImplementedError(
            f"OpSpec->KTIR: expected exactly one output, got {len(outputs)}"
        )
    if len(inputs) != 2:
        raise NotImplementedError(
            f"OpSpec->KTIR: 'add' expects two inputs, got {len(inputs)}"
        )
    out = outputs[0]

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

    # Every operand buffer must be a func parameter (register-threaded fused
    # intermediates are not supported yet).
    for arg in spec.args:
        if _buf_id(arg) not in memory_views:
            raise NotImplementedError(
                "OpSpec->KTIR: fused intermediates (register threading) "
                "not supported yet"
            )

    loaded = [
        _emit_load(ir, ktdp, arg, memory_views[_buf_id(arg)], c0) for arg in inputs
    ]

    # linalg named op over an (uninitialised) tensor.empty out; a scalar arith
    # op on tensors would leave the ktdp.load unlowered downstream (see
    # _LINALG_FLOAT_OP).
    elt_t = _mlir_elt_type(ir, out.device_dtype)
    tensor_t = ir.RankedTensorType.get(out_extents, elt_t)
    empty = _val(tensor.EmptyOp(out_extents, elt_t))
    builder = getattr(linalg, _LINALG_FLOAT_OP[spec.op])
    result = _val(builder(*loaded, outs=[empty], result_tensors=[tensor_t]))

    _emit_store(ir, ktdp, out, memory_views[_buf_id(out)], result, c0)


def _emit_access_tile(ir, ktdp, arg: TensorArg, memory_view, c0):
    """Emit ``ktdp.construct_access_tile`` for ``arg``, return its SSA value."""
    sizes = [int(s) for s in arg.device_size]
    rank = len(sizes)
    at_t = ktdp.AccessTileType.get(sizes, ir.IndexType.get())
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))
    tile_set = _coordinate_set_attr(ir, sizes)
    return ktdp.construct_access_tile(
        at_t,
        memory_view,
        identity,
        [c0] * rank,
        [],
        tile_set,
        identity,
    )


def _emit_load(ir, ktdp, arg: TensorArg, memory_view, c0):
    """Emit an access tile + ``ktdp.load`` for an input ``arg``."""
    sizes = [int(s) for s in arg.device_size]
    tensor_t = ir.RankedTensorType.get(sizes, _mlir_elt_type(ir, arg.device_dtype))
    tile = _emit_access_tile(ir, ktdp, arg, memory_view, c0)
    return ktdp.load(tensor_t, tile)


def _emit_store(ir, ktdp, arg: TensorArg, memory_view, value, c0):
    """Emit an access tile + ``ktdp.store`` of ``value`` into output ``arg``."""
    tile = _emit_access_tile(ir, ktdp, arg, memory_view, c0)
    ktdp.store(value, tile)


# ---------------------------------------------------------------------------
# Attribute builders
# ---------------------------------------------------------------------------


def _coordinate_set_attr(ir, sizes: list[int]):
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
        upper = ir.AffineExpr.get_add(neg_dim, ir.AffineExpr.get_constant(int(s) - 1))
        exprs.append(upper)
        eq_flags.append(False)
    integer_set = ir.IntegerSet.get(len(sizes), 0, exprs, eq_flags)
    return ir.IntegerSetAttr.get(integer_set)
