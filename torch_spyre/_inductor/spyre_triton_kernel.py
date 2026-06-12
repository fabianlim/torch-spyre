# Copyright 2025 The Torch-Spyre Authors.
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

import logging
from typing import Any, Optional, Sequence, Union

import sympy
import torch.utils._pytree as pytree
from torch._inductor.codegen.common import CSEProxy, CSEVariable
from torch._inductor.codegen.triton import (
    FixedTritonConfig,
    TritonKernel,
    TritonOverrides,
)
from torch._inductor.utils import IndentedBuffer, sympy_subs
from torch._inductor.virtualized import StoreMode, V
from torch_spyre._C import DataFormats

from .constants import SPYRE_FP32_OPS
from .errors import Unsupported
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import OpSpec, TensorArg
from .pass_utils import apply_splits_from_index_coeff, iteration_space
from .spyre_kernel import (
    PointwiseOp,
    SpyreOpFuncs,
    TensorAccess,
    UnimplementedOp,
    simplify_op_spec,
)
from .views import compute_coordinates

logger = get_inductor_logger("spyre_triton_kernel")


# ---------------------------------------------------------------------------
# Serialisation helpers — wrappers that __repr__ as Python source so OpSpec
# metadata can be embedded in the generated Triton kernel string.
# ---------------------------------------------------------------------------


class SympyExpr:
    """Wrapper for sympy expressions that serializes to sympify() calls."""

    def __init__(self, expr: sympy.Expr) -> None:
        self.expr = str(expr)

    def __repr__(self) -> str:
        return f"sympify('{self.expr}')"


class IterationSpaceDict:
    """Wrapper for iteration_space dict that serializes properly."""

    def __init__(self, it_space: dict[sympy.Symbol, tuple[sympy.Expr, int]]) -> None:
        self.items = [
            (SympyExpr(k), (SympyExpr(v[0]), v[1])) for k, v in it_space.items()
        ]

    def __repr__(self) -> str:
        items_str = ", ".join(f"{k!r}: ({v[0]!r}, {v[1]})" for k, v in self.items)
        return f"{{{items_str}}}"


class TensorArgDict:
    """Wrapper for TensorArg that serializes properly."""

    def __init__(self, arg: TensorArg) -> None:
        self.is_input = arg.is_input
        self.arg_index = arg.arg_index
        self.device_dtype = arg.device_dtype
        self.device_size = arg.device_size
        self.device_coordinates = [SympyExpr(e) for e in arg.device_coordinates]
        self.allocation = arg.allocation

    def __repr__(self) -> str:
        coords_str = ", ".join(repr(c) for c in self.device_coordinates)
        dtype_str = f"DataFormats.{self.device_dtype.name}"
        return (
            f"TensorArg("
            f"is_input={self.is_input}, "
            f"arg_index={self.arg_index}, "
            f"device_dtype={dtype_str}, "
            f"device_size={self.device_size!r}, "
            f"device_coordinates=[{coords_str}], "
            f"allocation={self.allocation!r})"
        )


class OpSpecDict:
    """Wrapper for OpSpec that serializes properly."""

    def __init__(self, op_spec: OpSpec) -> None:
        self.op = op_spec.op
        self.is_reduction = op_spec.is_reduction
        self.iteration_space = IterationSpaceDict(op_spec.iteration_space)
        self.args = [TensorArgDict(arg) for arg in op_spec.args]
        self.op_info = op_spec.op_info

    def __repr__(self) -> str:
        args_str = ", ".join(repr(arg) for arg in self.args)
        return (
            f"OpSpec("
            f"op={self.op!r}, "
            f"is_reduction={self.is_reduction}, "
            f"iteration_space={self.iteration_space!r}, "
            f"args=[{args_str}], "
            f"op_info={self.op_info!r})"
        )


class UnimplementedOpDict:
    """Wrapper for UnimplementedOp that serializes properly."""

    def __init__(self, op: UnimplementedOp) -> None:
        self.op = op.op

    def __repr__(self) -> str:
        return f"UnimplementedOp(op={self.op!r})"


class TritonOpSpecMapDict:
    """Wrapper for triton_opspec_map that serializes properly."""

    def __init__(self, mapping: dict[str, list[sympy.Symbol]]) -> None:
        self.items = [
            (prefix, [SympyExpr(sym) for sym in symbols])
            for prefix, symbols in mapping.items()
        ]

    def __repr__(self) -> str:
        items_str = ", ".join(
            f"{prefix!r}: [{', '.join(repr(sym) for sym in symbols)}]"
            for prefix, symbols in self.items
        )
        return f"{{{items_str}}}"


# ---------------------------------------------------------------------------
# Ops handler overrides for the Triton path
# ---------------------------------------------------------------------------


class SpyreTritonOverrides(TritonOverrides):
    @staticmethod
    def get_spyre_pointwise(
        name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Optional[PointwiseOp]:
        if kwargs:
            return None
        if not hasattr(SpyreOpFuncs, name):
            return None
        spyre_func = getattr(SpyreOpFuncs, name)
        result = spyre_func(*args)
        return result if isinstance(result, PointwiseOp) else None


class SpyreTritonCSEProxy(CSEProxy):
    def _default(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        result = super()._default(name, args, kwargs)

        if not isinstance(self.kernel, SpyreTritonKernel):
            return result

        spyre_pointwise = SpyreTritonOverrides.get_spyre_pointwise(
            name, args, kwargs
        )
        if spyre_pointwise is None:
            return result

        def record_metadata(v: CSEVariable) -> CSEVariable:
            kernel = self.kernel
            if isinstance(kernel, SpyreTritonKernel):
                kernel.cse_var_to_pointwise[str(v)] = spyre_pointwise
            return v

        return pytree.tree_map(record_metadata, result)


# ---------------------------------------------------------------------------
# SpyreTritonKernel — TritonKernel subclass that emits layout-aware Triton
# code with tl.make_tensor_descriptor calls and Spyre OpSpec metadata.
# ---------------------------------------------------------------------------


class SpyreTritonKernel(TritonKernel):
    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        min_elem_per_thread: int = 0,
        optimize_mask: bool = True,
        fixed_config: Optional[FixedTritonConfig] = None,
        hint_override: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tiling,
            min_elem_per_thread,
            optimize_mask,
            fixed_config,
            hint_override,
            **kwargs,
        )
        self.op_specs: list[Union[OpSpec, UnimplementedOp]] = []
        self.spyre_kernel_args: list[tuple[str, TensorArg]] = []
        # Track loaded tensor args to use in store
        self.loaded_tensor_args: dict[str, TensorArg] = {}
        # Track originating Spyre pointwise op for Triton CSE values
        self.cse_var_to_pointwise: dict[str, PointwiseOp] = {}
        # Mapping from Triton prefixes (x, y, z, r0_, ...) to OpSpec
        # iteration space symbols (c0, c1, ...). Shows which OpSpec
        # dimensions are flattened into each Triton dimension.
        self.triton_opspec_map: dict[str, list[sympy.Symbol]] = {}

    def __enter__(self) -> "SpyreTritonKernel":
        # Skip TritonKernel.__enter__ to install our own CSEProxy
        super(TritonKernel, self).__enter__()
        self.exit_stack.enter_context(
            V.set_ops_handler(SpyreTritonCSEProxy(self, SpyreTritonOverrides()))
        )
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        original_code = super().codegen_kernel(name)

        code = IndentedBuffer()
        code.splice("from torch_spyre._inductor.op_spec import TensorArg, OpSpec")
        code.splice(
            "from torch_spyre._inductor.spyre_kernel import UnimplementedOp"
        )
        code.splice("import torch")
        code.splice(
            "from torch_spyre._C import DataFormats, SpyreTensorLayout"
        )
        code.splice("from sympy import sympify")
        return code.getvalue() + original_code

    def codegen_body(self) -> None:
        if self.triton_meta is not None:
            # Simplify op_specs to align tensors and add singleton dimensions
            for op_spec in self.op_specs:
                if not isinstance(op_spec, UnimplementedOp):
                    simplify_op_spec(op_spec)

            serializable_specs = []
            for op_spec in self.op_specs:
                if isinstance(op_spec, UnimplementedOp):
                    serializable_specs.append(UnimplementedOpDict(op_spec))
                else:
                    serializable_specs.append(OpSpecDict(op_spec))

            serializable_mapping = TritonOpSpecMapDict(self.triton_opspec_map)

            self.triton_meta["spyre_options"] = {
                "op_specs": serializable_specs,
                "triton_opspec_map": serializable_mapping,
            }
        return super().codegen_body()

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """Codegen a load from an InputBuffer and track the TensorAccess."""
        if self.current_node is None:
            raise RuntimeError("current_node is None")

        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if not layout.allocation:
            _ = self.args.input(name)

        # Create TensorArg for this load and store it
        opspec_index = self._get_opspec_index(name, is_load=True)
        tensor_access = TensorAccess(name, opspec_index, layout)
        tensor_arg = self._create_tensor_arg(True, name, tensor_access)
        self.loaded_tensor_args[name] = tensor_arg

        if logger.isEnabledFor(logging.DEBUG):
            triton_is = self.get_triton_iteration_space(index)
            opspec_is = iteration_space(self.current_node)
            logger.debug(
                f"load: name={name} triton_is={triton_is} "
                f"triton_index={index} opspec_is={opspec_is} "
                f"opspec_index={opspec_index}"
            )

        return super().load(name, index)

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: Optional[StoreMode] = None,
    ) -> None:
        """Store and create OpSpec following SpyreKernel pattern."""
        if self.current_node is None:
            raise RuntimeError("current_node is None")

        _ = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)

        opspec_index = self._get_opspec_index(name, is_load=False)
        dst = TensorAccess(name, opspec_index, layout)
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            V.graph.removed_buffers.add(name)

        op_info: dict[str, Any] = {}
        if hasattr(self.current_node, "op_dim_splits"):
            op_info["op_dim_splits"] = self.current_node.op_dim_splits  # type: ignore[union-attr]
        if hasattr(self.current_node, "n_cores_used"):
            op_info["n_cores_used"] = self.current_node.n_cores_used  # type: ignore[union-attr]

        output_tensor_arg = self._create_tensor_arg(False, real_dst_name, dst)

        if logger.isEnabledFor(logging.DEBUG):
            triton_is = self.get_triton_iteration_space(index)
            opspec_is = iteration_space(self.current_node)
            logger.debug(
                f"store: name={name} triton_is={triton_is} "
                f"triton_index={index} opspec_is={opspec_is} "
                f"opspec_index={opspec_index}"
            )

        pointwise = self.cse_var_to_pointwise.get(str(value))
        if pointwise is None:
            raise Unsupported(
                f"Could not recover Spyre pointwise op for Triton value {value}"
            )

        op_info.update(pointwise.op_info)
        self._create_opspec_for_store(
            real_dst_name, output_tensor_arg, pointwise.op, False, op_info
        )

        return super().store(name, index, value, mode)

    def store_reduction(
        self, name: str, index: sympy.Expr, value: CSEVariable
    ) -> None:
        """Store reduction result and create OpSpec following SpyreKernel pattern."""
        if self.current_node is None:
            raise RuntimeError("current_node is None")

        _ = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)

        opspec_index = self._get_opspec_index(name, is_load=False)

        dst = TensorAccess(name, opspec_index, layout)
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            V.graph.removed_buffers.add(name)

        op_info: dict[str, Any] = {}
        if hasattr(self.current_node.node.data, "op_info"):  # type: ignore[union-attr]
            op_info.update(self.current_node.node.data.op_info)  # type: ignore[union-attr]

        data = self.current_node.node.data  # type: ignore[union-attr]
        if not hasattr(data, "reduction_type"):
            raise RuntimeError(
                f"Reduction node missing reduction_type attribute: {data}"
            )

        reduction_type = data.reduction_type
        if reduction_type in ("dot", "matmul", "batchmatmul"):
            reduction_op = "matmul"
        else:
            reduction_op = reduction_type

        output_tensor_arg = self._create_tensor_arg(False, real_dst_name, dst)

        if logger.isEnabledFor(logging.DEBUG):
            triton_is = self.get_triton_iteration_space(index)
            opspec_is = iteration_space(self.current_node)
            logger.debug(
                f"store_reduction: name={name} triton_is={triton_is} "
                f"triton_index={index} opspec_is={opspec_is} "
                f"opspec_index={opspec_index} (from input)"
            )

        self._create_opspec_for_store(
            real_dst_name, output_tensor_arg, reduction_op, True, op_info
        )

        return super().store_reduction(name, index, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_opspec_for_store(
        self,
        real_dst_name: str,
        output_tensor_arg: TensorArg,
        op: str,
        is_reduction: bool,
        op_info: dict[str, Any],
    ) -> None:
        """Create OpSpec for store operations."""
        actuals = self.args.python_argdefs()[1]

        args: list[TensorArg] = []
        for arg_name in actuals:
            if arg_name in self.loaded_tensor_args:
                tensor_arg = self.loaded_tensor_args[arg_name]
                tensor_arg.arg_index = len(args)
                args.append(tensor_arg)
            elif arg_name == real_dst_name:
                output_tensor_arg.arg_index = len(args)
                args.append(output_tensor_arg)

        # Create the Triton-to-OpSpec mapping once per kernel
        if not self.triton_opspec_map:
            self.triton_opspec_map = self._create_triton_opspec_map()
            # Store block size metadata for heuristics access
            spyre_triton_block_size = self._get_triton_block_size()
            assert spyre_triton_block_size is not None
            setattr(
                V.graph, "_spyre_triton_block_size", spyre_triton_block_size
            )
            logger.debug(
                f"Stored spyre_triton_block_size in V.graph: "
                f"{spyre_triton_block_size}"
            )

        op_spec = self._create_op_spec(op, is_reduction, args, op_info)
        self.op_specs.append(op_spec)

    def _get_opspec_index(
        self, name: str, is_load: bool
    ) -> sympy.Expr:
        """Get the index expression from MemoryDep for a specific tensor."""
        assert self.current_node is not None
        deps = (
            self.current_node.read_writes.reads
            if is_load
            else self.current_node.read_writes.writes
        )

        for dep in deps:
            if dep.name == name:
                return dep.index

        raise RuntimeError(
            f"Could not find MemoryDep for "
            f"{'load' if is_load else 'store'} of {name}"
        )

    def get_triton_iteration_space(
        self, index: sympy.Expr
    ) -> dict[str, int]:
        """Extract the Triton iteration space from an index expression."""
        triton_symbols = index.free_symbols

        symbol_coeffs = []
        for sym in triton_symbols:
            if isinstance(sym, sympy.Symbol) and sym in self.range_tree_nodes:
                coeff = index.coeff(sym)
                if coeff is not None:
                    symbol_coeffs.append((sym, coeff))

        symbol_coeffs.sort(
            key=lambda x: V.graph.sizevars.size_hint(x[1]), reverse=True
        )

        triton_is: dict[str, int] = {}
        for sym, _ in symbol_coeffs:
            sym_name = str(sym)
            range_entry = self.range_tree_nodes[sym]
            triton_is[sym_name] = V.graph.sizevars.size_hint(
                range_entry.length
            )

        return triton_is

    def _create_triton_opspec_map(
        self,
    ) -> dict[str, list[sympy.Symbol]]:
        """Create a mapping from Triton iteration space prefixes to
        OpSpec iteration space symbols."""
        assert self.current_node is not None
        it_space = iteration_space(self.current_node)
        opspec_symbols = sorted(it_space.keys(), key=lambda x: str(x))

        triton_prefixes = sorted(
            self.numels.keys(), key=lambda p: (1 if p.startswith("r") else 0, p)
        )

        mapping: dict[str, list[sympy.Symbol]] = {
            prefix: [] for prefix in triton_prefixes
        }

        opspec_size_hints = {
            sym: V.graph.sizevars.size_hint(it_space[sym])
            for sym in opspec_symbols
        }

        triton_size_hints = {
            prefix: V.graph.sizevars.size_hint(self.numels[prefix])
            for prefix in triton_prefixes
        }

        # Check if this is a matmul/dot operation
        is_matmul = False
        if (
            hasattr(self.current_node, "node")
            and self.current_node.node is not None
            and hasattr(self.current_node.node, "data")
        ):
            data = self.current_node.node.data
            if hasattr(data, "reduction_type"):
                is_matmul = data.reduction_type in (
                    "dot",
                    "matmul",
                    "batchmatmul",
                )

        # For matmul/dot, use direct 1:1 mapping
        if is_matmul and len(triton_prefixes) == len(opspec_symbols):
            logger.debug("Using direct 1:1 mapping for matmul/dot operation")
            triton_by_size = sorted(
                triton_prefixes, key=lambda p: triton_size_hints[p]
            )
            opspec_by_size = sorted(
                opspec_symbols, key=lambda s: opspec_size_hints[s]
            )

            for triton_prefix, opspec_sym in zip(triton_by_size, opspec_by_size):
                mapping[triton_prefix] = [opspec_sym]
                logger.debug(
                    f"  {triton_prefix} "
                    f"({triton_size_hints[triton_prefix]}) -> "
                    f"{opspec_sym} ({opspec_size_hints[opspec_sym]})"
                )
        else:
            # Flattening algorithm for non-matmul operations
            opspec_symbols_reversed = list(reversed(opspec_symbols))
            triton_prefixes_reversed = list(reversed(triton_prefixes))

            opspec_idx = 0

            for triton_prefix in triton_prefixes_reversed:
                tnumel = triton_size_hints[triton_prefix]

                if tnumel == 1:
                    continue

                matched_symbols: list[sympy.Symbol] = []
                product = 1

                temp_idx = opspec_idx
                while temp_idx < len(opspec_symbols_reversed):
                    sym = opspec_symbols_reversed[temp_idx]
                    product *= opspec_size_hints[sym]
                    matched_symbols.append(sym)

                    if product == tnumel:
                        mapping[triton_prefix] = list(reversed(matched_symbols))
                        opspec_idx = temp_idx + 1
                        break
                    elif product > tnumel:
                        raise RuntimeError(
                            f"Cannot map Triton dimension '{triton_prefix}' "
                            f"(numel={tnumel}) to OpSpec dimensions. "
                            f"Product {product} exceeds target. "
                            f"Attempted symbols: {matched_symbols}"
                        )

                    temp_idx += 1
                else:
                    raise RuntimeError(
                        f"Cannot map Triton dimension '{triton_prefix}' "
                        f"(numel={tnumel}) to OpSpec dimensions. "
                        f"Accumulated product: {product}, "
                        f"symbols: {matched_symbols}"
                    )

        logger.debug(f"Triton to OpSpec mapping: {mapping}")
        logger.debug(f"Triton iteration space: {triton_size_hints}")
        logger.debug(f"OpSpec iteration space: {opspec_size_hints}")
        return mapping

    def _get_triton_block_size(self) -> dict[str, int]:
        """Extract per-core block sizes from OpSpec iteration space
        via triton_opspec_map."""
        if not self.triton_opspec_map:
            raise RuntimeError(
                "triton_opspec_map is not available - cannot compute block sizes"
            )

        assert self.current_node is not None
        it_space = iteration_space(self.current_node)
        ir_node = self.current_node.node

        core_division: dict[sympy.Symbol, int] = {}
        if hasattr(ir_node, "op_it_space_splits"):
            write_index = next(
                iter(self.current_node.read_writes.writes)
            ).index
            read_index = next(
                iter(self.current_node.read_writes.reads)
            ).index
            core_division = apply_splits_from_index_coeff(
                ir_node.op_it_space_splits,  # type: ignore[attr-defined]
                write_index,
                read_index,
                it_space,
            )
            logger.debug(f"Core division: {core_division}")
        else:
            raise RuntimeError(
                f"ir_node {ir_node} does not have op_it_space_splits"
            )

        spyre_triton_block_size: dict[str, int] = {}

        for triton_prefix, opspec_syms in self.triton_opspec_map.items():
            if not opspec_syms:
                continue

            total_cores = 1
            total_size = 1
            for sym in opspec_syms:
                cores = core_division.get(sym, 1)
                total_cores *= cores
                if sym in it_space:
                    size_expr = it_space[sym]
                    size_val = V.graph.sizevars.size_hint(size_expr)
                    total_size *= size_val

            block_per_core = total_size // total_cores
            spyre_triton_block_size[triton_prefix] = max(1, block_per_core)

        logger.debug(f"spyre_triton_block_size: {spyre_triton_block_size}")
        return spyre_triton_block_size

    def _create_tensor_arg(
        self, is_input: bool, name: str, tensor: TensorAccess
    ) -> TensorArg:
        """Create a TensorArg following the same pattern as SpyreKernel."""
        assert self.current_node is not None
        device_coords = compute_coordinates(
            tensor.layout.device_layout.device_size,  # type: ignore[arg-type]
            tensor.layout.device_layout.stride_map,  # type: ignore[arg-type]
            var_ranges=iteration_space(self.current_node),
            index=tensor.index,
        )
        tensor_arg = TensorArg(
            is_input,
            -1,
            tensor.layout.device_layout.device_dtype,
            tensor.layout.device_layout.device_size,
            device_coords,
            tensor.layout.allocation,
        )
        if not tensor.layout.allocation:
            self.spyre_kernel_args.append((name, tensor_arg))
        return tensor_arg

    def _create_op_spec(
        self,
        op: str,
        is_reduction: bool,
        args: Sequence[TensorArg],
        op_info: dict[str, Any],
    ) -> OpSpec:
        """Create an OpSpec following the same pattern as SpyreKernel."""
        for arg in args:
            if arg.device_dtype == DataFormats.IEEE_FP32 and op not in SPYRE_FP32_OPS:
                raise Unsupported(f"{op} on {arg.device_dtype}")
            elif arg.device_dtype not in (
                DataFormats.IEEE_FP32,
                DataFormats.SEN169_FP16,
            ):
                raise Unsupported(f"operation on {arg.device_dtype}")

        assert self.current_node is not None
        it_space = iteration_space(self.current_node)

        ir_node = self.current_node.node
        core_division: dict[sympy.Symbol, int] = {}
        if hasattr(ir_node, "op_it_space_splits"):
            write_index = next(
                iter(self.current_node.read_writes.writes)
            ).index
            read_index = next(
                iter(self.current_node.read_writes.reads)
            ).index
            core_division = apply_splits_from_index_coeff(
                ir_node.op_it_space_splits,  # type: ignore[attr-defined]
                write_index,
                read_index,
                it_space,
            )

        it_space_extended = {
            k: (v, core_division.get(k, 1)) for k, v in it_space.items()
        }

        return OpSpec(
            op,
            is_reduction,
            it_space_extended,
            args,
            op_info,
        )
