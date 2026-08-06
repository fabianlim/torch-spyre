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

"""Unit tests for the pure OpSpec-reading helpers in ``opspec_utils``.

These helpers (buffer identity, row-major strides, device-dim classification,
reshape/broadcast alignment) are pure sympy/int computations with no MLIR
emission, so this suite runs in CI without ``mlir_ktdp`` installed -- unlike
``test_ktir_emitter`` which pins the emitted MLIR text and skips without it.
"""

import unittest

import sympy
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.opspec_utils import (
    _DIM_BARE,
    _DIM_CONST,
    _DIM_OUTER_STICK,
    _DIM_WITHIN_STICK,
    _align_reshape_plan,
    _buf_id,
    _decompose_work_divisions,
    _device_block_shape,
    _dim_info,
    _iteration_space_key,
    _row_major_strides,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _arg(name, arg_index=0) -> TensorArg:
    """Minimal TensorArg; only ``name`` matters for ``_buf_id``."""
    return TensorArg(
        is_input=True,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[64],
        device_coordinates=[sympy.Symbol("d0")],
        allocation={"hbm": None},
        name=name,
    )


class TestRowMajorStrides(unittest.TestCase):
    def test_rank3(self):
        self.assertEqual(_row_major_strides([16, 512, 64]), [32768, 64, 1])

    def test_rank1(self):
        self.assertEqual(_row_major_strides([64]), [1])

    def test_rank2(self):
        self.assertEqual(_row_major_strides([8, 3]), [3, 1])


class TestBufId(unittest.TestCase):
    def test_name_is_identity(self):
        self.assertEqual(_buf_id(_arg("buf0")), "buf0")

    def test_same_name_same_id_across_input_output(self):
        # An intermediate appearing as both input and output (arg_index sentinel
        # -1 either side) must key on the shared name, not arg_index.
        as_in = _arg("buf7", arg_index=-1)
        as_out = _arg("buf7", arg_index=-1)
        self.assertEqual(_buf_id(as_in), _buf_id(as_out))

    def test_none_name_raises_value_error(self):
        # Missing name is a broken internal invariant, not a capability gap:
        # ValueError, not NotImplementedError.
        with self.assertRaises(ValueError):
            _buf_id(_arg(None))


class TestDimInfo(unittest.TestCase):
    def test_const_dim(self):
        self.assertEqual(_dim_info(sympy.Integer(1)), (_DIM_CONST, None))

    def test_bare_symbol(self):
        d0 = sympy.Symbol("d0")
        self.assertEqual(_dim_info(d0), (_DIM_BARE, d0))

    def test_within_stick(self):
        d0 = sympy.Symbol("d0")
        kind, sym = _dim_info(sympy.Mod(d0, 64))
        self.assertEqual(kind, _DIM_WITHIN_STICK)
        self.assertEqual(sym, d0)

    def test_outer_stick(self):
        d0 = sympy.Symbol("d0")
        kind, sym = _dim_info(FloorDiv(d0, 64))
        self.assertEqual(kind, _DIM_OUTER_STICK)
        self.assertEqual(sym, d0)

    def test_multi_symbol_raises(self):
        d0, d1 = sympy.symbols("d0 d1")
        with self.assertRaises(NotImplementedError):
            _dim_info(d0 * 8 + d1)

    def test_single_symbol_unknown_form_raises(self):
        # A single-symbol coordinate that is neither bare, within-stick, nor
        # outer-stick (e.g. ``2*d0 + 1``) must raise, not fall through.
        d0 = sympy.Symbol("d0")
        with self.assertRaises(NotImplementedError):
            _dim_info(2 * d0 + 1)


class TestIterationSpaceKey(unittest.TestCase):
    def test_order_independent(self):
        d0, d1 = sympy.symbols("d0 d1")
        spec_a = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (16, 1), d1: (512, 1)},
            args=[],
            op_info={},
        )
        spec_b = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d1: (512, 1), d0: (16, 1)},
            args=[],
            op_info={},
        )
        self.assertEqual(_iteration_space_key(spec_a), _iteration_space_key(spec_b))

    def test_distinct_ranges_differ(self):
        d0 = sympy.Symbol("d0")
        spec_a = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (16, 1)},
            args=[],
            op_info={},
        )
        spec_b = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (32, 1)},
            args=[],
            op_info={},
        )
        self.assertNotEqual(_iteration_space_key(spec_a), _iteration_space_key(spec_b))


class TestAlignReshapePlan(unittest.TestCase):
    def test_identity_returns_none(self):
        d0, d1 = sympy.symbols("d0 d1")
        self.assertIsNone(_align_reshape_plan([d0, d1], [16, 64], [d0, d1], [16, 64]))

    def test_broadcast_unmatched_output_axis(self):
        # Input [a, within] aligned into output [a, b, within]: the output's
        # ``b`` axis has no input counterpart, so it reshapes to extent 1 and
        # then broadcasts to the output block.
        a, b, c = sympy.symbols("a b c")
        plan = _align_reshape_plan(
            [a, sympy.Mod(c, 64)],
            [16, 64],
            [a, b, sympy.Mod(c, 64)],
            [16, 8, 64],
        )
        self.assertEqual(plan, ([16, 1, 64], [16, 8, 64]))

    def test_transpose_raises(self):
        # Matched input axes in decreasing order -> would need a permute.
        a, b, c = sympy.symbols("a b c")
        with self.assertRaises(NotImplementedError):
            _align_reshape_plan(
                [b, a, sympy.Mod(c, 64)],
                [2, 3, 64],
                [a, b, sympy.Mod(c, 64)],
                [3, 2, 64],
            )

    def test_dropped_extent_raises(self):
        # An input axis with extent > 1 that no output axis matches would lose
        # data -> needs a cross-stick transpose (restickify), not a reshape.
        a, d, c = sympy.symbols("a d c")
        with self.assertRaises(NotImplementedError):
            _align_reshape_plan(
                [a, sympy.Mod(c, 64)],
                [4, 64],
                [d, sympy.Mod(c, 64)],
                [4, 64],
            )


class TestDecomposeWorkDivisions(unittest.TestCase):
    def test_no_division(self):
        # Every symbol has div == 1: nothing splits, single-core grid.
        d0, d1 = sympy.symbols("d0 d1")
        work, total = _decompose_work_divisions({d0: (16, 1), d1: (512, 1)})
        self.assertEqual(work, [])
        self.assertEqual(total, 1)

    def test_single_divided_symbol(self):
        # One divided axis owns the whole flat grid: inner_cores == 1.
        d0, d1 = sympy.symbols("d0 d1")
        work, total = _decompose_work_divisions({d0: (16, 1), d1: (512, 32)})
        self.assertEqual(work, [(d1, 32, 1)])
        self.assertEqual(total, 32)

    def test_multiple_divided_symbols_mixed_radix(self):
        # Two divided axes: the flat index is mixed-radix, innermost (last in
        # iteration order) first, so d1 gets inner_cores 1 and d0 gets 4.
        d0, d1 = sympy.symbols("d0 d1")
        work, total = _decompose_work_divisions({d0: (16, 2), d1: (512, 4)})
        self.assertEqual(work, [(d1, 4, 1), (d0, 2, 4)])
        self.assertEqual(total, 8)

    def test_undivided_axis_between_divided_is_skipped(self):
        # A div == 1 axis contributes no core portion even when it sits between
        # two divided axes.
        d0, d1, d2 = sympy.symbols("d0 d1 d2")
        work, total = _decompose_work_divisions(
            {d0: (16, 2), d1: (512, 1), d2: (64, 4)}
        )
        self.assertEqual(work, [(d2, 4, 1), (d0, 2, 4)])
        self.assertEqual(total, 8)


def _arg_nd(sizes, coords) -> TensorArg:
    """TensorArg with explicit device sizes/coordinates for block-shape tests."""
    return TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=sizes,
        device_coordinates=coords,
        allocation={"hbm": None},
        name="buf0",
    )


class TestDeviceBlockShape(unittest.TestCase):
    def test_undivided_returns_full_shape(self):
        c0, c1, c2 = sympy.symbols("c0 c1 c2")
        arg = _arg_nd([16, 512, 64], [c0, c1, sympy.Mod(c2, 64)])
        self.assertEqual(_device_block_shape(arg, {}), [16, 512, 64])

    def test_divided_outer_dim(self):
        # c1 divided by 32 -> that dim shrinks; the within-stick last dim and
        # the undivided c0 dim stay full.
        c0, c1, c2 = sympy.symbols("c0 c1 c2")
        arg = _arg_nd([16, 512, 64], [c0, c1, sympy.Mod(c2, 64)])
        self.assertEqual(_device_block_shape(arg, {c1: 32}), [16, 16, 64])

    def test_within_stick_never_divided(self):
        # Even if the last dim's symbol carries a divisor, the within-stick dim
        # stays at its full device size.
        c0, c1, c2 = sympy.symbols("c0 c1 c2")
        arg = _arg_nd([16, 512, 64], [c0, c1, sympy.Mod(c2, 64)])
        self.assertEqual(_device_block_shape(arg, {c2: 2}), [16, 512, 64])

    def test_outer_stick_dim_divided(self):
        # An outer-stick coordinate (c0 // 64) carries one symbol and divides.
        c0, c2 = sympy.symbols("c0 c2")
        arg = _arg_nd([16, 64], [FloorDiv(c0, 64), sympy.Mod(c2, 64)])
        self.assertEqual(_device_block_shape(arg, {c0: 2}), [8, 64])

    def test_constant_axis_left_full(self):
        # A broadcast axis carries a constant coordinate (no symbols): no
        # divisor applies, so it keeps its full size.
        c2 = sympy.Symbol("c2")
        arg = _arg_nd([8, 64], [sympy.Integer(1), sympy.Mod(c2, 64)])
        self.assertEqual(_device_block_shape(arg, {}), [8, 64])

    def test_multi_symbol_axis_raises(self):
        c0, c1, c2 = sympy.symbols("c0 c1 c2")
        arg = _arg_nd([128, 64], [c0 * 8 + c1, sympy.Mod(c2, 64)])
        with self.assertRaises(NotImplementedError):
            _device_block_shape(arg, {c0: 2})


if __name__ == "__main__":
    unittest.main()
