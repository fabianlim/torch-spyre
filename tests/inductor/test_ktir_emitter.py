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

"""Golden-text snapshot test for the OpSpec->KTIR emitter (``generate_ktir``).

Self-contained: it builds the OpSpecs directly (no live Inductor graph, no
compiler run) and asserts the emitter's canonical MLIR text for a single
pointwise ``add``, a fused pointwise chain (register-threaded intermediates),
and a work-divided pointwise ``add`` (multi-core grid).  Skipped only where
``mlir_ktdp`` is not installed.
"""

import unittest

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg


def _mlir_ktdp_available() -> bool:
    """True when mlir_ktdp is built with the func/arith dialect Python bindings."""
    try:
        from mlir_ktdp import ir  # noqa: F401
        from mlir_ktdp.dialects import arith, func, ktdp  # noqa: F401
    except ImportError:
        return False
    return True


# Device shape [16, 512, 64] fp16 (a [512, 1024] logical tensor stickified).
_D0, _D1, _D2 = sympy.symbols("d0 d1 d2")
_COORDS = [_D0, _D1, _D2]
_SIZE = [16, 512, 64]

# Per-arg_index HBM byte addresses handed to the emitter.  Arbitrary and opaque:
# the emitter only scales them to elements, so all that matters is that they are
# distinct and agree with the goldens below (which pin these // 2 for fp16).
_HBM_BYTES = [0x0, 0x400000000, 0x800000000, 0xC00000000]


def _arg(is_input: bool, index: int, name: str) -> TensorArg:
    """A TensorArg at the identity coordinates over the [16, 512, 64] shape.

    ``index < 0`` marks a fused-away intermediate (no assigned arg slot); such a
    buffer is register-threaded rather than materialized to a func parameter, so
    its allocation is never read and stays unassigned.

    Assigned slots get an arbitrary distinct **byte** address from
    ``_HBM_BYTES``.  The emitter treats it as opaque -- it reads
    ``allocation["hbm"]`` and scales it to elements -- so these values only have
    to be distinct and to match the goldens, not to be plausible addresses.
    """
    return TensorArg(
        is_input=is_input,
        arg_index=index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=list(_SIZE),
        device_coordinates=list(_COORDS),
        allocation={"hbm": _HBM_BYTES[index] if index >= 0 else None},
        name=name,
    )


def _it_space(divs=(1, 1, 1)) -> dict:
    """Iteration space over (d0, d1, d2) with per-dim work divisions ``divs``."""
    return {
        _D0: (_SIZE[0], divs[0]),
        _D1: (_SIZE[1], divs[1]),
        _D2: (_SIZE[2], divs[2]),
    }


def _add_op_specs() -> list:
    """Finished OpSpec list for ``a + b`` (two HBM inputs, one HBM output)."""
    return [
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=_it_space(),
            args=[_arg(True, 0, "arg0"), _arg(True, 1, "arg1"), _arg(False, 2, "buf0")],
            op_info={},
        )
    ]


def _fused_chain_op_specs() -> list:
    """``(a + b) + c``: the first add's output is a fused-away intermediate.

    ``buf0`` (arg_index -1) is produced by the first add and consumed by the
    second; it must be register-threaded, never materialized to HBM.
    """
    return [
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=_it_space(),
            args=[
                _arg(True, 0, "arg0"),
                _arg(True, 1, "arg1"),
                _arg(False, -1, "buf0"),
            ],
            op_info={},
        ),
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=_it_space(),
            args=[
                _arg(True, -1, "buf0"),
                _arg(True, 2, "arg2"),
                _arg(False, 3, "buf1"),
            ],
            op_info={},
        ),
    ]


def _fused_add_mul_op_specs() -> list:
    """``(a + b) * (c + d)``: two fused-away intermediates + a ``mul``."""
    return [
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=_it_space(),
            args=[
                _arg(True, 0, "arg0"),
                _arg(True, 1, "arg1"),
                _arg(False, -1, "buf0"),
            ],
            op_info={},
        ),
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=_it_space(),
            args=[
                _arg(True, 2, "arg2"),
                _arg(True, 3, "arg3"),
                _arg(False, -1, "buf1"),
            ],
            op_info={},
        ),
        OpSpec(
            op="mul",
            is_reduction=False,
            iteration_space=_it_space(),
            args=[
                _arg(True, -1, "buf0"),
                _arg(True, -1, "buf1"),
                _arg(False, 4, "buf2"),
            ],
            op_info={},
        ),
    ]


def _work_divided_add_op_specs() -> list:
    """``a + b`` with the d1 axis split across 32 cores (512 // 32 = 16)."""
    return [
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=_it_space((1, 32, 1)),
            args=[_arg(True, 0, "arg0"), _arg(True, 1, "arg1"), _arg(False, 2, "buf0")],
            op_info={},
        )
    ]


# The canonical KTIR text ``generate_ktir`` emits for a single pointwise ``add``
# over a [512, 1024] fp16 tensor stickified to device shape [16, 512, 64].
_EXPECTED_ADD_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_0(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %4 = ktdp.load %3 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %5 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %6 = ktdp.load %5 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %7 = arith.addf %4, %6 : tensor<16x512x64xf16>
    %8 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %7, %8 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""


# ``(a + b) + c``: buf0 threads from the first addf straight into the second
# (%8 feeds %11) -- no memory view / access tile / load / store for it.
_EXPECTED_FUSED_CHAIN_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_add_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.construct_memory_view %arg3, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %4 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %5 = ktdp.load %4 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %6 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %7 = ktdp.load %6 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %8 = arith.addf %5, %7 : tensor<16x512x64xf16>
    %9 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %10 = ktdp.load %9 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %11 = arith.addf %8, %10 : tensor<16x512x64xf16>
    %12 = ktdp.construct_access_tile %3[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %11, %12 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""


# ``(a + b) * (c + d)``: buf0 (%9) and buf1 (%14) both thread into the mulf
# (%15) with no materialization; only buf2 (%arg4) is stored.
_EXPECTED_FUSED_ADD_MUL_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_add_mul_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.construct_memory_view %arg3, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %4 = ktdp.construct_memory_view %arg4, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %5 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %6 = ktdp.load %5 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %7 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %8 = ktdp.load %7 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %9 = arith.addf %6, %8 : tensor<16x512x64xf16>
    %10 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %11 = ktdp.load %10 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %12 = ktdp.construct_access_tile %3[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %13 = ktdp.load %12 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %14 = arith.addf %11, %13 : tensor<16x512x64xf16>
    %15 = arith.mulf %9, %14 : tensor<16x512x64xf16>
    %16 = ktdp.construct_access_tile %4[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %15, %16 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""


# ``a + b`` split over 32 cores on d1: grid = [32], a flat get_compute_tile_id,
# per-core base ``id * 16`` on the d1 axis, per-core access tiles of shape
# [16, 16, 64] under the shrunk #set1; the memory views stay full [16, 512, 64].
_EXPECTED_WORK_DIVIDED_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_0(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [32]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.get_compute_tile_id : index
    %c16 = arith.constant 16 : index
    %4 = arith.muli %3, %c16 : index
    %5 = ktdp.construct_access_tile %0[%c0, %4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<16x512x64xf16> -> !ktdp.access_tile<16x16x64xindex>
    %6 = ktdp.load %5 : <16x16x64xindex> -> tensor<16x16x64xf16>
    %c16_0 = arith.constant 16 : index
    %7 = arith.muli %3, %c16_0 : index
    %8 = ktdp.construct_access_tile %1[%c0, %7, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<16x512x64xf16> -> !ktdp.access_tile<16x16x64xindex>
    %9 = ktdp.load %8 : <16x16x64xindex> -> tensor<16x16x64xf16>
    %10 = arith.addf %6, %9 : tensor<16x16x64xf16>
    %c16_1 = arith.constant 16 : index
    %11 = arith.muli %3, %c16_1 : index
    %12 = ktdp.construct_access_tile %2[%c0, %11, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<16x512x64xf16> -> !ktdp.access_tile<16x16x64xindex>
    ktdp.store %10, %12 : tensor<16x16x64xf16>, <16x16x64xindex>
    return
  }
}
"""


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with func/arith dialect bindings is not installed",
)
class TestKtirEmitter(unittest.TestCase):
    def setUp(self):
        # The emitter bakes literal addresses, so it only accepts the
        # non-symbolic path; on the symbolic path allocation["hbm"] is a
        # sentinel arg_index rather than an address.  The default is symbolic,
        # so every emitting test has to opt out (see
        # test_symbolic_args_unsupported for the guard itself).
        patcher = _spyre_config.patch(bundle_symbolic_args=False)
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)

    def test_pointwise_add_golden(self):
        """Single pointwise add, single core -- the pointwise-PR baseline."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _add_op_specs())
        self.assertEqual(emitted, _EXPECTED_ADD_KTIR)

    def test_fused_chain_golden(self):
        """``(a + b) + c``: the intermediate is register-threaded, not stored."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_add_0", _fused_chain_op_specs())
        self.assertEqual(emitted, _EXPECTED_FUSED_CHAIN_KTIR)

    def test_fused_add_mul_golden(self):
        """``(a + b) * (c + d)``: two threaded intermediates + a ``mul``."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_add_mul_0", _fused_add_mul_op_specs())
        self.assertEqual(emitted, _EXPECTED_FUSED_ADD_MUL_KTIR)

    def test_work_divided_add_golden(self):
        """Pointwise add split across 32 cores on the d1 axis."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _work_divided_add_op_specs())
        self.assertEqual(emitted, _EXPECTED_WORK_DIVIDED_KTIR)

    def test_reduction_unsupported(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        specs[0].is_reduction = True
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_add_0", specs)

    def test_symbolic_args_unsupported(self):
        """The symbolic path must be refused, not silently miscompiled.

        There ``allocation["hbm"]`` holds a sentinel ``arg_index`` (the real
        address is substituted at launch), so baking it would emit addresses of
        0/0/1.  Nothing downstream would catch that -- the compute path
        validates neither the slot index nor the offset -- so the kernel would
        run and return wrong data.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        with _spyre_config.patch(bundle_symbolic_args=True):
            with self.assertRaises(NotImplementedError) as cm:
                generate_ktir("ktir_fused_add_0", _add_op_specs())
        self.assertIn("literal address path", str(cm.exception))

    def test_unassigned_address_rejected(self):
        """An unplanned buffer must raise rather than bake ``None``."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        specs[0].args[1].allocation = {"hbm": None}
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_add_0", specs)

    def test_non_hbm_allocation_rejected(self):
        """LX / pool buffers must raise: every emitted view hardcodes HBM."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        for allocation in ({"lx": 0x1000}, {"hbm_pool": 0x2000}):
            with self.subTest(allocation=allocation):
                specs = _add_op_specs()
                specs[0].args[1].allocation = dict(allocation)
                with self.assertRaises(NotImplementedError):
                    generate_ktir("ktir_fused_add_0", specs)

    def test_unsupported_op(self):
        """A pointwise op outside the wired-up set (e.g. ``sub``) must raise."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        specs[0].op = "sub"
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_sub_0", specs)

    def test_loopspec_unsupported(self):
        """A counted loop (``LoopSpec``) is out of scope and must raise."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = [LoopSpec(count=4, body=_add_op_specs())]
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_loop_0", specs)


if __name__ == "__main__":
    unittest.main()
