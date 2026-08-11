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

Self-contained: it builds the ``add`` OpSpec directly (no live Inductor graph,
no compiler run) and asserts the emitter's canonical MLIR text.
Skipped only where ``mlir_ktdp`` is not installed.
"""

import unittest
from unittest import mock

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _mlir_ktdp_available() -> bool:
    """True when mlir_ktdp is built with the func/arith dialect Python bindings."""
    try:
        from mlir_ktdp import ir  # noqa: F401
        from mlir_ktdp.dialects import arith, func, ktdp  # noqa: F401
    except ImportError:
        return False
    return True


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


def _add_op_specs() -> list:
    """Finished OpSpec list for ``a + b`` at device shape [16, 512, 64] fp16.

    This mirrors what the SuperDSC frontend produces for a pointwise ``a + b``:
    two HBM inputs and one HBM output, each addressed at the identity
    coordinates ``(d0, d1, d2)`` over the stickified device shape.
    """
    d0, d1, d2 = sympy.symbols("d0 d1 d2")
    coords = [d0, d1, d2]
    size = [16, 512, 64]

    def arg(is_input: bool, index: int, name: str) -> TensorArg:
        return TensorArg(
            is_input=is_input,
            arg_index=index,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(size),
            device_coordinates=list(coords),
            allocation={"hbm": None},
            name=name,
        )

    return [
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (16, 1), d1: (512, 1), d2: (64, 1)},
            args=[
                arg(True, 0, "arg0"),
                arg(True, 1, "arg1"),
                arg(False, 2, "buf0"),
            ],
            op_info={},
        )
    ]


# The emitter only supports the single-core (SENCORES=1) grid so far; pin it so
# these tests exercise their intended guards rather than the multi-core guard,
# which would otherwise fire first on the default SENCORES=32.
@mock.patch("torch_spyre._inductor.config.sencores", 1)
@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with func/arith dialect bindings is not installed",
)
class TestKtirEmitter(unittest.TestCase):
    def test_pointwise_add_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _add_op_specs())
        self.assertEqual(emitted, _EXPECTED_ADD_KTIR)

    def test_reduction_unsupported(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        specs[0].is_reduction = True
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_add_0", specs)

    def test_non_add_unsupported(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        specs[0].op = "mul"
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_mul_0", specs)


class TestKtirCapabilityGuards(unittest.TestCase):
    """Guards that fire before the mlir_ktdp import, so they need no dialect."""

    @mock.patch("torch_spyre._inductor.config.sencores", 2)
    def test_multicore_unsupported(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_add_0", _add_op_specs())


# ---------------------------------------------------------------------------
# The baked-constant address form (config.bundle_symbolic_args = False)
# ---------------------------------------------------------------------------
#
# Everything above exercises the default, symbolic form, where a base address is
# a func argument and ``allocation["hbm"]`` is a sentinel the emitter never
# reads.  The tests below pin the other form, the one the dbo backend requires
# (dataflow-scheduler#65): a zero-arg func with each base address materialised as
# an ``arith.constant``.


def _mlir_ktdp_linalg_available() -> bool:
    """True when the linalg/tensor bindings the baked form needs are present too.

    Separate from ``_mlir_ktdp_available`` on purpose: only the baked form emits
    a linalg named op, so the symbolic goldens must not start skipping in an
    environment that has func/arith but not linalg.
    """
    if not _mlir_ktdp_available():
        return False
    try:
        from mlir_ktdp.dialects import linalg, tensor  # noqa: F401
    except ImportError:
        return False
    return True


# One HBM segment slot per argument (``slot << 34``), as the literal BYTE offsets
# memory planning resolves into ``allocation["hbm"]`` on the non-symbolic path.
# Written out here rather than imported from the table the emitter reads, so the
# element addresses in the golden below are an independent statement of what the
# emitter must produce rather than a restatement of its own input.
_HBM_BYTE_ADDRESS = [0x0, 0x400000000, 0x800000000]


def _add_op_specs_baked() -> list:
    """``_add_op_specs`` with each buffer's HBM base address already resolved.

    The symbolic form ignores these (its ``allocation["hbm"]`` is a sentinel
    ``arg_index``); the baked form bakes them in.
    """
    specs = _add_op_specs()
    for arg in specs[0].args:
        arg.allocation = {"hbm": _HBM_BYTE_ADDRESS[arg.arg_index]}
    return specs


# The same add as ``_EXPECTED_ADD_KTIR``, in the baked form.  Two differences,
# both load-bearing for the backend rather than cosmetic:
#
#   * the func takes NO arguments and each memory view is rooted at an
#     ``arith.constant``, because address assignment requires
#     compile-time-constant HBM addresses (dataflow-scheduler#65);
#   * the compute op is a ``linalg`` named op over a ``tensor.empty`` out, not
#     ``arith.addf`` on tensors, because only a linalg consumer makes the
#     three-stage-pipeline pass rewrite ``ktdp.load`` into a FIFO read.
#
# The constants are the ``_HBM_BYTE_ADDRESS`` slots in ELEMENTS -- byte offset / 2
# for fp16: 0x0 -> 0, 0x400000000 -> 8589934592, 0x800000000 -> 17179869184.
_EXPECTED_ADD_KTIR_BAKED = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_0() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %c0_0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %c8589934592 = arith.constant 8589934592 : index
    %1 = ktdp.construct_memory_view %c8589934592, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %c17179869184 = arith.constant 17179869184 : index
    %2 = ktdp.construct_memory_view %c17179869184, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %4 = ktdp.load %3 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %5 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %6 = ktdp.load %5 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %7 = tensor.empty() : tensor<16x512x64xf16>
    %8 = linalg.add ins(%4, %6 : tensor<16x512x64xf16>, tensor<16x512x64xf16>) outs(%7 : tensor<16x512x64xf16>) -> tensor<16x512x64xf16>
    %9 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %8, %9 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""


@mock.patch("torch_spyre._inductor.config.bundle_symbolic_args", False)
@mock.patch("torch_spyre._inductor.config.sencores", 1)
@unittest.skipUnless(
    _mlir_ktdp_linalg_available(),
    "mlir_ktdp with func/arith/linalg/tensor dialect bindings is not installed",
)
class TestKtirBakedAddresses(unittest.TestCase):
    def test_pointwise_add_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _add_op_specs_baked())
        self.assertEqual(emitted, _EXPECTED_ADD_KTIR_BAKED)


@mock.patch("torch_spyre._inductor.config.bundle_symbolic_args", True)
@mock.patch("torch_spyre._inductor.config.sencores", 1)
@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with func/arith dialect bindings is not installed",
)
class TestKtirSymbolicAddresses(unittest.TestCase):
    def test_resolved_addresses_are_ignored(self):
        """The symbolic form emits func args even when real addresses exist.

        Same assertion as ``TestKtirEmitter.test_pointwise_add_golden`` (whose
        fixture carries no address at all), but with the flag pinned rather than
        defaulted, so it holds regardless of BUNDLE_SYMBOLIC_ARGS in the
        environment.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _add_op_specs_baked())
        self.assertEqual(emitted, _EXPECTED_ADD_KTIR)


class TestKtirBakedAddressGuards(unittest.TestCase):
    """``_base_address_elements``: the one place an HBM address is resolved.

    Exercised directly (not through ``generate_ktir``) so these run without the
    dialect build -- the whole point of resolving addresses in a pure helper.
    None of them applies to the symbolic form, which never calls it.
    """

    @staticmethod
    def _arg(allocation):
        arg = _add_op_specs()[0].args[1]
        arg.allocation = allocation
        return arg

    def test_byte_offset_is_scaled_to_elements(self):
        from torch_spyre._inductor.codegen.ktir import _base_address_elements

        # fp16: 2 bytes per element, so the element address is half the byte one.
        self.assertEqual(
            _base_address_elements(self._arg({"hbm": 0x400000000})), 0x200000000
        )
        # A zero address is a real address, not "unset".
        self.assertEqual(_base_address_elements(self._arg({"hbm": 0})), 0)

    def test_unassigned_address_refused(self):
        from torch_spyre._inductor.codegen.ktir import _base_address_elements

        with self.assertRaises(NotImplementedError):
            _base_address_elements(self._arg({"hbm": None}))

    def test_misaligned_address_refused(self):
        from torch_spyre._inductor.codegen.ktir import _base_address_elements

        # An odd byte offset cannot be expressed as an fp16 element index.
        with self.assertRaises(NotImplementedError):
            _base_address_elements(self._arg({"hbm": 0x401}))

    def test_non_hbm_allocations_refused(self):
        from torch_spyre._inductor.codegen.ktir import _base_address_elements

        # Every emitted memory view hardcodes memory_space = HBM, so a buffer
        # living anywhere else must be rejected rather than mislabelled.
        for allocation in ({"lx": 0x1000}, {"hbm_pool": 0x1000}, {}):
            with (
                self.subTest(allocation=allocation),
                self.assertRaises(NotImplementedError),
            ):
                _base_address_elements(self._arg(allocation))


if __name__ == "__main__":
    unittest.main()
