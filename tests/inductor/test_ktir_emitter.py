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

"""Golden-text snapshot tests for the OpSpec->KTIR emitter (``generate_ktir``).

**Everything in this file needs the ``mlir_ktdp`` dialect build and is skipped
without it.**  The emitter's *rejections* need no dialect -- ``ktir.validate``
raises them all before the lazy import -- so they live in
``test_ktir_validate.py``, which is never skipped and which owns the shared
``_add_op_specs`` fixture.

Self-contained otherwise: no live Inductor graph, no compiler run.
"""

import unittest
from unittest import mock

from test_ktir_validate import _add_op_specs, _baked_add_op_specs

_CONFIG = "torch_spyre._inductor.config"


def _mlir_ktdp_available() -> bool:
    """Whether this build can emit, asked of the emitter rather than guessed.

    The import list belongs to ``KtirBuilder.create``; duplicating it here is how
    the two drift, and a build missing one binding would then error instead of
    skipping.  ``ktir`` imports without a dialect build, so this is safe at
    module scope.
    """
    from torch_spyre._inductor.codegen.ktir import dialect_available

    return dialect_available()


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


# ``bundle_symbolic_args`` is pinned True for the same reason
# ``TestKtirBakedAddresses`` pins it False: _EXPECTED_ADD_KTIR is the symbolic
# form, so leaving it to ambient BUNDLE_SYMBOLIC_ARGS makes the golden fail under
# BUNDLE_SYMBOLIC_ARGS=0 -- which is exactly how the device path is run.
# ``sencores`` is pinned to the single core the emitted grid hardcodes.
@mock.patch(f"{_CONFIG}.bundle_symbolic_args", True)
@mock.patch(f"{_CONFIG}.sencores", 1)
@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestKtirEmitter(unittest.TestCase):
    def test_pointwise_add_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _add_op_specs())
        self.assertEqual(emitted, _EXPECTED_ADD_KTIR)


@mock.patch(f"{_CONFIG}.bundle_symbolic_args", False)
@mock.patch(f"{_CONFIG}.sencores", 1)
@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestKtirBakedAddresses(unittest.TestCase):
    def test_baked_form_deltas(self):
        """The baked form (dataflow-scheduler#65) vs ``_EXPECTED_ADD_KTIR``.

        Asserted as deltas rather than a second golden: the two texts differ in
        5 of 24 lines, so a full copy would be 19 lines of duplication that churn
        together, and this form is deleted outright when #65 is fixed.  The
        loads / tiles / views the two share are already pinned by
        ``_EXPECTED_ADD_KTIR``.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _baked_add_op_specs())

        # 1. No address is a runtime value: zero-arg func, no %arg anywhere.
        self.assertIn("func.func @ktir_fused_add_0() attributes {grid = [1]}", emitted)
        self.assertNotIn("%arg", emitted)
        # 2. Each base is a constant, in ELEMENTS (the byte slot >> 1 for fp16).
        for arg_index in range(3):
            with self.subTest(arg_index=arg_index):
                base = (arg_index << 34) // 2
                self.assertIn(f"arith.constant {base} : index", emitted)
        # 3. linalg over tensor.empty, never arith on tensors -- required for the
        #    memref offset to fold to static, which ktdp.load's verifier needs.
        self.assertIn("tensor.empty()", emitted)
        self.assertIn("linalg.add ins(", emitted)
        self.assertNotIn("arith.addf", emitted)


if __name__ == "__main__":
    unittest.main()
