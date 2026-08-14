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
without it.**  The emitter's *rejections* need no dialect -- the plan walk
raises them all before the lazy import -- so they live in
``test_ktir_validate.py``, which is never skipped and which owns the shared
``_add_op_specs`` fixture.

Self-contained otherwise: no live Inductor graph, no compiler run.
"""

import unittest

from test_ktir_validate import _add_op_specs, _baked_add_op_specs

from torch_spyre._inductor.codegen.ktir import PlanOptions

# One core, and a plan walk that descends a LoopSpec instead of refusing it:
# what the tiled-nest tests need and nothing else asks for.  ``ktir`` imports
# without a dialect build, so naming PlanOptions here costs no skip.
_WALK_ONE_CORE = PlanOptions(sencores=1, counted_loops="walk")


def _mlir_ktdp_available() -> bool:
    """Whether this build can emit, asked of the emitter rather than guessed.

    The import list belongs to ``KtirBuilder.create``; duplicating it here is how
    the two drift, and a build missing one binding would then error instead of
    skipping.  ``ktir`` imports without a dialect build, so this is safe at
    module scope.
    """
    from torch_spyre._inductor.codegen.ktir import dialect_available

    return dialect_available()


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestKtirEmitter(unittest.TestCase):
    """The flat, untiled form: one pointwise add over a whole [16, 512, 64]
    device tile, which is what the frontend produces today."""

    # The canonical KTIR text ``generate_ktir`` emits for a single pointwise
    # ``add`` over a [512, 1024] fp16 tensor stickified to device shape
    # [16, 512, 64].
    EXPECTED_ADD_KTIR = """\
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
    %7 = tensor.empty() : tensor<16x512x64xf16>
    %8 = linalg.add ins(%4, %6 : tensor<16x512x64xf16>, tensor<16x512x64xf16>) outs(%7 : tensor<16x512x64xf16>) -> tensor<16x512x64xf16>
    %9 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %8, %9 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""

    def test_pointwise_add_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _add_op_specs(), sencores=1)
        self.assertEqual(emitted, self.EXPECTED_ADD_KTIR)

    def test_registered_ops_reach_their_own_binding(self):
        """A second op costs one recipe: same shape, different linalg builder.

        Asserted as a delta against the golden rather than a second copy of it --
        only the compute line differs.
        """
        import dataclasses

        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = [dataclasses.replace(_add_op_specs()[0], op="mul")]
        emitted = generate_ktir("ktir_fused_mul_0", specs, sencores=1)
        self.assertIn("linalg.mul ins(", emitted)
        self.assertNotIn("linalg.add", emitted)
        # Everything either side of the compute op is unchanged by the op name.
        self.assertEqual(
            emitted.replace("linalg.mul", "linalg.add").replace(
                "@ktir_fused_mul_0", "@ktir_fused_add_0"
            ),
            self.EXPECTED_ADD_KTIR,
        )


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestKtirBakedAddresses(unittest.TestCase):
    def test_baked_form_deltas(self):
        """The baked form (#65) vs ``TestKtirEmitter.EXPECTED_ADD_KTIR``.

        Asserted as deltas rather than a second golden: the two texts differ
        only in how base addresses are spelled, so a full copy would duplicate
        every line that churns together.  Reverting #65 deletes the baked arm of
        the two address helpers; the compute form does not move.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir(
            "ktir_fused_add_0", _baked_add_op_specs(), sencores=1, bake_addresses=True
        )

        # 1. No address is a runtime value: zero-arg func, no %arg anywhere.
        self.assertIn("func.func @ktir_fused_add_0() attributes {grid = [1]}", emitted)
        self.assertNotIn("%arg", emitted)
        # 2. Each base is a constant, in ELEMENTS (the byte slot >> 1 for fp16).
        for arg_index in range(3):
            with self.subTest(arg_index=arg_index):
                base = (arg_index << 34) // 2
                self.assertIn(f"arith.constant {base} : index", emitted)
        # Compute is deliberately NOT asserted here: both forms emit the same
        # linalg.add over a tensor.empty, so it is pinned by the symbolic golden
        # and is not a delta.  The two texts now differ only in addressing.


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestInternalBufferIsThreaded(unittest.TestCase):
    """Two ops in one kernel, with the intermediate threaded as an SSA value.

    The signal is stubbed (``ktir.is_internal`` reads a TensorArg field that does
    not exist yet), so this fakes it to pin the emission shape now: an internal
    buffer is neither stored nor loaded, and the second op consumes the first
    op's result directly.
    """

    def _chain(self):
        """``(a + b) + c``, where the intermediate is internal."""
        import dataclasses

        base = _add_op_specs()[0]
        a, b, mid = base.args
        c = dataclasses.replace(a, name="arg2", arg_index=3)
        out = dataclasses.replace(mid, name="buf1", arg_index=4)
        mid.is_internal = True
        first = dataclasses.replace(base, args=[a, b, mid])
        mid_in = dataclasses.replace(mid, is_input=True)
        mid_in.is_internal = True
        second = dataclasses.replace(base, args=[mid_in, c, out])
        return [first, second]

    def test_intermediate_is_neither_stored_nor_loaded(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_add_0", self._chain(), sencores=1)
        # Three loads (a, b, c) -- not four: the intermediate is a live value.
        self.assertEqual(emitted.count("ktdp.load"), 3)
        # One store: only the kernel's real output is materialised.
        self.assertEqual(emitted.count("ktdp.store"), 1)
        # Two adds, and the second consumes the first's result directly.
        adds = [ln for ln in emitted.splitlines() if "linalg.add ins(" in ln]
        self.assertEqual(len(adds), 2)
        produced = adds[0].split("=")[0].strip()
        self.assertIn(f"ins({produced},", adds[1])
        # Four buffers reach memory, not five: the intermediate gets no func
        # parameter and no memory view.
        self.assertEqual(emitted.count("ktdp.construct_memory_view"), 4)
        self.assertIn(
            "(%arg0: index, %arg1: index, %arg2: index, %arg3: index)", emitted
        )


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestTiledLoopEmission(unittest.TestCase):
    """A two-level nest, planned and emitted through the ordinary path.

    ``generate_ktir`` refuses a ``LoopSpec``, so what this changes is one
    argument: the plan is built with ``counted_loops='walk'`` instead of the
    default ``'reject'``.  Everything after that -- the plan, the func signature,
    the views, the walk, the builders -- is what an add over a nest emits today.
    The subscripts and view extents are the ones the committed ``sum`` 1-core
    KTIR fixture carries (``[2, 256, 64]`` strides ``[16384, 64, 1]``, tiles
    indexed ``[%n_stick, %m, %c0]``), so what comes out is a form a consumer
    already reads.
    """

    EXPECTED_TILED_ADD_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_tiled_add_0(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0_0 to %c2 step %c1 {
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      scf.for %arg4 = %c0_1 to %c256 step %c1_2 {
        %3 = ktdp.construct_access_tile %0[%arg3, %arg4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %4 = ktdp.load %3 : <1x1x64xindex> -> tensor<1x1x64xf16>
        %5 = ktdp.construct_access_tile %1[%arg3, %arg4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %6 = ktdp.load %5 : <1x1x64xindex> -> tensor<1x1x64xf16>
        %7 = tensor.empty() : tensor<1x1x64xf16>
        %8 = linalg.add ins(%4, %6 : tensor<1x1x64xf16>, tensor<1x1x64xf16>) outs(%7 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
        %9 = ktdp.construct_access_tile %2[%arg3, %arg4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        ktdp.store %8, %9 : tensor<1x1x64xf16>, <1x1x64xindex>
      }
    }
    return
  }
}
"""

    @staticmethod
    def _tiled_nest():
        """``a + b`` over one row per iteration of a two-level nest.

        The nest is the whole kernel contract: the op sits in the inner body, so
        it is reached by walking, not by being handed out separately.
        """
        import sympy

        from torch_spyre._C import DataFormats
        from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg

        n_stick, m = sympy.symbols("n_stick m")
        advance = 16384 * n_stick + 64 * m

        def arg(name, index, is_input):
            return TensorArg(
                is_input=is_input,
                arg_index=index,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[1, 1, 64],
                device_coordinates=[],
                allocation={"hbm": None},
                name=name,
                device_tile_advance_expr=advance,
            )

        spec = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={},
            args=[arg("arg0", 0, True), arg("arg1", 1, True), arg("buf0", 2, False)],
            op_info={},
            tiled_symbols=[[m], [n_stick]],  # innermost-first
            tiled_symbol_trip_counts={m: 256, n_stick: 2},
        )
        return LoopSpec(count=2, body=[LoopSpec(count=256, body=[spec])])

    def test_two_level_nest_golden(self):
        from torch_spyre._inductor.codegen import ktir

        nest = self._tiled_nest()
        # The plan walk descends the nest, planning each buffer at the depth its
        # op sits at and turning the nest into LoopSteps: the extents below are
        # what the two levels walk over.
        plan = ktir.build_kernel_plan([nest], _WALK_ONE_CORE)
        b = ktir.KtirBuilder.create(plan)
        # The builder already has the plan; opening the kernel needs only a name,
        # and the body is the plan's own steps -- the nest is not walked again.
        with b.open_kernel("ktir_tiled_add_0"):
            b.emit(plan.steps)
        # Pretty (non-generic) MLIR: the module verifies, terminators included.
        self.assertEqual(b.finish(), self.EXPECTED_TILED_ADD_KTIR)

    def test_plan_walk_grows_the_views_out_of_the_tile(self):
        """The buffer extents in the golden, read off the plan the walk built."""
        from torch_spyre._inductor.codegen import ktir

        plan = ktir.build_kernel_plan([self._tiled_nest()], _WALK_ONE_CORE)
        self.assertEqual([b.buf_id for b in plan.parameters], ["arg0", "arg1", "buf0"])
        for buffer in plan.parameters:
            with self.subTest(buf_id=buffer.buf_id):
                self.assertEqual(buffer.layout.extent, (2, 256, 64))
                self.assertEqual(buffer.layout.strides, (16384, 64, 1))

    def test_generate_ktir_still_refuses_the_loop(self):
        """The default mode, which is the one ``generate_ktir`` asks for."""
        from torch_spyre._inductor.codegen import ktir

        nest = self._tiled_nest()
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.generate_ktir("ktir_tiled_add_0", [nest], sencores=1)
        self.assertIn("counted loops", str(ctx.exception))
        # Same refusal from the plan walk itself, at its default mode.
        with self.assertRaises(NotImplementedError):
            ktir.build_kernel_plan([nest], ktir.PlanOptions(sencores=1))


if __name__ == "__main__":
    unittest.main()
