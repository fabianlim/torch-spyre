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

"""Dialect-free tests for the OpSpec->KTIR emitter's *rejections*.

**Nothing in this file imports ``mlir_ktdp``, directly or transitively, and
nothing in it is skipped.**  That is the property ``ktir.build_kernel_plan``
exists to provide: every ``NotImplementedError`` the emitter can raise is raised
by a pure walk over the spec tree, before the lazy dialect import, so the whole rejection
surface is covered wherever ``import torch_spyre`` works.

``test_ktir_emitter.py`` holds the complement -- the golden MLIR snapshots, which
do need the dialect build and are skipped without it.  It imports the shared
``_add_op_specs`` fixture from here, so the fixture itself stays dialect-free.
"""

import ast
import contextlib
import dataclasses
import importlib
import inspect
import re
import sys
import unittest

import sympy

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.codegen import ktir
from torch_spyre._inductor.constants import STAGGERED_EAS
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp


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


def _baked_add_op_specs() -> list:
    """``_add_op_specs`` with the byte HBM addresses the baked form needs."""
    specs = _add_op_specs()
    for arg in specs[0].args:
        arg.allocation = {"hbm": arg.arg_index << 34}
    return specs


class TestValidateRejections(unittest.TestCase):
    """One test per rejection ``build_kernel_plan`` is responsible for.

    Each asserts the exception type and a distinguishing fragment of the
    message, so a rejection cannot silently turn into a different rejection.

    ``_rejects`` defaults to the symbolic address form, which reads no
    ``allocation["hbm"]``, so the rejection under test is the one the fixture is
    about rather than a missing address.
    """

    def _rejects(self, specs, fragment, **options):
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs, ktir.PlanOptions(**options))
        self.assertIn(fragment, str(ctx.exception))

    # -- whole-request capability ------------------------------------------

    def test_empty_spec_list_rejected(self):
        self._rejects([], "no OpSpec to emit")

    def test_mixed_work_division_rejected(self):
        """Two ops in one kernel, two grids: there is only one grid to emit."""
        specs = _add_op_specs() + _add_op_specs()
        d1 = next(s for s in specs[1].iteration_space if str(s) == "d1")
        specs[1].iteration_space[d1] = (512, 2)
        self._rejects(specs, "different work divisions")

    def test_ragged_work_division_rejected(self):
        """A division that does not divide the axis evenly has no per-core tile."""
        specs = _add_op_specs()
        d1 = next(s for s in specs[0].iteration_space if str(s) == "d1")
        specs[0].iteration_space[d1] = (512, 7)  # 512 / 7 is not a whole tile
        self._rejects(specs, "do not divide evenly")

    # -- spec-tree shape ---------------------------------------------------

    def test_unimplemented_op_rejected(self):
        self._rejects([UnimplementedOp(op="atan2")], "unimplemented op 'atan2'")

    def test_unexpected_entry_rejected(self):
        self._rejects(["not a spec"], "unexpected spec entry str")

    def test_family_mismatch_rejected(self):
        """An ``add`` asked for as a reduction: the recipe is what has an
        emission, so the request is refused rather than emitted elementwise."""
        specs = _add_op_specs()
        specs[0].is_reduction = True
        self._rejects(specs, "registered as ELEMENTWISE")

    def test_unregistered_op_rejected(self):
        """An op with no recipe is rejected, and the message names what exists."""
        specs = _add_op_specs()
        specs[0].op = "atan2"
        self.assertNotIn("atan2", ktir.KtirBuilder.RECIPES)
        self._rejects(specs, "op 'atan2' is not supported yet")

    # -- per-op roles ------------------------------------------------------

    def test_multiple_outputs_rejected(self):
        specs = _add_op_specs()
        specs[0].args[1].is_input = False
        self._rejects(specs, "expected exactly one output, got 2")

    def test_wrong_arity_rejected(self):
        specs = _add_op_specs()
        del specs[0].args[1]
        self._rejects(specs, "'add' expects 2 inputs, got 1")

    def test_in_place_rejected(self):
        specs = _add_op_specs()
        specs[0].args[0].name = specs[0].args[-1].name
        self._rejects(specs, "in-place ops (input aliases output)")

    def test_broadcast_operand_rejected(self):
        specs = _add_op_specs()
        # A unit outer-stick extent against the output's 16: a real broadcast.
        specs[0].args[0].device_size = [1, 512, 64]
        self._rejects(specs, "broadcast / reshape operands")

    # -- per-buffer --------------------------------------------------------

    def test_non_kernel_argument_buffer_rejected(self):
        """arg_index stays -1 for LX / HBM-pool buffers; only HBM is emitted."""
        specs = _add_op_specs()
        specs[0].args[0].arg_index = -1
        self._rejects(specs, "is not a kernel argument")

    def test_unsupported_dtype_rejected(self):
        specs = _add_op_specs()
        for arg in specs[0].args:
            arg.device_dtype = DataFormats.SENINT8
        self._rejects(specs, "unsupported device dtype")
        self.assertNotIn(DataFormats.SENINT8, ktir.ElemTypes.NAMES)

    def test_baked_non_hbm_allocation_rejected(self):
        """An allocation that is neither HBM nor one this emitter threads."""
        specs = _baked_add_op_specs()
        specs[0].args[0].allocation = {"somewhere_new": 0x1000}
        self._rejects(specs, "is not HBM-allocated", bake_addresses=True)

    def test_threaded_input_without_a_producer_rejected(self):
        """An lx buffer this kernel reads but does not produce: threading it has
        no value to read, so it needs materialising."""
        specs = _add_op_specs()
        specs[0].args[0].allocation = {"lx": 0x1000}
        specs[0].args[0].arg_index = -1
        self._rejects(specs, "no op in this kernel produces it")

    def test_baked_unassigned_hbm_address_rejected(self):
        # _add_op_specs leaves every 'hbm' address None.
        self._rejects(_add_op_specs(), "unassigned 'hbm' address", bake_addresses=True)


class TestRejectionsThroughGenerateKtir(unittest.TestCase):
    """``generate_ktir`` surfaces the rejections *without* reaching the dialect.

    These would pass vacuously if ``generate_ktir`` validated after importing
    ``mlir_ktdp``; they run here precisely because it validates first.
    """

    def test_family_mismatch_unsupported(self):
        specs = _add_op_specs()
        specs[0].is_reduction = True
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs)

    def test_unregistered_op_unsupported(self):
        specs = _add_op_specs()
        specs[0].op = "atan2"
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_atan2_0", specs)

    def test_ragged_work_division_unsupported(self):
        specs = _add_op_specs()
        d1 = next(s for s in specs[0].iteration_space if str(s) == "d1")
        specs[0].iteration_space[d1] = (512, 7)
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs)

    def test_unknown_option_is_a_typeerror(self):
        """Options are PlanOptions fields; a typo is not silently ignored."""
        with self.assertRaises(TypeError) as ctx:
            ktir.generate_ktir("k", _add_op_specs(), bake_address=True)
        self.assertIn("bake_address", str(ctx.exception))


class TestPlanOptions(unittest.TestCase):
    """The caller's two choices -- both about spelling, neither a capability.

    What the kernel does comes from the contract, so there is nothing here to
    turn a feature on with: no core count (the iteration space states the grid)
    and no loop mode (a ``LoopSpec`` is a loop).
    """

    def test_defaults_are_the_canonical_form(self):
        options = ktir.PlanOptions()
        self.assertFalse(options.bake_addresses)  # symbolic addresses
        self.assertEqual(options.symbolic_extent, "static")

    def test_options_are_only_about_spelling(self):
        self.assertEqual(
            sorted(f.name for f in dataclasses.fields(ktir.PlanOptions)),
            ["bake_addresses", "symbolic_extent"],
        )

    def test_unknown_symbolic_extent_mode_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ktir.PlanOptions(symbolic_extent="guess")
        self.assertIn("symbolic_extent", str(ctx.exception))


class TestWorkDivision(unittest.TestCase):
    """The grid, and the per-core tile, as ``iteration_space`` states them.

    ``work_division.py`` has already turned ``config.sencores`` into a per-symbol
    division by the time the emitter sees a spec, so the emitter reads the
    contract and never the config -- the same source the SDSC path reads as its
    work slices.
    """

    @staticmethod
    def _divided(divisions):
        """``_add_op_specs`` with ``{symbol name: division}`` applied."""
        specs = _add_op_specs()
        space = specs[0].iteration_space
        for symbol, (extent, _div) in list(space.items()):
            space[symbol] = (extent, divisions.get(str(symbol), 1))
        return specs

    def test_an_undivided_space_is_one_core(self):
        plan = ktir.build_kernel_plan(_add_op_specs())
        self.assertEqual(plan.grid, (1,))
        self.assertEqual(plan.divisions, ())

    def test_the_grid_is_the_product_of_the_divisions(self):
        plan = ktir.build_kernel_plan(self._divided({"d1": 32}))
        self.assertEqual(plan.grid, (32,))
        self.assertEqual(plan.divisions, (ktir.Division(symbol="d1", div=32, inner=1),))

    def test_two_divided_symbols_are_mixed_radix(self):
        """Outermost-first, and ``inner`` is that symbol's stride in the grid."""
        plan = ktir.build_kernel_plan(self._divided({"d0": 2, "d1": 4}))
        self.assertEqual(plan.grid, (8,))
        self.assertEqual(
            plan.divisions,
            (
                ktir.Division(symbol="d0", div=2, inner=4),
                ktir.Division(symbol="d1", div=4, inner=1),
            ),
        )

    def test_the_tile_shrinks_and_the_view_does_not(self):
        """One core's tile is its share; every core addresses the whole buffer."""
        plan = ktir.build_kernel_plan(self._divided({"d1": 32}))
        for buffer in plan.parameters:
            with self.subTest(buf_id=buffer.buf_id):
                self.assertEqual(buffer.layout.extent, (16, 512, 64))
        step = plan.steps[0]
        self.assertEqual(step.out.extent, (16, 16, 64))  # 512 / 32 rows
        # The division walks dim 1 in per-core-extent steps, and nothing else.
        self.assertEqual(step.out.index_coeffs, ((0,), (16,), (0,)))

    def test_a_division_no_output_axis_follows_is_rejected(self):
        """A stick is the unit of transfer, so the lane axis is never divided --
        which leaves a division of the lane symbol with no axis to walk, and every
        core writing the same elements.  Refused rather than silently duplicated."""
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(self._divided({"d2": 2}))
        self.assertIn("no device axis of the output", str(ctx.exception))


class TestKernelPlan(unittest.TestCase):
    """What ``build_kernel_plan`` returns: the func signature, before any emission."""

    def test_param_entries_are_ordered_by_arg_index(self):
        specs = _add_op_specs()
        # Registration order (spec.args) is 0, 1, 2; shuffle it so the sort is
        # doing the work rather than agreeing with insertion order by luck.
        specs[0].args = [specs[0].args[2], specs[0].args[0], specs[0].args[1]]
        plan = ktir.build_kernel_plan(specs)
        self.assertEqual([e.arg_index for e in plan.parameters], [0, 1, 2])
        self.assertEqual([e.buf_id for e in plan.parameters], ["arg0", "arg1", "buf0"])
        # The plan holds the derived records, so the buffer's extent and its
        # row-major strides are readable here rather than only in the MLIR.
        self.assertEqual(plan.parameters[0].layout.extent, (16, 512, 64))
        self.assertEqual(plan.parameters[0].layout.strides, (32768, 64, 1))

    def test_symbolic_form_resolves_no_base_addresses(self):
        plan = ktir.build_kernel_plan(_add_op_specs())
        # Every 'hbm' address in the fixture is None and never read: the bases
        # are func arguments.
        self.assertEqual([e.base_elements for e in plan.parameters], [None] * 3)

    def test_baked_form_resolves_bases_in_elements(self):
        plan = ktir.build_kernel_plan(
            _baked_add_op_specs(),
            ktir.PlanOptions(bake_addresses=True),
        )
        # fp16: 2 bytes per element, so the byte slot halves.
        self.assertEqual(
            [e.base_elements for e in plan.parameters],
            [0, (1 << 34) // 2, (2 << 34) // 2],
        )

    def test_repeated_buffer_is_registered_once(self):
        specs = _add_op_specs() + _add_op_specs()
        plan = ktir.build_kernel_plan(specs)
        self.assertEqual(len(plan.buffers), 3)


class TestBaseAddressElements(unittest.TestCase):
    """``_base_address_elements`` in isolation, with no dialect and no config."""

    @staticmethod
    def _arg(allocation):
        arg = _add_op_specs()[0].args[1]
        arg.allocation = allocation
        return arg

    def test_byte_address_scales_to_elements(self):
        # fp16: 2 bytes per element.  Zero is a real address, not "unset".
        self.assertEqual(
            ktir._base_address_elements(self._arg({"hbm": 1 << 34})), 1 << 33
        )
        self.assertEqual(ktir._base_address_elements(self._arg({"hbm": 0})), 0)

    def test_unassigned_or_non_hbm_rejected(self):
        for allocation in ({"hbm": None}, {"lx": 0x1000}, {"hbm_pool": 0x1000}, {}):
            with (
                self.subTest(alloc=allocation),
                self.assertRaises(NotImplementedError),
            ):
                ktir._base_address_elements(self._arg(allocation))


class TestInternalBufferSignal(unittest.TestCase):
    """``is_internal`` decides materialise-vs-thread, from ``allocation``.

    The same field ``create_tensor_arg`` uses to decide what becomes a kernel
    argument at all, so the two cannot disagree about which buffers the kernel
    owns.
    """

    def test_an_hbm_buffer_is_passed_in_not_owned(self):
        for arg in _add_op_specs()[0].args:
            self.assertFalse(ktir.is_internal(arg))

    def test_planning_placed_it_means_the_kernel_owns_it(self):
        for allocation in ({"lx": 0x1000}, {"hbm_pool": 0x2000}):
            with self.subTest(allocation=allocation):
                arg = _add_op_specs()[0].args[2]
                arg.allocation = allocation
                self.assertTrue(ktir.is_internal(arg))

    def test_an_unrecognised_allocation_is_not_threaded(self):
        """Threading is chosen on a positive signal, so an allocation this
        emitter does not know reaches the buffer rejection instead."""
        specs = _add_op_specs()
        specs[0].args[2].allocation = {"somewhere_new": 0}
        self.assertFalse(ktir.is_internal(specs[0].args[2]))

    def test_a_threaded_buffer_nothing_reads_is_rejected(self):
        """An intermediate whose consumer is in another kernel: not stored, and
        not read here either, so the op that produced it would write nowhere."""
        specs = _add_op_specs()
        specs[0].args[2].allocation = {"lx": 0x1000}
        specs[0].args[2].arg_index = -1
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("nothing in this kernel", str(ctx.exception))


class TestRecipes(unittest.TestCase):
    """One recipe per op, and every recipe is emittable by some family method."""

    def test_every_recipe_is_complete(self):
        self.assertTrue(ktir.KtirBuilder.RECIPES)
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            with self.subTest(op=op):
                self.assertGreaterEqual(recipe.arity, 1)
                self.assertIsInstance(recipe.family, ktir.Family)
                # A thunk, not the builder itself: resolving it here would need
                # the dialect, which this module deliberately does not require.
                self.assertTrue(callable(recipe.binding))
                # The family it declares must be one the builder can emit,
                # otherwise the walk fails at emit time rather than here.
                self.assertTrue(
                    callable(
                        getattr(ktir.KtirBuilder, recipe.family.name.lower(), None)
                    ),
                    f"KtirBuilder has no {recipe.family.name.lower()}() for {op!r}",
                )

    def test_recipe_rejects_a_nonsense_arity(self):
        """A duplicate op name is ruff F601; arity is checked at construction."""
        with self.assertRaises(ValueError):
            ktir.Recipe(arity=0, family=ktir.Family.ELEMENTWISE, binding=lambda: None)

    def test_family_comes_from_the_spec_not_the_name(self):
        """A reducing spec asks for REDUCTION even when the op is registered
        elementwise -- which is why the plan walk rejects it rather than the walk
        silently emitting the wrong shape."""
        spec = _add_op_specs()[0]
        self.assertIs(ktir.Family.of(spec), ktir.Family.ELEMENTWISE)
        reducing = dataclasses.replace(spec, is_reduction=True)
        self.assertIs(ktir.Family.of(reducing), ktir.Family.REDUCTION)

    def test_emit_asserts_on_an_unplanned_step(self):
        """The emitter's only remaining ``raise`` is this plan-bug guard.

        Called unbound with ``self=None``: the type check happens before any
        builder state is touched, which is why this needs no dialect build.
        ``UnimplementedOp`` cannot reach emission at all now -- a step tree holds
        only steps -- so the guard is about a malformed plan, not a rejected op.
        """
        with self.assertRaises(AssertionError):
            ktir.KtirBuilder.emit(None, [UnimplementedOp(op="atan2")])


def _tiled_reduction_specs() -> tuple:
    """The loop-nest shape of a hand-written 1-core KTIR ``sum`` kernel.

    Two ``scf.for`` levels over a [2, 256, 64] fp16 input reduced to a [2, 64]
    output: the outer level walks whole sticks (2 trips), the inner level walks
    rows within a stick (256 trips).  The input's tile is one row, the output's
    one stick, and each arg's ``device_tile_advance_expr`` is the linearized
    element offset for one step of each level:

        a: 16384*n_stick + 64*m     c: 64*n_stick

    Returns ``(spec, loops)``: the op, and the ``LoopSpec`` chain the plan walk
    would reach it with, read off one real nest so the trip counts the derivations
    see are the nest's own.
    """
    n_stick, m = sympy.symbols("n_stick m")

    def arg(name, index, is_input, size, advance):
        return TensorArg(
            is_input=is_input,
            arg_index=index,
            device_dtype=DataFormats.IEEE_FP16,
            device_size=list(size),
            device_coordinates=[],
            allocation={"hbm": 0},
            name=name,
            device_tile_advance_expr=advance,
        )

    spec = OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={},
        args=[
            arg("a", 0, True, [1, 1, 64], 16384 * n_stick + 64 * m),
            arg("c", 1, False, [1, 64], 64 * n_stick),
        ],
        op_info={},
        # innermost-first, one entry per enclosing level
        tiled_symbols=[[m], [n_stick]],
        tiled_symbol_trip_counts={m: 256, n_stick: 2},
    )
    nest = LoopSpec(count=2, body=[LoopSpec(count=256, body=[spec])])
    return spec, [nest, nest.body[0]]


class TestLoopDerivations(unittest.TestCase):
    """``_levels`` / ``_solve_layout`` / ``_access`` against that ``sum`` nest.

    The numbers are pinned against a KTIR kernel a scheduler already consumes, so
    what the loop form should be is not this emitter's invention.
    """

    def test_levels_are_outermost_first_with_their_trip_counts(self):
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        self.assertEqual([lvl.trip for lvl in levels], [2, 256])
        # tiled_symbols is innermost-first; the levels come back outermost-first.
        self.assertEqual(
            [str(s) for lvl in levels for s in lvl.symbols], ["n_stick", "m"]
        )

    def test_levels_must_match_the_enclosing_nest(self):
        spec, loops = _tiled_reduction_specs()
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._levels(spec, loops[:1])
        self.assertIn("tiled_symbols", str(ctx.exception))

    def test_a_symbolic_trip_count_is_read_not_refused(self):
        """``_trip`` reads the count; whether one can be emitted is the plan's
        call, because only the plan knows the ``symbolic_extent`` mode."""
        s0 = sympy.Symbol("s0")
        self.assertEqual(ktir._trip(LoopSpec(count=4, body=[])), 4)
        self.assertEqual(ktir._trip(LoopSpec(count=s0, body=[])), s0)

    def test_buffer_extent_grows_out_of_the_tile_extent(self):
        """``E_i = A_i + q[l][i] * (T_l - 1)``, matching that kernel's views."""
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        a, c = spec.args

        a_layout, a_q = ktir._solve_layout(a, levels)
        # 2 = 1 + 1*(2-1), 256 = 1 + 1*(256-1), and the stick dim is unchanged.
        self.assertEqual(a_layout.extent, (2, 256, 64))
        self.assertEqual(a_layout.strides, (16384, 64, 1))
        # One dim per level: the outer level walks dim 0, the inner walks dim 1.
        self.assertEqual(a_q, [(1, 0, 0), (0, 1, 0)])

        c_layout, c_q = ktir._solve_layout(c, levels)
        self.assertEqual(c_layout.extent, (2, 64))
        self.assertEqual(c_layout.strides, (64, 1))
        # The inner level does not move the output: it is the reduced dim.
        self.assertEqual(c_q, [(1, 0), (0, 0)])

    def test_access_indices_are_the_kernel_subscripts(self):
        """``%a_view[%n_stick, %m, %c0]`` and ``%c_view[%n_stick, %c0]``."""
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        a, c = spec.args

        a_layout, a_q = ktir._solve_layout(a, levels)
        a_access = ktir._access(a, a.device_size, a_q, a_layout)
        # The tile extent is device_size, which is what tiling already baked in.
        self.assertEqual(a_access.extent, (1, 1, 64))
        # Per view dim, the step each level takes: dim 0 <- n_stick, dim 1 <- m,
        # dim 2 <- nothing, i.e. the constant zero the kernel spells as %c0.
        self.assertEqual(a_access.index_coeffs, ((1, 0), (0, 1), (0, 0)))

        c_layout, c_q = ktir._solve_layout(c, levels)
        c_access = ktir._access(c, c.device_size, c_q, c_layout)
        self.assertEqual(c_access.extent, (1, 64))
        self.assertEqual(c_access.index_coeffs, ((1, 0), (0, 0)))

    def test_untiled_access_sits_at_the_view_origin(self):
        """Depth zero is the general answer, not a special case."""
        arg = _add_op_specs()[0].args[0]
        layout, q = ktir._solve_layout(arg, [])
        self.assertEqual(layout.extent, (16, 512, 64))
        self.assertEqual(q, [])
        access = ktir._access(arg, arg.device_size, q, layout)
        # One empty sum per dim: every index expression is zero.
        self.assertEqual(access.index_coeffs, ((), (), ()))

    def test_advance_no_dim_divides_is_reported(self):
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        a = spec.args[0]
        # 100 elements is not a whole number of steps along any dim of a view
        # whose strides are 16384, 64 and 1 (the stick dim is never stepped).
        a.device_tile_advance_expr = 100 * sympy.Symbol("n_stick")
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._solve_layout(a, levels)
        self.assertIn("not a whole number of steps", str(ctx.exception))


# ---------------------------------------------------------------------------
# What we generate
# ---------------------------------------------------------------------------


class TestSymbolicExtentModes(unittest.TestCase):
    """The three answers to "this device size is a sympy expression".

    ``symbolic_extent`` picks one: refuse it, take it as a func argument, or bake
    its upper bound.  The last is what the SDSC path does (``_resolve_sdsc_size``
    reads the same ``symbolic_dim_bounds`` max), so 'max' is parity with the
    bundle emitter and 'dynamic' is the form the KTDP lowering builds for a
    non-constant descriptor dimension.
    """

    @staticmethod
    def _symbolic_arg():
        arg = _add_op_specs()[0].args[0]
        # A symbolic outer-stick count, as a dynamic batch dim produces.
        arg.device_size = [sympy.Symbol("s0"), 512, 64]
        return arg

    def test_static_mode_refuses_a_symbolic_extent(self):
        with self.assertRaises(ktir.Unimplemented) as ctx:
            ktir._layout(self._symbolic_arg(), [], [])
        self.assertIn("static-view-extent", str(ctx.exception))
        # The message points at the two modes that can express it.
        self.assertIn("symbolic_extent='dynamic'", str(ctx.exception))

    def test_dynamic_mode_keeps_the_symbol(self):
        s0 = sympy.Symbol("s0")
        layout = ktir._layout(self._symbolic_arg(), [], [], symbolic_extent="dynamic")
        # The extent stays symbolic and the strides are row-major over it: the
        # trailing two dims are still integers, so the outer stride is a product
        # of integers and no stride arithmetic is needed to emit this.
        self.assertEqual(layout.extent, (s0, 512, 64))
        self.assertEqual(layout.strides, (32768, 64, 1))

    def test_max_mode_bakes_the_bound(self):
        layout = ktir._layout(
            self._symbolic_arg(),
            [],
            [],
            symbolic_extent="max",
            bounds={"s0": (16, 1)},  # (max, granularity), as SDSC reads it
        )
        self.assertEqual(layout.extent, (16, 512, 64))
        self.assertEqual(layout.strides, (32768, 64, 1))

    def test_max_mode_needs_a_bound(self):
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._layout(self._symbolic_arg(), [], [], symbolic_extent="max", bounds={})
        self.assertIn("no bound for 's0'", str(ctx.exception))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            ktir.PlanOptions(symbolic_extent="guess")


class TestDimensionArguments(unittest.TestCase):
    """A symbolic dimension becomes a func argument the plan names.

    One argument does both jobs: it sizes the dynamic memref dim and bounds the
    loop that walks it, which is why the two used to look like separate
    unsupported capabilities.
    """

    @staticmethod
    def _dynamic_nest():
        """``a + b`` over a run-time number of sticks, one stick per iteration."""
        s0, n = sympy.symbols("s0 n")

        def arg(name, index, is_input):
            return TensorArg(
                is_input=is_input,
                arg_index=index,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[1, 64],
                device_coordinates=[],
                allocation={"hbm": None},
                name=name,
                device_tile_advance_expr=64 * n,
            )

        spec = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={},
            args=[arg("arg0", 0, True), arg("arg1", 1, True), arg("buf0", 2, False)],
            op_info={},
            tiled_symbols=[[n]],
            tiled_symbol_trip_counts={n: s0},
        )
        return LoopSpec(count=s0, body=[spec])

    @staticmethod
    def _options(**overrides):
        fields = {"symbolic_extent": "dynamic"}
        return ktir.PlanOptions(**(fields | overrides))

    def test_the_plan_names_the_dimension_it_needs(self):
        plan = ktir.build_kernel_plan([self._dynamic_nest()], self._options())
        self.assertEqual(plan.dims, ("s0",))
        # The buffer grows to the symbol, and its strides stay integers.
        for buffer in plan.parameters:
            with self.subTest(buf_id=buffer.buf_id):
                self.assertEqual(buffer.layout.extent, (sympy.Symbol("s0"), 64))
                self.assertEqual(buffer.layout.strides, (64, 1))

    def test_the_loop_bound_is_that_same_dimension(self):
        plan = ktir.build_kernel_plan([self._dynamic_nest()], self._options())
        self.assertEqual(plan.steps[0].trip, "s0")
        self.assertIn(plan.steps[0].trip, plan.dims)

    def test_static_mode_refuses_a_symbolic_trip_count(self):
        """The loop's bound is planned before its body, so this is the refusal
        that fires -- the symbolic extents inside it are never reached."""
        with self.assertRaises(ktir.Unimplemented) as ctx:
            ktir.build_kernel_plan(
                [self._dynamic_nest()], self._options(symbolic_extent="static")
            )
        self.assertIn("symbolic-loop-count", str(ctx.exception))
        self.assertIn("symbolic_extent='dynamic'", str(ctx.exception))

    def test_a_computed_dimension_is_refused(self):
        """Only a bare symbol can be an argument; an expression would have to be
        computed from the arguments."""
        nest = self._dynamic_nest()
        nest.count = 2 * sympy.Symbol("s0")
        for arg in nest.body[0].args:
            arg.device_tile_advance_expr = None
        with self.assertRaises(ktir.Unimplemented) as ctx:
            ktir.build_kernel_plan([nest], self._options())
        self.assertIn("computed-dimension", str(ctx.exception))

    def test_a_symbolic_stride_is_refused(self):
        """A symbolic dim that is not the outermost makes an outer stride an
        expression, which would need arithmetic on the dimension arguments."""
        nest = self._dynamic_nest()
        spec = nest.body[0]
        for arg in spec.args:
            # [2, s0, 64]: the middle dim is the one that grows, so stride 0
            # becomes s0*64 rather than an integer.
            arg.device_size = [2, 1, 64]
            arg.device_tile_advance_expr = 64 * sympy.Symbol("n")
        with self.assertRaises(ktir.Unimplemented) as ctx:
            ktir.build_kernel_plan([nest], self._options())
        self.assertIn("computed-view-stride", str(ctx.exception))


# ---------------------------------------------------------------------------
# What we refuse, and why
# ---------------------------------------------------------------------------


class TestRefusals(unittest.TestCase):
    """The labelled capabilities this emitter does not implement.

    A label is a token shared by the raise and this test, so grepping it finds
    both.  No message here claims a consumer is the blocker: this repository
    cannot run dbo-opt or the scheduler, so what they accept is not observable
    from these tests, and two labels that used to claim it were both wrong.
    """

    def test_staggered_arrangement_is_unimplemented(self):
        """FAILS ONCE THE STAGGERED LAYOUT IS IMPLEMENTED, deliberately.

        The permutation has never been written down as numbers, so unlike every
        other refusal there is no derivation behind this one.  The test fails the
        moment ``_arrangement_layout`` returns numbers instead of raising, which
        is the prompt to delete it along with the label.
        """
        arrangement = next(iter(STAGGERED_EAS))
        with self.assertRaises(ktir.Unimplemented) as ctx:
            ktir._arrangement_layout(arrangement, (16, 512, 64), (32768, 64, 1))
        self.assertIn("staggered-element-arrangement", str(ctx.exception))

    def test_standard_arrangement_is_plain_row_major(self):
        extent, strides = (16, 512, 64), (32768, 64, 1)
        for arrangement in (
            None,
            ElementArrangement.STANDARD,
            ElementArrangement.QFP8CH,
        ):
            with self.subTest(arrangement=arrangement):
                self.assertEqual(
                    ktir._arrangement_layout(arrangement, extent, strides),
                    (extent, strides),
                )

    def test_every_label_is_greppable_and_uniquely_owned(self):
        """Each label is raised from exactly one site, so grepping it is exact."""
        source = inspect.getsource(ktir)
        labels = re.findall(r'_unimplemented\(\s*\n?\s*"([^"]+)"', source)
        self.assertEqual(sorted(labels), sorted(set(labels)))
        self.assertEqual(
            sorted(labels),
            [
                "computed-dimension",
                "computed-view-stride",
                "staggered-element-arrangement",
                "static-view-extent",
                "symbolic-loop-count",
            ],
        )

    def test_no_refusal_message_blames_a_consumer(self):
        """A refusal says what is missing here, not what someone else rejects.

        Checked over the ``_unimplemented`` messages rather than the whole file:
        naming dbo-opt is legitimate where it explains why an *option* exists
        (baking addresses), but not as the reason a capability is refused, which
        this repository cannot observe.
        """
        tree = ast.parse(inspect.getsource(ktir))
        messages = [
            " ".join(
                part.value
                for part in ast.walk(node.args[1])
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_unimplemented"
        ]
        self.assertEqual(len(messages), 5)
        for message in messages:
            with self.subTest(message=message[:40]):
                for blame in ("dbo-opt", "no consumer", "nothing lowers", "scheduler"):
                    self.assertNotIn(blame, message)


class TestWithoutTheDialectBuild(unittest.TestCase):
    """``ktir`` imports, and rejects, with ``mlir_ktdp`` made unimportable.

    The rest of this module relies on the dialect never being needed; here it is
    actively blocked, so the reliance is checked rather than assumed.
    """

    class _Blocker:
        """A ``sys.meta_path`` finder that refuses ``mlir_ktdp``."""

        def find_spec(self, name, path=None, target=None):
            if name == "mlir_ktdp" or name.startswith("mlir_ktdp."):
                raise ImportError(f"blocked: {name}")
            # None: every other name falls through to the real finders.

    @contextlib.contextmanager
    def _blocked(self):
        """A freshly imported ``ktir`` that cannot reach the dialect.

        A fresh module because ``_load_dialects`` caches its handles: an already
        loaded ``ktir`` in this process may have bound them before the block.
        """
        name = ktir.__name__
        blocker = self._Blocker()
        saved = {
            key: module
            for key, module in sys.modules.items()
            if key == "mlir_ktdp" or key.startswith("mlir_ktdp.")
        }
        saved[name] = sys.modules.pop(name)
        for key in saved:
            sys.modules.pop(key, None)
        sys.meta_path.insert(0, blocker)
        try:
            yield importlib.import_module(name)
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.pop(name, None)
            sys.modules.update(saved)

    def test_imports_and_rejects_without_the_dialect(self):
        with self._blocked() as fresh:
            self.assertFalse(fresh.dialect_available())
            # A rejection, not an ImportError: the plan walk runs first and needs
            # nothing from the dialect.
            with self.assertRaises(NotImplementedError) as ctx:
                fresh.generate_ktir("k", [UnimplementedOp(op="atan2")])
            self.assertIn("unimplemented op", str(ctx.exception))
            # And the derivations answer, dialect or no dialect.
            layout, _ = fresh._solve_layout(_add_op_specs()[0].args[0], [])
            self.assertEqual(layout.extent, (16, 512, 64))

    def test_emission_is_what_needs_the_dialect(self):
        # A *valid* request gets as far as the builder and no further.
        with self._blocked() as fresh, self.assertRaises(ImportError):
            fresh.generate_ktir("k", _add_op_specs())


class TestScopeStack(unittest.TestCase):
    """The lexical scope the walk carries: induction variables and live values."""

    def test_produced_values_are_scoped(self):
        env = ktir.ScopeStack()
        self.assertIsNone(env.produced("buf0"))
        with env.scope():
            env.bind_produced("buf0", "v0")
            self.assertEqual(env.produced("buf0"), "v0")
        self.assertIsNone(env.produced("buf0"))

    def test_inner_scope_shadows_outer(self):
        env = ktir.ScopeStack()
        env.bind_produced("buf0", "outer")
        with env.scope():
            env.bind_produced("buf0", "inner")
            self.assertEqual(env.produced("buf0"), "inner")
        self.assertEqual(env.produced("buf0"), "outer")

    def test_value_only_scopes_are_not_loops(self):
        """A frame with no induction variable adds no level to zip against."""
        env = ktir.ScopeStack()
        with env.scope(iv="i"):
            with env.scope():  # a plain value scope
                self.assertEqual(env.ivs(), ["i"])
            with env.scope(iv="j"):
                self.assertEqual(env.ivs(), ["i", "j"])

    def test_ivs_are_innermost_last(self):
        env = ktir.ScopeStack()
        self.assertEqual(env.ivs(), [])
        with env.scope(iv="i"):
            with env.scope(iv="j"):
                self.assertEqual(env.ivs(), ["i", "j"])
            self.assertEqual(env.ivs(), ["i"])
        self.assertEqual(env.ivs(), [])


class TestEmissionCannotRefuse(unittest.TestCase):
    """Nothing reachable from ``KtirBuilder.emit`` can raise a rejection.

    This is the property the step tree buys: the plan runs every derivation and
    every guard, and emission consumes only the plan's records, so a request the
    plan accepted cannot be refused half-emitted.  Asserted over the call graph
    rather than trusted, because the failure mode is silent -- a derivation called
    from the emission path would keep working right up to the input that trips its
    guard, with a half-built module already in hand.
    """

    def test_only_plan_bug_assertions_are_reachable_from_emit(self):
        tree = ast.parse(inspect.getsource(ktir))
        builder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KtirBuilder"
        )
        methods = {
            node.name: node
            for node in builder.body
            if isinstance(node, ast.FunctionDef)
        }
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        seen: set[str] = set()
        # ``compute`` reaches a family's method by name (``Family.ELEMENTWISE`` ->
        # ``elementwise``), which no call-graph walk can follow, so every family
        # method is a root here: a new family cannot escape this check by being
        # dispatched dynamically.
        families = [family.name.lower() for family in ktir.Family]
        for family in families:
            self.assertIn(family, methods, f"KtirBuilder has no {family}()")
        pending = ["emit", *families]
        raised: list[tuple[str, str]] = []
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            node = methods.get(name) or functions.get(name)
            if node is None:  # a dialect builder, not ours
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise):
                    called = getattr(sub.exc, "func", sub.exc)
                    raised.append(
                        (
                            name,
                            getattr(called, "id", None) or getattr(called, "attr", ""),
                        )
                    )
                if isinstance(sub, ast.Call):
                    callee = getattr(sub.func, "id", None) or getattr(
                        sub.func, "attr", None
                    )
                    if callee in methods or callee in functions:
                        pending.append(callee)

        # Every raise on the emission path is a malformed-plan assertion.  In
        # particular no guard (_downstream_unsupported / _unspecified) and no
        # derivation that can raise is reachable.
        self.assertTrue(raised, "expected the plan-bug assertions to be found")
        self.assertEqual({kind for _, kind in raised}, {"AssertionError"})
        for guard in ("_downstream_unsupported", "_unspecified", "_levels", "_access"):
            with self.subTest(unreachable=guard):
                self.assertNotIn(guard, seen)


class TestNoModuleLevelDialectImport(unittest.TestCase):
    """The property this whole file depends on, asserted rather than assumed.

    ``ktir`` must not import ``mlir_ktdp`` at module level, or the plan walk --
    and every test above -- becomes unrunnable without the dialect build.
    """

    def test_ktir_has_no_top_level_mlir_ktdp_import(self):
        tree = ast.parse(inspect.getsource(ktir))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                self.assertFalse(
                    name.split(".")[0] == "mlir_ktdp",
                    f"ktir.py imports {name} at module level",
                )


if __name__ == "__main__":
    unittest.main()
