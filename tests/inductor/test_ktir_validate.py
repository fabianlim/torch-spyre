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
import sys
import unittest
from unittest import mock

import sympy

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.codegen import ktir
from torch_spyre._inductor.constants import STAGGERED_EAS
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

_CONFIG = "torch_spyre._inductor.config"

# The single core the emitted grid is the only supported one, pinned explicitly
# so these tests do not depend on how this build is configured.
_ONE_CORE = ktir.PlanOptions(sencores=1)


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

    ``_rejects`` pins two options rather than patching globals: one core, so a
    per-op guard fires instead of the multi-core guard; and the symbolic address
    form, which reads no ``allocation["hbm"]``, so the rejection under test is
    the one the fixture is about.
    """

    def _rejects(self, specs, fragment, **options):
        options.setdefault("sencores", 1)
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs, ktir.PlanOptions(**options))
        self.assertIn(fragment, str(ctx.exception))

    # -- whole-request capability ------------------------------------------

    def test_multicore_rejected(self):
        self._rejects(_add_op_specs(), "multi-core work division", sencores=2)

    def test_empty_spec_list_rejected(self):
        self._rejects([], "no OpSpec to emit")

    # -- spec-tree shape ---------------------------------------------------

    def test_unimplemented_op_rejected(self):
        self._rejects([UnimplementedOp(op="atan2")], "unimplemented op 'atan2'")

    def test_counted_loop_rejected(self):
        self._rejects([LoopSpec(count=4, body=_add_op_specs())], "counted loops")

    def test_unexpected_entry_rejected(self):
        self._rejects(["not a spec"], "unexpected spec entry str")

    def test_reduction_rejected(self):
        specs = _add_op_specs()
        specs[0].is_reduction = True
        self._rejects(specs, "reductions are not supported")

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
        self.assertNotIn(DataFormats.SENINT8, ktir._MLIR_ELT_TYPE_NAMES)

    def test_baked_non_hbm_allocation_rejected(self):
        specs = _baked_add_op_specs()
        specs[0].args[0].allocation = {"lx": 0x1000}
        self._rejects(specs, "is not HBM-allocated", bake_addresses=True)

    def test_baked_unassigned_hbm_address_rejected(self):
        # _add_op_specs leaves every 'hbm' address None.
        self._rejects(_add_op_specs(), "unassigned 'hbm' address", bake_addresses=True)


class TestRejectionsThroughGenerateKtir(unittest.TestCase):
    """``generate_ktir`` surfaces the rejections *without* reaching the dialect.

    These would pass vacuously if ``generate_ktir`` validated after importing
    ``mlir_ktdp``; they run here precisely because it validates first.
    """

    def test_reduction_unsupported(self):
        specs = _add_op_specs()
        specs[0].is_reduction = True
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs, sencores=1)

    def test_unregistered_op_unsupported(self):
        specs = _add_op_specs()
        specs[0].op = "atan2"
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_atan2_0", specs, sencores=1)

    def test_multicore_unsupported(self):
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", _add_op_specs(), sencores=2)

    def test_unknown_option_is_a_typeerror(self):
        """Options are PlanOptions fields; a typo is not silently ignored."""
        with self.assertRaises(TypeError) as ctx:
            ktir.generate_ktir("k", _add_op_specs(), bake_address=True)
        self.assertIn("bake_address", str(ctx.exception))


class TestPlanOptions(unittest.TestCase):
    """The caller's choices, including the one that defaults to the config.

    ``sencores=None`` is the production case: nothing passes a core count, so the
    option has to resolve to whatever this build is configured for.  That default
    is the reason to test the config read at all -- every other test pins the
    value explicitly instead.
    """

    def test_core_count_defaults_to_the_configured_one(self):
        with mock.patch(f"{_CONFIG}.sencores", 7):
            self.assertEqual(ktir.PlanOptions().cores, 7)
            # An explicit value wins over the configuration.
            self.assertEqual(ktir.PlanOptions(sencores=1).cores, 1)

    def test_defaults_are_the_canonical_form(self):
        options = ktir.PlanOptions()
        self.assertFalse(options.bake_addresses)  # symbolic addresses
        self.assertEqual(options.counted_loops, "reject")

    def test_unknown_counted_loops_mode_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ktir.PlanOptions(counted_loops="unroll")
        self.assertIn("counted_loops", str(ctx.exception))

    def test_grid_comes_from_the_core_count(self):
        self.assertEqual(ktir.KernelPlan(_ONE_CORE).grid, (1,))
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.KernelPlan(ktir.PlanOptions(sencores=2))
        self.assertIn("multi-core work division", str(ctx.exception))


class TestKernelPlan(unittest.TestCase):
    """What ``build_kernel_plan`` returns: the func signature, before any emission."""

    def test_param_entries_are_ordered_by_arg_index(self):
        specs = _add_op_specs()
        # Registration order (spec.args) is 0, 1, 2; shuffle it so the sort is
        # doing the work rather than agreeing with insertion order by luck.
        specs[0].args = [specs[0].args[2], specs[0].args[0], specs[0].args[1]]
        plan = ktir.build_kernel_plan(specs, _ONE_CORE)
        self.assertEqual([e.arg_index for e in plan.parameters], [0, 1, 2])
        self.assertEqual([e.buf_id for e in plan.parameters], ["arg0", "arg1", "buf0"])
        # The plan holds the derived records, so the buffer's extent and its
        # row-major strides are readable here rather than only in the MLIR.
        self.assertEqual(plan.parameters[0].layout.extent, (16, 512, 64))
        self.assertEqual(plan.parameters[0].layout.strides, (32768, 64, 1))

    def test_symbolic_form_resolves_no_base_addresses(self):
        plan = ktir.build_kernel_plan(_add_op_specs(), _ONE_CORE)
        # Every 'hbm' address in the fixture is None and never read: the bases
        # are func arguments.
        self.assertEqual([e.base_elements for e in plan.parameters], [None] * 3)

    def test_baked_form_resolves_bases_in_elements(self):
        plan = ktir.build_kernel_plan(
            _baked_add_op_specs(),
            ktir.PlanOptions(sencores=1, bake_addresses=True),
        )
        # fp16: 2 bytes per element, so the byte slot halves.
        self.assertEqual(
            [e.base_elements for e in plan.parameters],
            [0, (1 << 34) // 2, (2 << 34) // 2],
        )

    def test_repeated_buffer_is_registered_once(self):
        specs = _add_op_specs() + _add_op_specs()
        plan = ktir.build_kernel_plan(specs, _ONE_CORE)
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
    """``is_internal`` decides materialise-vs-thread. Nothing sets it yet."""

    def test_nothing_is_internal_today(self):
        for arg in _add_op_specs()[0].args:
            self.assertFalse(ktir.is_internal(arg))

    def test_reads_the_flag_when_a_spec_carries_one(self):
        """The signal is a TensorArg field OpSpec does not have yet, so this
        fakes it: when it appears, only ``is_internal``'s body changes."""
        arg = _add_op_specs()[0].args[2]
        arg.is_internal = True
        self.assertTrue(ktir.is_internal(arg))


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
    """The loop-nest shape of the committed ``sum`` 1-core KTIR fixture.

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
    """``_levels`` / ``_solve_layout`` / ``_access`` against the ``sum`` fixture.

    ``generate_ktir`` still rejects a ``LoopSpec``, so no emission reaches these
    numbers yet; they are pinned against a KTIR file that a scheduler already consumes,
    so enabling loops is a matter of dropping that rejection rather than of
    working out what the loop form should be.
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

    def test_symbolic_trip_count_is_guarded(self):
        spec, loops = _tiled_reduction_specs()
        loops[0] = (LoopSpec(count=sympy.Symbol("s0"), body=[]), "%n_stick")
        with self.assertRaises(ktir.DownstreamUnsupported) as ctx:
            ktir._levels(spec, loops)
        self.assertIn("symbolic-loop-count", str(ctx.exception))

    def test_buffer_extent_grows_out_of_the_tile_extent(self):
        """``E_i = A_i + q[l][i] * (T_l - 1)``, matching the fixture's views."""
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

    def test_access_indices_are_the_fixture_subscripts(self):
        """``%a_view[%n_stick, %m, %c0]`` and ``%c_view[%n_stick, %c0]``."""
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        a, c = spec.args

        a_layout, a_q = ktir._solve_layout(a, levels)
        a_access = ktir._access(a, levels, a_q, a_layout)
        # The tile extent is device_size, which is what tiling already baked in.
        self.assertEqual(a_access.extent, (1, 1, 64))
        # Per view dim, the step each level takes: dim 0 <- n_stick, dim 1 <- m,
        # dim 2 <- nothing, i.e. the constant zero the fixture spells as %c0.
        self.assertEqual(a_access.index_coeffs, ((1, 0), (0, 1), (0, 0)))

        c_layout, c_q = ktir._solve_layout(c, levels)
        c_access = ktir._access(c, levels, c_q, c_layout)
        self.assertEqual(c_access.extent, (1, 64))
        self.assertEqual(c_access.index_coeffs, ((1, 0), (0, 0)))

    def test_untiled_access_sits_at_the_view_origin(self):
        """Depth zero is the general answer, not a special case."""
        arg = _add_op_specs()[0].args[0]
        layout, q = ktir._solve_layout(arg, [])
        self.assertEqual(layout.extent, (16, 512, 64))
        self.assertEqual(q, [])
        access = ktir._access(arg, [], q, layout)
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


class TestGuardedDerivationsStillProduceTheirAnswer(unittest.TestCase):
    """A guarded capability's derivation is exercised with the guard bypassed.

    The guards are one call each in front of code that works, so dropping a guard
    is all that enabling the capability takes.  These call the derivation
    directly, which is what proves there is something behind the guard.
    """

    @staticmethod
    def _symbolic_arg():
        arg = _add_op_specs()[0].args[0]
        # A symbolic outer-stick count, as a dynamic batch dim produces.
        arg.device_size = [sympy.Symbol("s0"), 512, 64]
        return arg

    def test_default_mode_guards_the_symbolic_extent(self):
        with self.assertRaises(ktir.DownstreamUnsupported) as ctx:
            ktir._layout(self._symbolic_arg(), [], [])
        self.assertIn("dynamic-view-extent", str(ctx.exception))

    def test_dynamic_mode_derives_the_symbolic_view(self):
        s0 = sympy.Symbol("s0")
        layout = ktir._layout(self._symbolic_arg(), [], [], symbolic_extent="dynamic")
        # The extent stays symbolic and the strides are row-major over it: the
        # trailing two dims are still integers, the outer stride is the product.
        self.assertEqual(layout.extent, (s0, 512, 64))
        self.assertEqual(layout.strides, (32768, 64, 1))

    def test_max_mode_bakes_the_bound(self):
        layout = ktir._layout(
            self._symbolic_arg(),
            [],
            [],
            symbolic_extent="max",
            bounds={"s0": (16, 1)},  # (max, granularity)
        )
        self.assertEqual(layout.extent, (16, 512, 64))
        self.assertEqual(layout.strides, (32768, 64, 1))

    def test_max_mode_needs_a_bound(self):
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._layout(self._symbolic_arg(), [], [], symbolic_extent="max", bounds={})
        self.assertIn("no bound for 's0'", str(ctx.exception))


class TestStatusTable(unittest.TestCase):
    """``STATUS_TABLE`` is the single record of what the emitter can emit."""

    def test_labels_are_unique_and_resolvable(self):
        labels = [row.label for row in ktir.STATUS_TABLE]
        self.assertEqual(len(labels), len(set(labels)))
        for label in labels:
            self.assertIs(ktir.status_of(label), ktir.status_of(label))
        with self.assertRaises(KeyError):
            ktir.status_of("no-such-capability")

    def test_every_guard_label_has_a_row(self):
        """A guard raises with a label; the label must be in the status table."""
        for label in ("dynamic-view-extent", "symbolic-loop-count"):
            with self.subTest(label=label):
                self.assertIs(
                    ktir.status_of(label).status, ktir.Status.DOWNSTREAM_GUARDED
                )

    def test_staggered_arrangement_is_the_only_unspecified_item(self):
        """FAILS ONCE THE STAGGERED LAYOUT IS IMPLEMENTED, deliberately.

        There is one capability with no derivation behind it.  This test fails
        the moment ``_arrangement_layout`` returns numbers for a staggered
        arrangement instead of raising, so the ``STATUS_TABLE`` row must be moved
        off ``UNSPECIFIED`` in the same commit that implements it -- the table
        cannot go stale while the code moves on.
        """
        unspecified = [
            row.label
            for row in ktir.STATUS_TABLE
            if row.status is ktir.Status.UNSPECIFIED
        ]
        self.assertEqual(unspecified, ["staggered-element-arrangement"])

        arrangement = next(iter(STAGGERED_EAS))
        with self.assertRaises(ktir.Unspecified) as ctx:
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

    def test_coordinate_set_is_recorded_as_informational(self):
        """It is emitted with no known reader; the row is why that is on purpose."""
        self.assertIs(
            ktir.status_of("coordinate-set").status, ktir.Status.INFORMATIONAL
        )


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
                fresh.generate_ktir("k", [LoopSpec(count=4, body=[])], sencores=1)
            self.assertIn("counted loops", str(ctx.exception))
            # And the derivations answer, dialect or no dialect.
            layout, _ = fresh._solve_layout(_add_op_specs()[0].args[0], [])
            self.assertEqual(layout.extent, (16, 512, 64))

    def test_emission_is_what_needs_the_dialect(self):
        # A *valid* request gets as far as the builder and no further.
        with self._blocked() as fresh, self.assertRaises(ImportError):
            fresh.generate_ktir("k", _add_op_specs(), sencores=1)


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
        pending = ["emit"]
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
