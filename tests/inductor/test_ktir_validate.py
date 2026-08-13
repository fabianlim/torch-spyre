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
nothing in it is skipped.**  That is the property ``ktir.validate`` exists to
provide: every ``NotImplementedError`` the emitter can raise is raised by a pure
walk over the spec tree, before the lazy dialect import, so the whole rejection
surface is covered wherever ``import torch_spyre`` works.

``test_ktir_emitter.py`` holds the complement -- the golden MLIR snapshots, which
do need the dialect build and are skipped without it.  It imports the shared
``_add_op_specs`` fixture from here, so the fixture itself stays dialect-free.
"""

import ast
import dataclasses
import inspect
import unittest
from unittest import mock

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen import ktir
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

_CONFIG = "torch_spyre._inductor.config"


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


# The symbolic address form is pinned by default: it is the form that reads no
# ``allocation["hbm"]``, so the rejections under test are the ones the fixture
# is about.  Single-core is pinned so per-op guards fire rather than the
# multi-core guard, which would otherwise come first on the default SENCORES=32.
@mock.patch(f"{_CONFIG}.bundle_symbolic_args", True)
@mock.patch(f"{_CONFIG}.sencores", 1)
class TestValidateRejections(unittest.TestCase):
    """One test per rejection ``validate`` is responsible for.

    Each asserts the exception type and a distinguishing fragment of the
    message, so a rejection cannot silently turn into a different rejection.
    """

    def _rejects(self, specs, fragment):
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.validate(specs)
        self.assertIn(fragment, str(ctx.exception))

    # -- whole-request capability ------------------------------------------

    def test_multicore_rejected(self):
        # Patched inside the body, not as a method decorator: the class-level
        # decorators are applied outermost and would override it.
        with mock.patch(f"{_CONFIG}.sencores", 2):
            self._rejects(_add_op_specs(), "multi-core work division")

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
        self.assertNotIn("atan2", ktir.REGISTRY)
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

    def test_fused_intermediate_rejected(self):
        specs = _add_op_specs()
        specs[0].args[0].arg_index = -1
        self._rejects(specs, "fused intermediates (register threading)")

    def test_unsupported_dtype_rejected(self):
        specs = _add_op_specs()
        for arg in specs[0].args:
            arg.device_dtype = DataFormats.SENINT8
        self._rejects(specs, "unsupported device dtype")
        self.assertNotIn(DataFormats.SENINT8, ktir._MLIR_ELT_TYPE_NAMES)

    def test_baked_non_hbm_allocation_rejected(self):
        specs = _baked_add_op_specs()
        specs[0].args[0].allocation = {"lx": 0x1000}
        with mock.patch(f"{_CONFIG}.bundle_symbolic_args", False):
            self._rejects(specs, "is not HBM-allocated")

    def test_baked_unassigned_hbm_address_rejected(self):
        # _add_op_specs leaves every 'hbm' address None.
        with mock.patch(f"{_CONFIG}.bundle_symbolic_args", False):
            self._rejects(_add_op_specs(), "unassigned 'hbm' address")


@mock.patch(f"{_CONFIG}.bundle_symbolic_args", True)
@mock.patch(f"{_CONFIG}.sencores", 1)
class TestRejectionsThroughGenerateKtir(unittest.TestCase):
    """``generate_ktir`` surfaces the rejections *without* reaching the dialect.

    These would pass vacuously if ``generate_ktir`` validated after importing
    ``mlir_ktdp``; they run here precisely because it validates first.
    """

    def test_reduction_unsupported(self):
        specs = _add_op_specs()
        specs[0].is_reduction = True
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs)

    def test_unregistered_op_unsupported(self):
        specs = _add_op_specs()
        specs[0].op = "atan2"
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_atan2_0", specs)

    def test_multicore_unsupported(self):
        with (
            mock.patch(f"{_CONFIG}.sencores", 2),
            self.assertRaises(NotImplementedError),
        ):
            ktir.generate_ktir("ktir_fused_add_0", _add_op_specs())


@mock.patch(f"{_CONFIG}.bundle_symbolic_args", True)
@mock.patch(f"{_CONFIG}.sencores", 1)
class TestBufferTable(unittest.TestCase):
    """What ``validate`` returns: the func signature, before any emission."""

    def test_param_entries_are_ordered_by_arg_index(self):
        specs = _add_op_specs()
        # Registration order (spec.args) is 0, 1, 2; shuffle it so the sort is
        # doing the work rather than agreeing with insertion order by luck.
        specs[0].args = [specs[0].args[2], specs[0].args[0], specs[0].args[1]]
        table = ktir.validate(specs)
        self.assertEqual([e.arg_index for e in table.param_entries], [0, 1, 2])
        self.assertEqual(
            [e.buf_id for e in table.param_entries], ["arg0", "arg1", "buf0"]
        )
        self.assertEqual(table.param_entries[0].sizes, [16, 512, 64])

    def test_symbolic_form_resolves_no_base_addresses(self):
        table = ktir.validate(_add_op_specs())
        # Every 'hbm' address in the fixture is None and never read: the bases
        # are func arguments.
        self.assertEqual([e.base_elements for e in table.param_entries], [None] * 3)

    def test_baked_form_resolves_bases_in_elements(self):
        with mock.patch(f"{_CONFIG}.bundle_symbolic_args", False):
            table = ktir.validate(_baked_add_op_specs())
        # fp16: 2 bytes per element, so the byte slot halves.
        self.assertEqual(
            [e.base_elements for e in table.param_entries],
            [0, (1 << 34) // 2, (2 << 34) // 2],
        )

    def test_repeated_buffer_is_registered_once(self):
        specs = _add_op_specs() + _add_op_specs()
        table = ktir.validate(specs)
        self.assertEqual(len(table.buffers), 3)


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


class TestRegistry(unittest.TestCase):
    """One recipe per op, and every recipe is emittable by some family method."""

    def test_every_recipe_is_complete(self):
        self.assertTrue(ktir.REGISTRY)
        for op, recipe in ktir.REGISTRY.items():
            with self.subTest(op=op):
                self.assertEqual(recipe.name, op)
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

    def test_register_rejects_a_duplicate_name(self):
        """Registration is the one place an op is declared, so it must be unique."""
        with self.assertRaises(ValueError):
            ktir.register("add", arity=2, family=ktir.Family.ELEMENTWISE)(lambda: None)

    def test_family_comes_from_the_spec_not_the_name(self):
        """A reducing spec asks for REDUCTION even when the op is registered
        elementwise -- which is why validate rejects it rather than the walk
        silently emitting the wrong shape."""
        spec = _add_op_specs()[0]
        self.assertIs(ktir.family_of(spec), ktir.Family.ELEMENTWISE)
        reducing = dataclasses.replace(spec, is_reduction=True)
        self.assertIs(ktir.family_of(reducing), ktir.Family.REDUCTION)

    def test_emit_specs_asserts_on_unvalidated_entries(self):
        """The emitter's only remaining ``raise`` is this validation-bug guard."""
        with self.assertRaises(AssertionError):
            ktir.emit_specs(None, [UnimplementedOp(op="atan2")])


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

    def test_ivs_are_innermost_last(self):
        env = ktir.ScopeStack()
        self.assertEqual(env.ivs(), [])
        with env.scope(iv="i"):
            with env.scope(iv="j"):
                self.assertEqual(env.ivs(), ["i", "j"])
            self.assertEqual(env.ivs(), ["i"])
        self.assertEqual(env.ivs(), [])


class TestNoModuleLevelDialectImport(unittest.TestCase):
    """The property this whole file depends on, asserted rather than assumed.

    ``ktir`` must not import ``mlir_ktdp`` at module level, or ``validate`` --
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
