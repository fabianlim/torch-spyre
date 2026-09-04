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
do need the dialect build and are skipped without it.  It imports the shared spec
builders from here (``make_op_spec`` and friends), so they stay dialect-free.
"""

import ast
import contextlib
import dataclasses
import importlib
import inspect
import sys
import unittest
from unittest import mock

import regex as re
import sympy

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen import ktir
from torch_spyre._inductor.constants import STAGGERED_EAS
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

# ---------------------------------------------------------------------------
# Building a spec
#
# One builder, so a test states only what it is about and states it on one line:
# ``make_op_spec()`` is the whole pointwise contract the frontend produces, and
# every keyword is one deviation from it.  The ``TensorArg``s are built inside,
# because their two positional fields are not a test's to choose: ``arg_index``
# is the position among the HBM args, and an arg that memory planning placed (an
# ``lx`` / ``hbm_pool`` allocation) is not passed in at all, so it takes -1 and
# consumes no position -- the frontend's own rule, applied here rather than
# restated per fixture.
#
# The per-arg keywords (``names`` / ``sizes`` / ``allocations`` / ``advances``)
# are indexed over inputs then outputs, and may be short or hold ``None`` to
# leave an arg at its default.
# ---------------------------------------------------------------------------

FP16 = DataFormats.SEN169_FP16
ADD_SIZE = [16, 512, 64]


def make_op_spec(
    op: str = "add",
    *,
    inputs: int = 2,
    outputs: int = 1,
    names: list | None = None,
    size: list = ADD_SIZE,
    sizes: list | None = None,
    coords: list | None = None,
    coords_per_arg: list | None = None,
    dtype: DataFormats = FP16,
    arrangements: list | None = None,
    allocations: list | None = None,
    baked: bool = False,
    advances: list | None = None,
    is_reduction: bool = False,
    divisions: dict | None = None,
    space: dict | None = None,
    tiled: list | None = None,
    trips: dict | None = None,
    first_arg_index: int = 0,
    op_info: dict | None = None,
) -> OpSpec:
    """A finished ``OpSpec``, defaulting to ``a + b`` at [16, 512, 64] fp16.

    That default is what the SuperDSC frontend produces for a pointwise add: two
    HBM inputs and one HBM output at identity ``(d0, ..., dn)`` coordinates, with
    the HBM address left unassigned (the symbolic form reads no address, and the
    baked form is the one that rejects that).

    The deviations, each a keyword:

    * ``op`` / ``inputs`` / ``outputs`` / ``names`` -- the op and its roles.
    * ``size`` for every arg, or ``sizes`` per arg; likewise ``coords`` for every
      arg or ``coords_per_arg`` for one list each, which a reduction needs because
      its output drops an axis.  ``coords=[]`` is a tiled arg, addressing through
      ``advances`` instead of coordinates.
    * ``arrangements`` per arg, for a buffer whose elements are not in the
      standard order -- ``ElementArrangement.EXX2`` is a statistic buffer holding
      two values in one element, and the default is ``STANDARD``.
    * ``allocations`` per arg, for an ``lx`` / ``hbm_pool`` intermediate or an
      unrecognised space; ``baked=True`` for the byte HBM address the baked form
      wants, which is the same field said the other way, so not both.
    * ``divisions`` maps a coordinate symbol's name to its work division;
      ``space`` replaces the iteration space outright (``{}`` for a tiled op).
    * ``tiled`` / ``trips`` are the loop-level symbols and trip counts, and
      ``first_arg_index`` continues the numbering for a second op in one kernel.
    * ``op_info`` is the op's auxiliary dict, which the recipes that take scalar
      arguments read (softplus's beta/threshold live in ``op_info["constants"]``);
      it defaults to empty, which every other op wants.
    """
    if allocations and baked:
        raise ValueError("make_op_spec: pass allocations= or baked=, not both")

    def at(per_arg: list | None, position: int):
        """``per_arg[position]``, treating short lists and ``None`` as default."""
        if per_arg is None or position >= len(per_arg):
            return None
        return per_arg[position]

    roles = [(True, i) for i in range(inputs)] + [(False, i) for i in range(outputs)]
    args, next_index = [], first_arg_index
    for position, (is_input, ordinal) in enumerate(roles):
        allocation = at(allocations, position)
        # An arg planning placed is not passed in: -1, and it takes no position.
        if allocation is not None and "hbm" not in allocation:
            arg_index = -1
        else:
            arg_index, next_index = next_index, next_index + 1
            if allocation is None:
                allocation = {"hbm": arg_index << 34 if baked else None}
        arg_size = at(sizes, position) or size
        arg_coords = at(coords_per_arg, position)
        if arg_coords is None:
            arg_coords = (
                sympy.symbols(f"d0:{len(arg_size)}") if coords is None else coords
            )
        args.append(
            TensorArg(
                is_input=is_input,
                arg_index=arg_index,
                device_dtype=dtype,
                device_size=list(arg_size),
                device_coordinates=list(arg_coords),
                allocation=allocation,
                name=at(names, position)
                or (f"arg{ordinal}" if is_input else f"buf{ordinal}"),
                device_tile_advance_expr=at(advances, position),
                element_arrangement=at(arrangements, position)
                or ElementArrangement.STANDARD,
            )
        )

    if space is None:
        out_size = args[-1].device_size
        space = {
            symbol: (extent, (divisions or {}).get(str(symbol), 1))
            for symbol, extent in zip(sympy.symbols(f"d0:{len(out_size)}"), out_size)
        }
    return OpSpec(
        op=op,
        is_reduction=is_reduction,
        iteration_space=space,
        args=args,
        op_info=op_info or {},
        tiled_symbols=tiled or [],
        tiled_symbol_trip_counts=trips or {},
    )


def make_chained_op_specs(ops: tuple = ("add", "mul"), **overrides) -> list:
    """The ops of one kernel, each threading its result into the next.

    Every op but the last writes an ``lx`` intermediate that the next op reads,
    which is the contract saying this kernel owns it: not passed in, no address,
    and nothing outside the kernel can reach it.  The fresh inputs and the final
    output are HBM args, numbered across the whole kernel rather than per op.
    """
    lx = {"lx": 0}
    specs, next_arg = [], 0
    for level, op in enumerate(ops):
        # The first op reads two fresh inputs; every later one reads the previous
        # result and one fresh input.  Only the last op's output is HBM.
        threaded = [] if level == 0 else [f"buf{level - 1}"]
        fresh = [f"arg{next_arg + i}" for i in range(2 - len(threaded))]
        specs.append(
            make_op_spec(
                op,
                names=[*threaded, *fresh, f"buf{level}"],
                allocations=[
                    *([lx] if threaded else []),
                    *([None] * len(fresh)),
                    None if level == len(ops) - 1 else lx,
                ],
                first_arg_index=next_arg,
                **overrides,
            )
        )
        next_arg += len(fresh)
    return specs


def make_nested_op_spec(*, levels: list, **overrides) -> tuple:
    """One op inside a loop nest, as ``(nest, op, loops)``.

    ``levels`` is ``[(symbol, trip count), ...]`` outermost-first, and it states
    the nest once: the ``scf.for`` trip counts, the op's ``tiled_symbols``
    (innermost-first, one entry per level) and its ``tiled_symbol_trip_counts``
    all come from it, so they cannot disagree.  A tiled op addresses through
    ``advances`` rather than coordinates, and its iteration space is empty.

    ``loops`` is the enclosing chain outermost-first -- what the plan walk would
    reach the op with, and what ``_levels`` takes.
    """
    spec = make_op_spec(
        coords=[],
        space={},
        tiled=[[symbol] for symbol, _ in reversed(levels)],
        trips=dict(levels),
        **overrides,
    )
    body: list = [spec]
    loops: list = []
    for _symbol, trip in reversed(levels):
        loops.insert(0, LoopSpec(count=trip, body=body))
        body = [loops[0]]
    return loops[0], spec, loops


def make_onstick_sum_specs(op: str = "sum", arrangements: list | None = None) -> list:
    """``sum(x[256, 128], dim=-1)`` on one core, as the frontend projects it.

    ``op`` names the reduction and ``arrangements`` is passed straight through, so
    the same vector serves any arity-1 reduction over the stick -- the shape is the
    fixture's contribution and the op is the caller's.

    The reduction runs along the *stick*, so it consumes both halves of the
    reduced symbol -- the outer-stick chunk index ``floor(c1 / 64)`` and the
    within-stick lane ``c1 % 64`` -- and the output nonetheless has 64 lanes at a
    constant coordinate.  Every number here is the frontend's own: device sizes
    [2, 256, 64] in and [1, 256, 64] out, the output's axis 0 a placeholder and
    its axis 2 the lane the D2H descriptor gathers across.

    Shared rather than local to one test class because both halves of the suite
    want it: the dialect-free plan assertions here, and the golden in
    ``test_ktir_emitter.py``.
    """
    rows, reduced = sympy.symbols("c0 c1")
    stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
    return [
        make_op_spec(
            op,
            is_reduction=True,
            inputs=1,
            arrangements=arrangements,
            sizes=[[2, 256, 64], [1, 256, 64]],
            coords_per_arg=[
                [stick, rows, lane],
                [sympy.Integer(0), rows, sympy.Integer(0)],
            ],
            space={rows: (256, 1), reduced: (128, 1)},
        )
    ]


def make_broadcast_op_spec(form: str = "row") -> OpSpec:
    """One pointwise op with a BROADCAST operand, in one of three forms.

    Named for the map each form produces, not for the op that wants it.  The op is
    whichever registered payload has the arity the form needs, and the geometry is
    the fixture's own: a two-input ``realdiv`` for the forms with a full operand
    beside the broadcast one, and a one-input intrinsic for the splat.

    * ``"row"`` -- the operand carries one coordinate of the middle axis and the
      axes either side of it: ``(d0, d1, d2) -> (d0, 0, d2)``.
    * ``"stat"`` -- the operand is of LOWER rank, one value per coordinate of the
      middle axis, at the head of its stick: ``(d0, d1, d2) -> (d1, 0)``.
    * ``"splat"`` -- one element read back across a whole stick:
      ``(d0, d1) -> (d0, 0)``.

    The two statistic operands are described at a WHOLE STICK per statistic, which
    is what a reduction writes; the one-element tile the map reads is derived
    (``_reads_stick_head``), not stated here.
    """
    d0, d1, d2 = sympy.symbols("d0 d1 d2")
    zero = sympy.Integer(0)
    if form == "row":
        return make_op_spec(
            "realdiv",
            sizes=[[16, 512, 64], [16, 1, 64], [16, 512, 64]],
            coords_per_arg=[[d0, d1, d2], [d0, zero, d2], [d0, d1, d2]],
        )
    if form == "stat":
        return make_op_spec(
            "realdiv",
            sizes=[[16, 512, 64], [512, 64], [16, 512, 64]],
            coords_per_arg=[[d0, d1, d2], [d1, zero], [d0, d1, d2]],
        )
    if form == "splat":
        return make_op_spec(
            "layernormscale",
            inputs=1,
            sizes=[[512, 64], [512, 64]],
            coords_per_arg=[[d0, zero], [d0, d1]],
            arrangements=[ElementArrangement.EXX2, ElementArrangement.STANDARD],
        )
    raise ValueError(f"make_broadcast_op_spec: unknown form {form!r}")


def make_statistic_reader_specs(reader: str = "realdiv") -> list:
    """A reduction, and a pointwise stage that READS the statistic it wrote.

    Two stages over one HBM buffer, each in its own iteration-space namespace:

    * stage 0 reduces ``[2, 256, 64]`` along the stick and writes ``buf0`` -- which
      the frontend describes as ``[1, 256, 64]`` at coordinates ``[0, c0, 0]``: a
      placeholder axis, the rows, and a whole stick per statistic at a constant
      coordinate.
    * stage 1 reads ``buf0`` at that same ``[1, 256, 64]`` description, alongside a
      full-extent operand, and writes a full-extent output.

    So the two ends of one buffer disagree about its rank and about how much of
    each stick is read, which is exactly the disagreement a consumer of a
    reduction has.  The ops are incidental: any reduction and any two-operand
    payload state the same thing.
    """
    rows, reduced = sympy.symbols("c0 c1")
    stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
    zero = sympy.Integer(0)
    statistic = [zero, rows, zero]
    produce = make_op_spec(
        "sum",
        is_reduction=True,
        inputs=1,
        names=["x0", "buf0"],
        sizes=[[2, 256, 64], [1, 256, 64]],
        coords_per_arg=[[stick, rows, lane], statistic],
        space={rows: (256, 1), reduced: (128, 1)},
    )
    e0, e1, e2 = sympy.symbols("e0 e1 e2")
    consume = make_op_spec(
        reader,
        inputs=2,
        names=["x1", "buf0", "out0"],
        sizes=[[2, 256, 64], [1, 256, 64], [2, 256, 64]],
        coords_per_arg=[[e0, e1, e2], [zero, e1, zero], [e0, e1, e2]],
        space={e0: (2, 1), e1: (256, 1), e2: (64, 1)},
        first_arg_index=2,
    )
    # ``buf0`` is ONE buffer at ONE index, which the per-spec numbering cannot
    # know; said here for the same reason ``TestAStageOwnsItsViews`` says it.
    consume.args[1].arg_index = 1  # buf0, as stage 0 numbered it
    consume.args[2].arg_index = 3  # out0, after x1
    return [produce, consume]


def make_linked_op_specs(
    ops: tuple = ("abs", "max"),
    *,
    reductions: tuple = (False, True),
    edges: tuple = ((0, 1),),
    dangling: tuple = (),
    prefixes: tuple | None = None,
    link: dict | None = None,
    links: dict | None = None,
    out_sizes: dict | None = None,
    out_coords: dict | None = None,
    in_sizes: dict | None = None,
    onstick: bool = False,
    chunks: int | None = None,
    rows: int = 256,
    lanes: int = 64,
    dtype: DataFormats = FP16,
    row_division: int = 1,
) -> list:
    """A kernel's OpSpec vector described TOPOLOGICALLY: stages and their links.

    The fixture for every fusion test, and it is deliberately not keyed to any
    entry in the table.  A test that can only be written by naming ``abs`` is a
    test of the shipped entry, not of the mechanism, and the mechanism is what
    has to keep working when the table grows.  So a test here says "two stages,
    scratchpad link, the second reduces" and the ops are incidental.

    One stage per entry in ``ops``, in vector order, each addressing its operands
    in its OWN iteration-space namespace (``d``, ``e``, ``f``, ...) -- which is
    what a real vector looks like once the LX producer's loop order has been
    aligned, and the difference a rebuild must not splice together.

    * ``ops`` / ``reductions`` -- one per stage.  A reduction folds axis 0 away,
      so it writes [1, rows, lanes] at a constant coordinate; a pointwise stage
      is the identity, which is what makes it access-preserving.
    * ``edges`` -- ``(producer, consumer)`` over stage indices.  The producer's
      output becomes a buffer memory planning placed and the consumer reads it,
      at the producer's own extent.  A stage with no incoming edge reads one
      fresh HBM input; a stage whose output feeds no edge writes HBM.
    * ``dangling`` -- stage indices whose output is an owned buffer even though
      nothing in the vector reads it.  It exists so that "the kernel does not own
      the link" and "nothing reads the link" can be tested separately: with
      ``edges=()`` alone the producer writes HBM and the first condition would
      decline before the second was reached.
    * ``link`` -- the allocation every internal buffer gets; ``links`` overrides
      it per producing stage, so one vector can mix ``lx`` and ``hbm_pool``.
    * ``onstick`` -- whether the reduction runs along the stick, which is the
      only thing the viability predicate turns on.  Off-stick folds the
      outer-stick axis of [64, 256, 64], which a ``linalg.reduce`` can say;
      on-stick folds the lanes of [2, 256, 64], the reduced symbol being
      consumed as both an outer-stick chunk ``floor/lanes`` and a within-stick
      ``Mod``, which only a ``linalg.generic`` can say.
    * ``out_sizes`` / ``out_coords`` / ``in_sizes`` -- per stage, each breaking
      one thing a rewrite may depend on: an output extent or a coordinate
      permutation makes a stage stop being access-preserving, and an input
      extent makes a consumer disagree with its producer about the shared
      buffer's shape (which nothing in the contract forbids, and which is the
      only way to observe whose description of it a rebuild kept).
    * ``lanes`` follows the format's stick (64 at fp16, 32 at fp32) and
      ``row_division`` divides the row axis of every stage, each in its own
      symbols.

    The four shapes the design is written against are all one call:

    * A, the pair the shipped entry collapses -- the default;
    * B, two stages nothing combines -- ``ops=("exp", "max")``;
    * C, three stages whose two links have different fates --
      ``ops=("exp", "abs", "max")``, ``edges=((0, 1), (1, 2))``;
    * D, one intermediate with two readers -- ``ops=("abs", "max", "sum",
      "add")``, ``edges=((0, 1), (0, 2), (1, 3), (2, 3))``.
    """
    if len(ops) != len(reductions):
        raise ValueError("make_linked_op_specs: one reduction flag per op")
    prefixes = prefixes or tuple(chr(ord("d") + index) for index in range(len(ops)))
    link = link or ({"lx": 0x1000} if onstick else {"hbm_pool": 0x2000})
    links, out_sizes, out_coords, in_sizes = (
        links or {},
        out_sizes or {},
        out_coords or {},
        in_sizes or {},
    )
    chunks = (2 if onstick else 64) if chunks is None else chunks
    full_size = [chunks, rows, lanes]
    reduced_size = [1, rows, lanes]

    def geometry(prefix: str) -> tuple[list, list, dict]:
        """One stage's coordinates for a full and a reduced buffer, and its space.

        Both spellings come from one place, so two stages differ only in the
        letter their symbols carry and never in how the shape is described.
        """
        s0, s1, s2 = sympy.symbols(f"{prefix}0:3")
        if onstick:
            full = [sympy.floor(s1 / lanes), s0, sympy.Mod(s1, lanes)]
            reduced = [sympy.Integer(0), s0, sympy.Integer(0)]
            space = {s0: (rows, row_division), s1: (chunks * lanes, 1)}
        else:
            full = [s0, s1, s2]
            reduced = [sympy.Integer(0), s1, s2]
            space = {s0: (chunks, 1), s1: (rows, row_division), s2: (lanes, 1)}
        return full, reduced, space

    specs: list = []
    next_arg = 0
    for index, op in enumerate(ops):
        full, reduced, space = geometry(prefixes[index])
        incoming = [producer for producer, consumer in edges if consumer == index]
        outgoing = index in dangling or any(producer == index for producer, _ in edges)
        if incoming:
            names = [f"t{producer}" for producer in incoming]
            allocations = [dict(links.get(producer, link)) for producer in incoming]
            # A read is described at the extent its producer wrote, in the
            # READER's symbols: whose description is kept is the fuser's problem,
            # so the fixture must not make the two accidentally identical.
            sizes = [
                in_sizes.get(index)
                or (reduced_size if reductions[producer] else full_size)
                for producer in incoming
            ]
            coords = [
                reduced if reductions[producer] else full for producer in incoming
            ]
        else:
            names, allocations = [f"x{index}"], [None]
            sizes, coords = [in_sizes.get(index) or full_size], [full]
        reduction = reductions[index]
        # A reduction folds axis 0 away; a pointwise stage is the IDENTITY on its
        # first operand, which is what makes it access-preserving and is why the
        # result follows that operand rather than the stage's nominal extent.
        result_size = reduced_size if reduction else sizes[0]
        result_coords = reduced if reduction else coords[0]
        spec = make_op_spec(
            op,
            inputs=len(names),
            is_reduction=reduction,
            names=[*names, f"t{index}" if outgoing else f"out{index}"],
            sizes=[*sizes, out_sizes.get(index) or result_size],
            coords_per_arg=[*coords, out_coords.get(index) or result_coords],
            allocations=[
                *allocations,
                dict(links.get(index, link)) if outgoing else None,
            ],
            dtype=dtype,
            space=space,
            first_arg_index=next_arg,
        )
        specs.append(spec)
        next_arg += sum(1 for arg in spec.args if arg.arg_index >= 0)
    return specs


def make_absmax_pair(**overrides) -> list:
    """``amax(abs(x), ...)``: shape A, and the only fixture that names the entry.

    A thin instantiation of the topological builder, kept because two tests are
    legitimately about the shipped table -- that it recognises this pair, and
    that it declines the one operand format the device gets wrong -- and naming
    the ops is the whole content of those. Everything else goes through the
    builder directly.
    """
    return make_linked_op_specs(ops=("abs", "max"), **overrides)


def make_plan_fusion(**overrides) -> ktir.PlanFusion:
    """A table entry defined by the test, defaulting to a two-slot collapse.

    Synthetic on purpose.  Testing the driver through ``PLAN_FUSIONS`` can only
    ever exercise one pattern and one result name, so every such test is the same
    scenario in different clothes and a driver that hard-coded the shipped
    entry's shape would pass all of them.

    The default ``result_op`` is a name no recipe defines, because the fuser is
    not supposed to consult ``RECIPES``: the emitter does that afterwards, and a
    spec it refuses is a refusal the table is entitled to produce.
    """
    entry = {
        "name": "probe",
        "pattern": (("abs", False), ("max", True)),
        "result_op": "fused",
        "why": "a probe entry, defined by the test that uses it",
    }
    entry.update(overrides)
    return ktir.PlanFusion(**entry)  # type: ignore[arg-type]


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
        specs = [make_op_spec(), make_op_spec(divisions={"d1": 2})]
        self._rejects(specs, "different work divisions")

    def test_ragged_work_division_rejected(self):
        """A division that does not divide the axis evenly has no per-core tile."""
        specs = [make_op_spec(divisions={"d1": 7})]  # 512 / 7 is not a whole tile
        self._rejects(specs, "do not divide evenly")

    # -- spec-tree shape ---------------------------------------------------

    def test_unimplemented_op_rejected(self):
        self._rejects([UnimplementedOp(op="atan2")], "unimplemented op 'atan2'")

    def test_unexpected_entry_rejected(self):
        self._rejects(["not a spec"], "unexpected spec entry str")

    def test_family_mismatch_rejected(self):
        """An ``add`` asked for as a reduction: the recipe is what has an
        emission, so the request is refused rather than emitted elementwise."""
        specs = [make_op_spec(is_reduction=True)]
        self._rejects(specs, "registered as NAMED")

    def test_unregistered_op_rejected(self):
        """An op with no recipe is rejected, and the message names what exists."""
        self.assertNotIn("atan2", ktir.KtirBuilder.RECIPES)
        self._rejects([make_op_spec("atan2")], "op 'atan2' is not supported yet")

    # -- per-op roles ------------------------------------------------------

    def test_multiple_outputs_rejected(self):
        specs = [make_op_spec(inputs=1, outputs=2)]
        self._rejects(specs, "expected exactly one output, got 2")

    def test_wrong_arity_rejected(self):
        self._rejects([make_op_spec(inputs=1)], "'add' expects 2 inputs, got 1")

    def test_in_place_rejected(self):
        """The output names an input, which is the aliasing this cannot emit."""
        specs = [make_op_spec(names=["arg0", "arg1", "arg0"])]
        self._rejects(specs, "in-place ops (input aliases output)")

    def test_stretched_operand_rejected(self):
        """A unit extent on an axis the operand's coordinate says it WALKS.

        Not a broadcast this can spell: the axis carries an iteration symbol, so
        the map row for it is a dim rather than a constant, and one element under a
        dim that runs 16 is a stretch.  A broadcast operand says so with a
        constant coordinate instead.
        """
        specs = [make_op_spec(sizes=[[1, 512, 64]])]
        self._rejects(specs, "not a stretch of it")

    def test_a_broadcast_operand_of_a_named_linalg_op_rejected(self):
        """A named ``linalg`` op states its own (identity) indexing, so a derived
        map row has nowhere to go, and the scalar spelling that would go in a
        generic's body is not something any recipe declares."""
        specs = [
            make_op_spec(
                sizes=[[16, 512, 1]],
                coords_per_arg=[
                    [*sympy.symbols("d0:2"), sympy.Integer(0)],
                    sympy.symbols("d0:3"),
                    sympy.symbols("d0:3"),
                ],
            )
        ]
        self._rejects(specs, "named linalg op, which states its own indexing")

    # -- per-buffer --------------------------------------------------------

    def test_non_kernel_argument_buffer_rejected(self):
        """arg_index stays -1 for LX / HBM-pool buffers; only HBM is emitted.

        Set here rather than asked of the builder: an HBM buffer that is *also*
        not a kernel argument is the contradiction under test, and the builder
        ties -1 to a non-HBM allocation precisely so it cannot produce one.
        """
        specs = [make_op_spec()]
        specs[0].args[0].arg_index = -1
        self._rejects(specs, "is not a kernel argument")

    def test_symbolic_trip_count_rejected(self):
        nest = LoopSpec(count=sympy.Symbol("s0"), body=[make_op_spec()])
        self._rejects([nest], "trip count s0 is symbolic")

    def test_unsupported_dtype_rejected(self):
        self._rejects([make_op_spec(dtype=DataFormats.SENINT8)], "unsupported device")
        self.assertNotIn(DataFormats.SENINT8, ktir.ElemTypes.NAMES)

    def test_baked_non_hbm_allocation_rejected(self):
        """An allocation that is neither HBM nor one this emitter threads.

        Set here for the same reason as ``test_non_kernel_argument_buffer``: the
        builder would read an unrecognised allocation as an intermediate and give
        it -1, which is a different rejection than the one under test.
        """
        specs = [make_op_spec(baked=True)]
        specs[0].args[0].allocation = {"somewhere_new": 0x1000}
        self._rejects(specs, "is not HBM-allocated", bake_addresses=True)

    def test_threaded_input_without_a_producer_rejected(self):
        """An lx buffer this kernel reads but does not produce: threading it has
        no value to read, so it needs materialising."""
        specs = [make_op_spec(allocations=[{"lx": 0x1000}])]
        self._rejects(specs, "no op in this kernel produces it")

    def test_baked_unassigned_hbm_address_rejected(self):
        # [make_op_spec()] leaves every 'hbm' address None.
        self._rejects([make_op_spec()], "unassigned 'hbm' address", bake_addresses=True)


class TestRejectionsThroughGenerateKtir(unittest.TestCase):
    """``generate_ktir`` surfaces the rejections *without* reaching the dialect.

    These would pass vacuously if ``generate_ktir`` validated after importing
    ``mlir_ktdp``; they run here precisely because it validates first.
    """

    def test_family_mismatch_unsupported(self):
        specs = [make_op_spec(is_reduction=True)]
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs)

    def test_unregistered_op_unsupported(self):
        specs = [make_op_spec()]
        specs[0].op = "atan2"
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_atan2_0", specs)

    def test_ragged_work_division_unsupported(self):
        specs = [make_op_spec(divisions={"d1": 7})]
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs)

    def test_unknown_option_is_a_typeerror(self):
        """Options are PlanOptions fields; a typo is not silently ignored."""
        with self.assertRaises(TypeError) as ctx:
            ktir.generate_ktir("k", [make_op_spec()], bake_address=True)
        self.assertIn("bake_address", str(ctx.exception))


class TestPlanOptions(unittest.TestCase):
    """The caller's one choice, and it is about spelling, not capability.

    What the kernel does comes from the contract, so there is nothing here to
    turn a feature on with: no core count and no loop mode (a ``LoopSpec`` is a
    loop).
    """

    def test_defaults_are_the_canonical_form(self):
        options = ktir.PlanOptions()
        self.assertFalse(options.bake_addresses)  # symbolic addresses

    def test_options_are_only_about_spelling(self):
        self.assertEqual(
            sorted(f.name for f in dataclasses.fields(ktir.PlanOptions)),
            ["bake_addresses"],
        )


class TestWorkDivision(unittest.TestCase):
    """The grid, and the per-core tile, as ``iteration_space`` states them.

    ``work_division.py`` has already turned ``config.sencores`` into a per-symbol
    division by the time the emitter sees a spec, so the emitter reads the
    contract and never the config -- the same source the SDSC path reads as its
    work slices.
    """

    def test_an_undivided_space_is_one_core(self):
        plan = ktir.build_kernel_plan([make_op_spec()])
        self.assertEqual(plan.grid, (1,))
        self.assertEqual(plan.divisions, ())

    def test_the_grid_is_the_product_of_the_divisions(self):
        plan = ktir.build_kernel_plan([make_op_spec(divisions={"d1": 32})])
        self.assertEqual(plan.grid, (32,))
        self.assertEqual(plan.divisions, (ktir.Division(symbol="d1", div=32, inner=1),))

    def test_two_divided_symbols_are_mixed_radix(self):
        """Outermost-first, and ``inner`` is that symbol's stride in the grid."""
        plan = ktir.build_kernel_plan([make_op_spec(divisions={"d0": 2, "d1": 4})])
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
        plan = ktir.build_kernel_plan([make_op_spec(divisions={"d1": 32})])
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
            ktir.build_kernel_plan([make_op_spec(divisions={"d2": 2})])
        self.assertIn("no device axis of the output", str(ctx.exception))


class TestKernelPlan(unittest.TestCase):
    """What ``build_kernel_plan`` returns: the func signature, before any emission."""

    def test_param_entries_are_ordered_by_arg_index(self):
        specs = [make_op_spec()]
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
        plan = ktir.build_kernel_plan([make_op_spec()])
        # Every 'hbm' address in the fixture is None and never read: the bases
        # are func arguments.
        self.assertEqual([e.base_elements for e in plan.parameters], [None] * 3)

    def test_baked_form_resolves_bases_in_elements(self):
        plan = ktir.build_kernel_plan(
            [make_op_spec(baked=True)],
            ktir.PlanOptions(bake_addresses=True),
        )
        # fp16: 2 bytes per element, so the byte slot halves.
        self.assertEqual(
            [e.base_elements for e in plan.parameters],
            [0, (1 << 34) // 2, (2 << 34) // 2],
        )

    def test_repeated_buffer_is_registered_once(self):
        specs = [make_op_spec()] + [make_op_spec()]
        plan = ktir.build_kernel_plan(specs)
        self.assertEqual(len(plan.buffers), 3)


class TestBaseAddressElements(unittest.TestCase):
    """``_base_address_elements`` in isolation, with no dialect and no config."""

    @staticmethod
    def _arg(allocation):
        """One input carrying ``allocation``, taken out of a whole spec."""
        return make_op_spec(allocations=[None, allocation]).args[1]

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
        for arg in make_op_spec().args:
            self.assertFalse(ktir.is_internal(arg))

    def test_planning_placed_it_means_the_kernel_owns_it(self):
        for allocation in ({"lx": 0x1000}, {"hbm_pool": 0x2000}):
            with self.subTest(allocation=allocation):
                spec = make_op_spec(allocations=[None, None, allocation])
                self.assertTrue(ktir.is_internal(spec.args[-1]))

    def test_an_unrecognised_allocation_is_not_threaded(self):
        """Threading is chosen on a positive signal, so an allocation this
        emitter does not know reaches the buffer rejection instead."""
        spec = make_op_spec(allocations=[None, None, {"somewhere_new": 0}])
        self.assertFalse(ktir.is_internal(spec.args[-1]))

    def test_a_threaded_buffer_nothing_reads_is_rejected(self):
        """An intermediate whose consumer is in another kernel: not stored, and
        not read here either, so the op that produced it would write nowhere."""
        specs = [make_op_spec(allocations=[None, None, {"lx": 0x1000}])]
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("nothing in this kernel", str(ctx.exception))


class TestRecipes(unittest.TestCase):
    """One recipe per op, and every surface the plan can pick has an arm."""

    def test_every_recipe_is_complete(self):
        self.assertTrue(ktir.KtirBuilder.RECIPES)
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            with self.subTest(op=op):
                self.assertGreaterEqual(recipe.arity, 1)
                self.assertTrue(recipe.arms)
                # A reader, not the values: resolving one needs an ``op_info``.
                self.assertTrue(recipe.attrs is None or callable(recipe.attrs))
                for index, arm in enumerate(recipe.arms):
                    with self.subTest(arm=index):
                        self.assertIsInstance(arm.kind, ktir.BindingKind)
                        # A thunk, not the builder itself: resolving it here would
                        # need the dialect, which this module deliberately does
                        # not require.
                        self.assertTrue(callable(arm.binding))
                # A one-armed op has to be reachable at every format, so that arm
                # cannot list any: a lone arm claiming a format would make
                # ``Recipe.arm`` refuse every other one.
                if len(recipe.arms) == 1:
                    self.assertEqual(recipe.arms[0].dtypes, ())

        # Every kind is now registered by some arm, so the mirror assertion is
        # worth making: PAYLOAD stopped being a hook nothing reaches when the
        # ``spyreop`` intrinsics landed on it.
        self.assertEqual(
            {arm.kind for r in ktir.KtirBuilder.RECIPES.values() for arm in r.arms},
            set(ktir.BindingKind),
        )

        # Which surface a step gets is the plan's choice, not a recipe's, so
        # completeness on this side is about ``compute`` rather than about any one
        # op: every ``Surface`` must appear as a ``case`` pattern.  Read off the
        # AST because ``case _:`` alone turns a missing arm into a runtime
        # discovery, at which point a module is already half built.
        tree = ast.parse(inspect.getsource(ktir))
        builder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KtirBuilder"
        )
        compute = next(
            node
            for node in builder.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute"
        )
        cased = {
            node.pattern.value.attr
            for node in ast.walk(compute)
            if isinstance(node, ast.match_case)
            and isinstance(node.pattern, ast.MatchValue)
            and isinstance(node.pattern.value, ast.Attribute)
        }
        for surface in ktir.Surface:
            self.assertIn(surface.name, cased, f"compute has no case for {surface}")

    def test_recipe_rejects_a_nonsense_arity(self):
        """A duplicate op name is ruff F601; arity is checked at construction."""
        with self.assertRaises(ValueError):
            ktir.Recipe(arity=0, arms=self._arm())

    def test_a_lone_arm_is_promoted_to_a_tuple(self):
        """``arms=Arm(...)`` and ``arms=(Arm(...),)`` are the same recipe.

        Asserted because the shorthand would otherwise be a second representation
        of the field: anything reading ``recipe.arms`` directly must see a tuple
        however the entry was written, or it iterates an ``Arm``'s attributes.
        """
        arm = self._arm()
        self.assertEqual(ktir.Recipe(arity=1, arms=arm).arms, (arm,))
        self.assertEqual(ktir.Recipe(arity=1, arms=(arm,)).arms, (arm,))
        # And every registered entry has been normalised, whichever form it used.
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            with self.subTest(op=op):
                self.assertIsInstance(recipe.arms, tuple)

    @staticmethod
    def _arm(*dtypes):
        return ktir.Arm(
            kind=ktir.BindingKind.NAMED, binding=lambda: None, dtypes=tuple(dtypes)
        )

    def test_recipe_rejects_an_ambiguous_arm_set(self):
        """The two ways a format could resolve to more than one arm.

        Both are refused where the table is written rather than at the lookup,
        because a table that can be read two ways is wrong however it is read --
        and ``Recipe.arm`` returning the first match would make which arm wins a
        fact about declaration order.
        """
        with self.assertRaises(ValueError):
            ktir.Recipe(arity=1, arms=())
        with self.assertRaises(ValueError):
            # Two arms claiming every unlisted format.
            ktir.Recipe(arity=1, arms=(self._arm(), self._arm()))
        with self.assertRaises(ValueError):
            # Two arms claiming the same format.
            ktir.Recipe(
                arity=1,
                arms=(
                    self._arm(DataFormats.IEEE_INT32),
                    self._arm(DataFormats.IEEE_INT32),
                ),
            )

    def test_an_op_with_two_spellings_resolves_on_the_format(self):
        """``add`` is a named linalg op at floats and a spyreop payload at int32.

        The point of the arms: one entry per op, and the format picks the spelling.
        Asserted on the recipe rather than through a plan so it holds without a
        dialect build -- the bindings stay unresolved thunks.
        """
        recipe = ktir.KtirBuilder.RECIPES["add"]
        self.assertIs(recipe.arm(DataFormats.SEN169_FP16).kind, ktir.BindingKind.NAMED)
        self.assertIs(recipe.arm(DataFormats.IEEE_INT32).kind, ktir.BindingKind.PAYLOAD)
        # Arity is the op's, not the arm's, so both spellings agree on it by
        # construction rather than by two entries happening to match.
        self.assertEqual(recipe.arity, 2)

    def test_an_op_with_one_spelling_reaches_it_at_every_format(self):
        """``sub`` has no integer intrinsic, so its one arm takes every format."""
        recipe = ktir.KtirBuilder.RECIPES["sub"]
        for dtype in (DataFormats.SEN169_FP16, DataFormats.IEEE_INT32, None):
            with self.subTest(dtype=dtype):
                self.assertIs(recipe.arm(dtype).kind, ktir.BindingKind.NAMED)

    def test_the_format_reaches_the_step_and_picks_the_surface(self):
        """An int32 ``add`` plans as a generic, and the step carries the format.

        The whole path in one assertion: the spec's format picks the payload arm,
        the payload arm picks ``Surface.GENERIC`` (a scalar builder needs a region),
        and the format lands on the step so emission resolves the same arm without
        seeing the spec.
        """
        spec = make_op_spec("add", dtype=DataFormats.IEEE_INT32)
        [step] = ktir.build_kernel_plan([spec]).steps
        self.assertIs(step.dtype, DataFormats.IEEE_INT32)
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        # The same op at fp16 is the named linalg op, which states its own
        # indexing and so needs no record.
        [float_step] = ktir.build_kernel_plan([make_op_spec("add")]).steps
        self.assertIs(float_step.surface, ktir.Surface.BARE)
        self.assertIsNone(float_step.indexing)

    def test_a_spec_that_mixes_formats_is_refused_by_the_plan(self):
        """No arm resolves a mixed request, so the plan refuses to guess one.

        Taking any single operand's format would emit an intrinsic for the wrong
        type on the others, and the old ``any(... == INT32)`` rule did exactly that
        for one int32 operand among floats.
        """
        spec = make_op_spec("add")
        mixed = dataclasses.replace(
            spec,
            args=[
                dataclasses.replace(spec.args[0], device_dtype=DataFormats.IEEE_INT32),
                *spec.args[1:],
            ],
        )
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan([mixed])
        self.assertIn("mixes device formats", str(ctx.exception))

    def test_a_format_no_arm_takes_is_refused(self):
        """An op with only a claimed arm does not exist at any other format.

        The membership question the two-table arrangement got wrong: an op is
        supported at a format or it is not, and there is no second table to fall
        back out of.
        """
        recipe = ktir.Recipe(arity=1, arms=(self._arm(DataFormats.IEEE_INT32),))
        self.assertIs(recipe.arm(DataFormats.IEEE_INT32).kind, ktir.BindingKind.NAMED)
        with self.assertRaises(NotImplementedError) as ctx:
            recipe.arm(DataFormats.SEN169_FP16)
        self.assertIn("no arm for", str(ctx.exception))

    def test_a_reduction_asked_for_elementwise_is_rejected(self):
        """The other direction of the agreement check, and the dangerous one.

        ``sum``'s binding is a two-operand combiner; with nothing labelled as
        reduced it would be handed a single operand and fail *inside* emission,
        with a half-built module in hand.  Refused by the plan instead.
        """
        specs = [make_op_spec("sum", inputs=1, is_reduction=False)]
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("registered as COMBINER", str(ctx.exception))
        self.assertIn("elementwise", str(ctx.exception))

    def test_emit_asserts_on_an_unplanned_step(self):
        """The emitter's only remaining ``raise`` is this plan-bug guard.

        Called unbound with ``self=None``: the type check happens before any
        builder state is touched, which is why this needs no dialect build.
        ``UnimplementedOp`` cannot reach emission at all now -- a step tree holds
        only steps -- so the guard is about a malformed plan, not a rejected op.
        """
        with self.assertRaises(AssertionError):
            ktir.KtirBuilder.emit(None, [UnimplementedOp(op="atan2")])


class TestReduceSurface(unittest.TestCase):
    """Which of the two reduction shapes a loop nest can be emitted as.

    ``linalg.reduce`` is the compact spelling: hand it the dimensions to fold
    away and it works out the rest itself.  The price is that it can only say a
    reduction that reads its input with one loop per input dimension and leaves
    the surviving dimensions where they were.  Anything else has to be a
    ``linalg.generic``, which spells the correspondence out in full.  These tests
    go straight at that rule -- no spec is involved.
    """

    def test_a_plain_reduction_can_be_a_linalg_reduce(self):
        """Fold away the middle dimension of three, keep the other two in order."""
        self.assertIs(
            ktir._reduce_surface(
                ("parallel", "reduction", "parallel"), (0, 1, 2), (0, 2)
            ),
            ktir.Surface.REDUCE,
        )

    def test_a_reduction_over_the_stick_cannot_be_a_linalg_reduce(self):
        """The on-stick sum, and the reason it is worth a test of its own.

        Judged on its output alone, ``(1, 3)`` reads as "keep dimensions 1 and 3
        of four, fold away 0 and 2" -- which ``linalg.reduce`` says perfectly
        well.  What it cannot say is the input side: three input dimensions
        addressed by a loop nest of four, because the 64 lanes are read as one
        dimension and written as a different one.  ``linalg.reduce`` always reads
        its input with exactly one loop per input dimension.

        So if this rule is ever relaxed to look only at the output, this is the
        test that fails -- and without it the emitter would quietly build a
        two-dimensional ``linalg.reduce`` that sums the wrong elements.
        """
        iters = ("reduction", "parallel", "reduction", "parallel")
        self.assertEqual(
            tuple(d for d, it in enumerate(iters) if it == "reduction"), (0, 2)
        )
        self.assertIs(
            ktir._reduce_surface(iters, (0, 1, 2), (1, 3)), ktir.Surface.GENERIC
        )

    def test_a_reduction_that_also_reorders_cannot_be_a_linalg_reduce(self):
        """It folds dimensions away; it never moves the ones that survive.

        Here the two survivors come out swapped, which the compact spelling has
        no way to express.
        """
        self.assertIs(
            ktir._reduce_surface(
                ("parallel", "reduction", "parallel"), (0, 1, 2), (2, 0)
            ),
            ktir.Surface.GENERIC,
        )


class TestOnlyAReductionOutputIsSqueezed(unittest.TestCase):
    """A pointwise op keeps a size-1 output dimension; only a reduction drops one.

    Dropping a size-1 dimension is safe when a reduction left it behind, because
    nothing was ever written along it.  It is not safe in general, and this spec
    is the counterexample: an ``add`` whose operands and output all carry the
    same size-1 dimension.  It compiles today, and it works precisely *because*
    all three agree on it.  Drop it from the output alone and ``linalg.add``
    would be handed a two-dimensional result against three-dimensional operands,
    which fails when the module is verified -- inside emission, the one place
    nothing is allowed to fail.

    So the drop happens only for a reduction, and this test is the reason.
    """

    @staticmethod
    def _size_one_add():
        rows = sympy.Symbol("c1")
        return [
            make_op_spec(
                size=[1, 256, 64],
                coords=[sympy.Integer(0), rows, sympy.Mod(rows, 64)],
            )
        ]

    def test_a_size_one_dimension_is_kept_when_nothing_is_reduced(self):
        plan = ktir.build_kernel_plan(self._size_one_add())
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.BARE)
        self.assertEqual(step.out.extent, (1, 256, 64))
        for _buf_id, access in step.ins:
            self.assertEqual(access.extent, (1, 256, 64))


class TestAnOutputLaneIsNotATranspose(unittest.TestCase):
    """A reduction may write an axis its input reduced; it may not reorder axes.

    Both shapes reach the same matching walk, and before the broadcast lane had a
    home the on-stick one came out of it with the *wrong* diagnostic: its output
    lane matched no input axis, so it was reported as a permutation needing a
    restickify.  It is not a permutation -- nothing moved -- so the two cases have
    to be told apart, and a refusal that still fires for the real thing is what
    says the first case was widened rather than the check being weakened.
    """

    def test_a_reduced_axis_may_be_written_again(self):
        plan = ktir.build_kernel_plan(make_onstick_sum_specs())
        [step] = plan.steps
        self.assertEqual(step.out.extent, (256, 64))

    def test_reordered_surviving_axes_are_still_refused(self):
        """The same reduction with its two kept axes swapped on the way out."""
        lanes, rows = sympy.symbols("c0 c1")
        stick, lane = sympy.floor(lanes / 64), sympy.Mod(lanes, 64)
        specs = [
            make_op_spec(
                "sum",
                is_reduction=True,
                inputs=1,
                sizes=[[32, 256, 64], [64, 32]],
                coords_per_arg=[[stick, rows, lane], [lane, stick]],
                space={lanes: (2048, 1), rows: (256, 1)},
            )
        ]
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("transpose", str(ctx.exception))


class TestAPayloadWithNoNamedOpGetsAGeneric(unittest.TestCase):
    """An elementwise op the dialect has no named op for, and how it is spelled.

    ``sqrt`` is one: its binding is ``spyreop.sqrt``, a *scalar* builder, so there
    is nothing to call it but a region and the step has to state the identity maps
    itself.  Everything here is the plan's choice, made before any dialect is
    reached, which is why these run without a dialect build.
    """

    def test_every_spyreop_intrinsic_is_a_payload(self):
        """The kind is what puts them on the generic, so it is asserted per op.

        Registered as PAYLOAD and not NAMED: a ``spyreop`` op is not a ``linalg``
        named op, and calling one as if it were would hand a scalar builder tensor
        operands inside emission.
        """
        for op in (
            "exp",
            "sqrt",
            "sigmoid",
            "reciprocal",
            "gelufwd",
            "layernormscale",
            "softplus",
        ):
            with self.subTest(op=op):
                recipe = ktir.KtirBuilder.RECIPES[op]
                [arm] = recipe.arms
                self.assertIs(arm.kind, ktir.BindingKind.PAYLOAD)
                self.assertEqual(recipe.arity, 1)

    def test_the_identity_maps_are_stated_rather_than_implied(self):
        plan = ktir.build_kernel_plan([make_op_spec("sqrt", inputs=1)])
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(step.reduce_dims, ())
        # Rank 3, one map per input and then the result: the operand and the
        # destination are read one element at a time in the same order.
        self.assertEqual(step.indexing.iters, ("parallel",) * 3)
        self.assertEqual(step.indexing.maps, ((0, 1, 2), (0, 1, 2)))

    def test_a_scalar_argument_is_read_at_plan_time(self):
        """softplus's two scalars land on the step, so emission derives nothing.

        The values are on the record and the reader is not: what ``op_info`` looks
        like is a fact about the request, and the step is what emission sees.
        """
        spec = make_op_spec(
            "softplus",
            inputs=1,
            op_info={"constants": {"softplusBeta": 1.0, "softplusThresh": 20.0}},
        )
        [step] = ktir.build_kernel_plan([spec]).steps
        self.assertEqual(step.attrs, (("beta", 1.0), ("threshold", 20.0)))

    def test_an_op_with_no_scalar_arguments_carries_none(self):
        """``attrs`` is empty for every op that is a function of its operands.

        Asserted over every registered recipe rather than one, so an ``attrs``
        reader added to an op that does not want one shows up here.
        """
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            # A reduction wants coordinates that actually reduce, which its own
            # fixtures own; the claim here is about the pointwise ops.
            # ``arm(None)`` is the arm an unlisted format reaches, which is the one
            # ``make_op_spec``'s fp16 args resolve to.
            if (
                recipe.attrs is not None
                or recipe.arm(None).kind is ktir.BindingKind.COMBINER
            ):
                continue
            with self.subTest(op=op):
                spec = make_op_spec(op, inputs=recipe.arity)
                [step] = ktir.build_kernel_plan([spec]).steps
                self.assertEqual(step.attrs, ())

    def test_a_missing_scalar_argument_is_the_plans_problem(self):
        """An ``op_info`` without the constants fails in the plan, not in emission.

        This is what reading the scalars at plan time buys: the failure arrives
        before ``KtirBuilder.create``, so there is no half-built module in hand.
        """
        with self.assertRaises(KeyError):
            ktir.build_kernel_plan([make_op_spec("softplus", inputs=1)])


class TestStepFieldsAgreeWithTheSurface(unittest.TestCase):
    """The price of two optional fields with one reader each, charged in one test.

    ``indexing`` is carried by the surface that reads it and by no other, and a
    nest with a reduced dim is never a bare named op.  Both are invariants of the
    plan rather than of any one fixture, so they are asserted over every accepted
    fixture in this file at once -- which is what stops the minimal record's
    optional fields drifting into a bug nobody's own test covers.
    """

    @staticmethod
    def _accepted_fixtures() -> dict:
        """Every spec list in this file that ``build_kernel_plan`` accepts."""
        n_stick, m = sympy.symbols("n_stick m")
        nest, _spec, _loops = make_nested_op_spec(
            levels=[(n_stick, 2), (m, 256)],
            size=[1, 1, 64],
            advances=[16384 * n_stick + 64 * m] * 3,
        )
        rows = sympy.Symbol("c1")
        lanes = sympy.Symbol("c0")
        stick, lane = sympy.floor(lanes / 64), sympy.Mod(lanes, 64)
        return {
            "pointwise": [make_op_spec()],
            "divided": [make_op_spec(divisions={"d1": 32})],
            "chained": make_chained_op_specs(("add", "mul")),
            "nested": [nest],
            "unit_axis_pointwise": [
                make_op_spec(
                    size=[1, 256, 64],
                    coords=[sympy.Integer(0), rows, sympy.Mod(rows, 64)],
                )
            ],
            "nonstick_reduction": [
                make_op_spec(
                    "sum",
                    is_reduction=True,
                    inputs=1,
                    sizes=[[32, 256, 64], [1, 32, 64]],
                    coords_per_arg=[
                        [stick, rows, lane],
                        [sympy.Integer(0), stick, lane],
                    ],
                    space={lanes: (2048, 32), rows: (256, 1)},
                )
            ],
            "onstick_reduction": make_onstick_sum_specs(),
            # A pointwise op whose payload is a ``spyreop`` intrinsic: the other
            # way onto ``Surface.GENERIC``, and the one that reaches it with no
            # reduced dim, which is the combination the two claims below split on.
            "intrinsic": [make_op_spec("sqrt", inputs=1)],
            "intrinsic_with_attrs": [
                make_op_spec(
                    "softplus",
                    inputs=1,
                    op_info={
                        "constants": {"softplusBeta": 1.0, "softplusThresh": 20.0}
                    },
                )
            ],
        }

    @staticmethod
    def _steps(steps):
        for step in steps:
            if isinstance(step, ktir.LoopStep):
                yield from TestStepFieldsAgreeWithTheSurface._steps(step.body)
            else:
                yield step

    def test_the_fixtures_cover_every_surface(self):
        """A vacuous invariant is the failure mode, so the coverage is asserted."""
        surfaces = {
            step.surface
            for specs in self._accepted_fixtures().values()
            for step in self._steps(ktir.build_kernel_plan(specs).steps)
        }
        self.assertEqual(surfaces, set(ktir.Surface))

    def test_a_generic_is_the_only_step_that_states_its_indexing(self):
        for name, specs in self._accepted_fixtures().items():
            for position, step in enumerate(
                self._steps(ktir.build_kernel_plan(specs).steps)
            ):
                with self.subTest(fixture=name, step=position):
                    self.assertIs(
                        step.indexing is not None, step.surface is ktir.Surface.GENERIC
                    )
                    if step.reduce_dims:
                        self.assertIsNot(step.surface, ktir.Surface.BARE)


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
    _nest, spec, loops = make_nested_op_spec(
        levels=[(n_stick, 2), (m, 256)],  # outermost-first
        inputs=1,
        names=["a", "c"],
        sizes=[[1, 1, 64], [1, 64]],
        advances=[16384 * n_stick + 64 * m, 64 * n_stick],
        allocations=[{"hbm": 0}, {"hbm": 0}],
        dtype=DataFormats.IEEE_FP16,
    )
    return spec, loops


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
        call, so this only checks the reading."""
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
        # ``elems`` is the CALLER's answer: which element type an access reads a
        # buffer at is the op's business (``Recipe.unfused``), not the arg's.
        a_access = ktir._access(
            a, a.device_size, a_q, a_layout, ktir.ElemTypes.of(a.device_dtype)
        )
        # The tile extent is device_size, which is what tiling already baked in.
        self.assertEqual(a_access.extent, (1, 1, 64))
        # Per view dim, the step each level takes: dim 0 <- n_stick, dim 1 <- m,
        # dim 2 <- nothing, i.e. the constant zero the kernel spells as %c0.
        self.assertEqual(a_access.index_coeffs, ((1, 0), (0, 1), (0, 0)))

        c_layout, c_q = ktir._solve_layout(c, levels)
        c_access = ktir._access(
            c, c.device_size, c_q, c_layout, ktir.ElemTypes.of(c.device_dtype)
        )
        self.assertEqual(c_access.extent, (1, 64))
        self.assertEqual(c_access.index_coeffs, ((1, 0), (0, 0)))

    def test_untiled_access_sits_at_the_view_origin(self):
        """Depth zero is the general answer, not a special case."""
        arg = make_op_spec().args[0]
        layout, q = ktir._solve_layout(arg, [])
        self.assertEqual(layout.extent, (16, 512, 64))
        self.assertEqual(q, [])
        access = ktir._access(
            arg, arg.device_size, q, layout, ktir.ElemTypes.of(arg.device_dtype)
        )
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


class TestStagesAreNumbered(unittest.TestCase):
    """``_stages`` hands each ``ComputeStep`` its own stage, over the whole tree.

    DECISION pinned: one compute is one stage, and the numbering is the walk's --
    not the emitter's, which sees a loop body as a fresh insertion point and
    could not tell a body's first step from the kernel's.

    LIMITATION forcing per-stage numbers at all: two stages cannot share one
    memory view (the backend's ``ComputeGroupExtraction`` aborts on
    ``op->use_empty()``), so the emitter has to ask "which stage" before it can
    answer with a view -- and the answer has to be unique per compute.
    """

    @staticmethod
    def _two_stages_in_one_body() -> LoopSpec:
        """``(a + b) * c`` at one row per iteration of a two-level nest.

        Tiled rather than coordinate-addressed, because a spec in a loop body is,
        and the point of the fixture is that the body's stages are numbered by
        the same counter as a top-level one's.
        """
        n_stick, m = sympy.symbols("n_stick m")
        tiled = {
            "coords": [],
            "space": {},
            "tiled": [[m], [n_stick]],  # innermost-first
            "trips": {n_stick: 2, m: 256},
            "size": [1, 1, 64],
            "advances": [16384 * n_stick + 64 * m] * 3,
        }
        return LoopSpec(
            count=2,
            body=[
                LoopSpec(
                    count=256,
                    body=[
                        make_op_spec(
                            "add",
                            names=["arg0", "arg1", "buf0"],
                            allocations=[None, None, {"lx": 0}],
                            **tiled,
                        ),
                        make_op_spec(
                            "mul",
                            names=["buf0", "arg2", "buf1"],
                            allocations=[{"lx": 0}, None, None],
                            first_arg_index=2,
                            **tiled,
                        ),
                    ],
                )
            ],
        )

    def test_one_op_is_stage_zero(self):
        [step] = ktir.build_kernel_plan([make_op_spec()]).steps
        self.assertEqual(step.stage, 0)

    def test_each_op_in_a_chain_is_its_own_stage(self):
        steps = ktir.build_kernel_plan(make_chained_op_specs(("add", "mul"))).steps
        self.assertEqual([step.stage for step in steps], [0, 1])

    def test_a_loop_body_continues_the_kernels_count(self):
        """The counter is the plan's, so recursion cannot restart it at zero."""
        plan = ktir.build_kernel_plan([self._two_stages_in_one_body()])
        [outer] = plan.steps
        [inner] = outer.body
        self.assertEqual([step.stage for step in inner.body], [0, 1])


_TABLE_DEFAULT = object()


def fuse(specs, table=_TABLE_DEFAULT) -> tuple:
    """The rewritten vector, dropping the report.

    Two helpers rather than one, because "what came out" and "what it cost" are
    separate assertions and a test that unpacked both would be stating the one it
    is not about.
    """
    return _fuse(specs, table)[0]


def fuse_report(specs, table=_TABLE_DEFAULT) -> ktir.FusionReport:
    """The report, dropping the vector."""
    return _fuse(specs, table)[1]


def _fuse(specs, table):
    """``apply_plan_fusions``, with the shipped table left as the default.

    A sentinel and not ``None``, because the empty table is a table several tests
    pass on purpose and it is falsy.
    """
    if table is _TABLE_DEFAULT:
        return ktir.apply_plan_fusions(specs)
    return ktir.apply_plan_fusions(specs, table)


def _symbols_of(spec) -> set:
    """Every free symbol in ``spec``'s operands' device coordinates.

    The yardstick for "did the rewrite keep one namespace", and
    ``iteration_space`` alone is not it: a spec inside a loop nest legitimately
    addresses tile symbols that only ``tiled_symbols`` names.
    """
    return {
        symbol
        for arg in spec.args
        for coordinate in arg.device_coordinates
        for symbol in getattr(coordinate, "free_symbols", ())
    }


class FusionCase(unittest.TestCase):
    """Base for the plan-fusion tests: the per-kernel summary is pinned off.

    Every successful fusion concedes something, and a concession logs a summary
    at ``warning``, so without this each test here would emit a paragraph through
    whatever handler the suite happens to have.  Switched back on by the one
    class that is about the report, which also makes the flag's effect something
    those tests state rather than inherit.
    """

    def setUp(self):
        patch = mock.patch.object(spyre_config, "plan_fusion_warn", False)
        patch.start()
        self.addCleanup(patch.stop)

    def assertDeclined(self, specs, table=_TABLE_DEFAULT, reason=None):
        """``specs`` came back unfused, in order, with nothing raised.

        A fusion table is not a legality authority: a sequence it does not
        recognise must come back untouched so the ordinary per-spec checks say
        what is wrong with it (for a surviving ``abs``, "op 'abs' is not
        supported yet").  Every decline test therefore asserts the ops are still
        the ops, and that no concession was recorded for a rewrite that did not
        happen.

        ``reason`` is a fragment of the ``debug`` line, and passing it is what
        makes a decline test about its own condition: the fixtures differ from a
        fusable one in one respect each, but several of the conditions would
        decline several of them, so an outcome-only assertion passes for the
        wrong reason and keeps passing when a condition is deleted.  Omitted only
        where there is no line to match -- a span the pattern never selected is
        not worth one, since every span in every kernel is one.
        """
        if reason is not None:
            with self.assertLogs(ktir.logger, level="DEBUG") as captured:
                vector, report = _fuse(specs, table)
            declines = [
                record.getMessage()
                for record in captured.records
                if "declines" in record.getMessage()
            ]
            self.assertTrue(any(reason in message for message in declines), declines)
        else:
            vector, report = _fuse(specs, table)
        self.assertEqual([spec.op for spec in vector], [spec.op for spec in specs])
        self.assertEqual(report.concessions, [])
        return vector


class TestPlanFusionRewrite(FusionCase):
    """Recognition and rewrite: what replaces a span the table names.

    DECISION: a matched span becomes one OpSpec, built from the last stage with
    the deleted stage's source spliced in by identity alone.
    LIMITATION: the device has one instruction for the pair and this emitter has
    no recipe for the pointwise half of it, so the choice is one fused op or a
    refused kernel -- there is no unfused kernel to fall back to.
    """

    def test_a_two_stage_span_the_table_names_becomes_one_stage(self):
        """DECISION: recognise the span positionally and replace the whole of it.

        LIMITATION forcing it: the emitter has no ``abs`` recipe, deliberately --
        the backend refuses a standalone ``math.absf``, so the fused reduction is
        the only shape the device takes an absolute value in.  If this fails the
        emitter is handed a bare ``abs`` and every ``amax(abs(x))`` kernel is
        refused, which is a working path today.

        Would be unnecessary if: the backend gained a pointwise absolute value
        and an intermediate could cross a compute op, at which point the two
        stages would simply both emit.
        """
        vector = fuse(make_absmax_pair())
        self.assertEqual([spec.op for spec in vector], ["absmax"])
        self.assertTrue(vector[0].is_reduction)
        self.assertIn("absmax", ktir.KtirBuilder.RECIPES)

    def test_the_fused_spec_keeps_the_survivors_access_and_the_sources_identity(self):
        """DECISION: splice buffer IDENTITY across, never access geometry.

        LIMITATION forcing it: each stage writes its coordinates against its own
        iteration-space symbols, and nothing downstream compares two stages'
        namespaces.  A rebuild that took the deleted stage's ``device_size`` or
        ``device_coordinates`` along with its buffer produces a kernel that
        compiles, runs, and addresses the wrong elements -- the worst failure
        available here, and the only one with no diagnostic at any layer.

        Both halves are asserted because either alone passes for the wrong
        reason.  ``in_sizes`` makes the two stages disagree about the shared
        buffer's extent on purpose: without it the assertion is vacuous, since an
        access-preserving producer's input, output and the survivor's read of
        that output are all the same list.
        """
        pair = make_absmax_pair(in_sizes={1: [32, 256, 64]})
        producer_in, producer_out = pair[0].args
        survivor_read, survivor_out = pair[1].args

        [fused] = fuse(pair)
        read, out = fused.args

        # Identity: the producer's own source, so the link is gone entirely.
        self.assertEqual(read.name, producer_in.name)
        self.assertEqual(read.arg_index, producer_in.arg_index)
        self.assertEqual(read.allocation, producer_in.allocation)
        self.assertEqual(read.device_dtype, producer_in.device_dtype)
        self.assertNotEqual(read.name, producer_out.name)

        # Access: the survivor's own, in the survivor's namespace and not the
        # producer's.
        self.assertEqual(read.device_size, survivor_read.device_size)
        self.assertNotEqual(read.device_size, producer_in.device_size)
        self.assertEqual(read.device_coordinates, survivor_read.device_coordinates)
        self.assertEqual(out.device_coordinates, survivor_out.device_coordinates)
        survivor_prefix = str(next(iter(pair[1].iteration_space)))[0]
        symbols = {
            str(symbol)
            for coordinate in read.device_coordinates
            for symbol in coordinate.free_symbols
        }
        self.assertTrue(symbols)
        self.assertTrue(all(s.startswith(survivor_prefix) for s in symbols), symbols)
        # And the survivor's iteration space, which is what ``_divisions`` reads.
        self.assertEqual(fused.iteration_space, pair[1].iteration_space)

    def test_the_pattern_matches_on_the_reduction_flag_as_well_as_the_name(self):
        """DECISION: a pattern slot is ``(op name, is_reduction)``, not a name.

        LIMITATION forcing it: ``_steps`` checks each spec's ``is_reduction``
        against whether its recipe's arm accumulates, and refuses a disagreement.
        A table that matched on the name alone would collapse a span whose second
        stage is an elementwise ``max`` into a reduction spec carrying
        ``is_reduction=False``, and the refusal would arrive from the recipe
        table naming the fused op -- with nothing pointing back at the fusion.

        Would be unnecessary if: OpSpec named the reduction in the op rather than
        in a flag beside it.
        """
        elementwise = make_linked_op_specs(reductions=(False, False))
        self.assertDeclined(elementwise)
        self.assertEqual([spec.op for spec in fuse(make_linked_op_specs())], ["absmax"])

    def test_a_vector_no_entry_matches_comes_back_as_the_same_objects_in_order(self):
        """DECISION: outside a matched span the fuser is the identity, by object.

        LIMITATION forcing it: this runs over the OpSpec vector of every kernel
        the emitter plans, fusable or not, and later passes mutate ``TensorArg``s
        in place -- so a spec rebuilt identically is a spec whose later edits go
        to a copy nothing emits.  Identity (``is``) and not equality for exactly
        that reason.
        """
        specs = make_chained_op_specs(("add", "mul", "sub"))
        vector = fuse(specs)
        self.assertEqual([spec.op for spec in vector], ["add", "mul", "sub"])
        for original, returned in zip(specs, vector, strict=True):
            self.assertIs(original, returned)

    def test_the_stages_around_a_collapsed_span_survive(self):
        """DECISION: a match rewrites its own span and nothing around it.

        LIMITATION forcing it: the fuser splices a shorter vector together by
        index, so an off-by-one silently swallows the stage before or after the
        span, and a missing stage surfaces downstream as a buffer nobody wrote
        rather than as a fusion bug.
        """
        before, after = make_op_spec("add"), make_op_spec("mul")
        vector = fuse([before, *make_absmax_pair(), after])
        self.assertEqual([spec.op for spec in vector], ["add", "absmax", "mul"])
        self.assertIs(vector[0], before)
        self.assertIs(vector[2], after)


class TestPlanFusionDeclines(FusionCase):
    """Every condition the rewrite checks, and the vector it hands back.

    DECISION: each condition DECLINES rather than raises.
    LIMITATION: the emitter's own per-spec checks are the ones that can say what
    is wrong with an unfused vector; a fusion that refused would replace "op
    'abs' is not supported yet" with a message about a table the user never
    asked for.
    """

    def test_a_producer_the_pattern_does_not_name_is_left_alone(self):
        """DECISION: decline anything the table does not recognise; never guess.

        Shape B -- two stages nothing combines.  LIMITATION forcing it: there is
        no KTIR cost model, so nothing here can justify inventing a fusion; the
        table says only which sequences the DEVICE computes as one instruction,
        and ``exp`` into ``max`` is not one of them.  Collapsing it would compute
        ``max(|x|)``, which is a silently wrong answer rather than a refusal.
        """
        self.assertDeclined(make_linked_op_specs(ops=("exp", "max")))

    def test_a_stage_between_the_two_is_not_a_match(self):
        """DECISION: match positionally over the vector's own step order.

        LIMITATION forcing it: a dataflow matcher would find pairs this misses,
        but would then have to prove that reordering them into adjacency is
        legal -- a scheduling decision, and this layer has no standing to make
        one.  Adjacency is not relied on for soundness; the reader count is.
        """
        producer, consumer = make_absmax_pair()
        self.assertDeclined([producer, make_op_spec("add"), consumer])

    def test_stages_with_no_link_between_them_are_not_a_match(self):
        """DECISION: the shapes matching is not enough; the dataflow must be there.

        LIMITATION forcing it: the rewrite repoints the survivor's read at the
        producer's source, which is only the same value if the survivor was
        reading the producer's result.  Here it reduces a different buffer, so
        folding the producer away deletes a write nothing replaces AND hands the
        reduction an operand it was never given.

        The producer still writes an owned buffer (``dangling``), so the only
        thing missing is the dataflow: otherwise this would decline for the
        ownership condition below and the two would share one test.
        """
        self.assertDeclined(
            make_linked_op_specs(edges=(), dangling=(0,)),
            reason="read 0 time(s)",
        )

    def test_a_link_the_kernel_does_not_own_is_not_deleted(self):
        """DECISION: only a buffer memory planning placed may be deleted.

        LIMITATION forcing it: ``_readers`` can only see the vector it is given,
        which is this kernel.  ``lx`` / ``hbm_pool`` is the contract's own way of
        saying nothing outside the kernel can reach the buffer, so ownership is
        what makes "read once here" mean "read once anywhere".  An ``hbm`` link
        is a real buffer another kernel may read, and deleting its producer
        leaves that reader on stale memory -- a wrong answer in a kernel that is
        not even the one that was rewritten.

        Would be unnecessary if: the fuser were handed the whole graph's reads
        rather than one kernel's.
        """
        self.assertDeclined(
            make_absmax_pair(link={"hbm": None}),
            reason="not a buffer this kernel owns",
        )

    def test_a_producer_that_resizes_is_not_access_preserving(self):
        """DECISION: the deleted producer's input and output must be the same shape.

        LIMITATION forcing it, and it is measured: the backend does NOT catch
        the result.  A broadcasting producer fuses into a reduction whose memory
        view is [2, 256, 64] over a 128-element buffer, and dbo-opt accepts it --
        a silent out-of-bounds read.  So this condition is the only thing between
        a broadcast and a wrong answer, and it has to be here.
        """
        self.assertDeclined(
            make_absmax_pair(out_sizes={0: [32, 256, 64]}),
            reason="not access-preserving",
        )

    def test_a_producer_that_moves_elements_is_not_access_preserving(self):
        """DECISION: same condition, in the spelling an extent check cannot see.

        LIMITATION forcing it: a transposing producer writes every element
        somewhere else at the SAME extent, so reading its source instead reads
        the right values in the wrong order -- and every size in the emitted
        kernel is correct, so nothing downstream has anything to object to.
        """
        d0, d1, d2 = sympy.symbols("d0 d1 d2")
        self.assertDeclined(
            make_absmax_pair(out_coords={0: [d1, d0, d2]}),
            reason="not access-preserving",
        )

    def test_a_link_with_two_readers_is_not_deleted(self):
        """DECISION: the link must be read exactly once, by the survivor.

        Shape D -- ``a = abs(x); amax(a, -1) + sum(a, -1)``, and the case
        adjacency gets wrong: the second consumer sits AFTER the pair, so the
        pair is still adjacent and a purely positional matcher fuses, deleting
        the producer of a buffer the ``sum`` still reads.

        LIMITATION forcing it: a deleted buffer has no memory behind it, so that
        reader has nothing to read.  ``_check_threaded_buffers`` does catch the
        wreckage downstream, but only because the buffer happens to be threaded,
        and its message then blames the surviving reader for a decision the
        fusion took.

        Would be unnecessary if: the fusion could keep the producer standing and
        add a reader path -- which is what this shape actually wants, and what
        needs an address for the intermediate first.
        """
        vector = self.assertDeclined(
            make_linked_op_specs(
                ops=("abs", "max", "sum", "add"),
                reductions=(False, True, True, False),
                edges=((0, 1), (0, 2), (1, 3), (2, 3)),
            ),
            reason="read 2 time(s)",
        )
        self.assertEqual([spec.op for spec in vector], ["abs", "max", "sum", "add"])

    def test_the_viability_predicate_declines_fp32_on_stick_and_not_fp16(self):
        """DECISION: decline a form the device computes INCORRECTLY.

        LIMITATION forcing it, measured on device: at fp32 with the reduction
        running along the stick, the backend emits its SFP splat/reduce at fp16
        mode over 4-byte lanes.  The kernel compiles and returns NaN or ~1e38
        with no diagnostic anywhere.  Declining leaves the producer, which has no
        recipe, so the kernel is REFUSED -- a refusal in place of a wrong answer
        is the entire purchase, and there is no working kernel being given up
        because this emitter cannot build the unfused one either.

        The fp16 half is not decoration: a predicate returning False
        unconditionally would pass without it.  Both fixtures are asserted to
        reach the same surface, so the format is the only input that differs.

        Would be unnecessary if: the backend picked its SFP mode from the
        operand format.
        """
        fp16 = make_absmax_pair(onstick=True)
        fp32 = make_absmax_pair(onstick=True, dtype=DataFormats.IEEE_FP32, lanes=32)
        for pair in (fp16, fp32):
            self.assertIs(ktir._reduction_surface(pair[1]), ktir.Surface.GENERIC)

        self.assertDeclined(fp32, reason="is not viable on this operand")
        self.assertEqual([spec.op for spec in fuse(fp16)], ["absmax"])

    def test_a_viability_predicate_that_cannot_decide_declines(self):
        """DECISION: an undecidable viability question declines, it does not raise.

        LIMITATION forcing it: viability is answered by the same derivations the
        emission will run, and a derivation is entitled to refuse a shape it does
        not handle.  Propagating that out of the fuser turns a missing fusion
        into a crash whose message blames the surface derivation instead of
        saying the op is unsupported.
        """

        def undecidable(fused):
            raise NotImplementedError("no surface for this shape")

        cases = {
            "answers no": (lambda fused: False, "is not viable on this operand"),
            "cannot answer": (undecidable, "no surface for this shape"),
        }
        for label, (viable, reason) in cases.items():
            with self.subTest(viable=label):
                entry = make_plan_fusion(viable=viable)
                self.assertDeclined(make_linked_op_specs(), (entry,), reason=reason)


class TestFusionReport(FusionCase):
    """What a successful fusion says it gave up.

    DECISION: report per kernel and per buffer, at ``debug`` in detail and at
    ``warning`` in summary, and reclaim nothing.
    LIMITATION: memory placement runs long before this does, and this emitter
    never issues the device allocate that would carry a placement out, so what a
    fusion drops is planning's STRATEGY for the buffer and not a reservation --
    there is nothing to give back because nothing was held.  A predicted-runtime
    figure is computed from the pre-fusion vector and this runs too late to
    correct it.  Reporting is the whole of what is available.
    """

    def setUp(self):
        super().setUp()
        self._warn = mock.patch.object(spyre_config, "plan_fusion_warn", True)
        self._warn.start()
        self.addCleanup(self._warn.stop)

    def test_shape_a_concedes_one_strategy_and_one_price(self):
        """DECISION: name the buffer, the space and the exact byte count.

        LIMITATION forcing it: nothing carries the placement out and nothing
        reclaims it, so the report is the entire discharge of the concession --
        and an unattributed modelling error is indistinguishable from an
        unnoticed one.  Real numbers, both spaces: if the size or the space
        drifts, the log keeps printing confidently and the figure it prints is no
        longer one anybody measured.  The figure is planning bookkeeping, not
        device capacity: this path issues no allocate for either space.
        """
        cases = {
            # Off-stick: the link is [64, 256, 64] fp16 in the HBM pool.
            "hbm_pool": (make_absmax_pair(), 64 * 256 * 64 * 2),
            # On-stick: [2, 256, 64] fp16, small enough for the scratchpad.
            "lx": (make_absmax_pair(onstick=True), 2 * 256 * 64 * 2),
        }
        for space, (pair, nbytes) in cases.items():
            with self.subTest(space=space):
                link = pair[0].args[-1].name
                report = fuse_report(pair)
                self.assertEqual(
                    [(item.kind, item.buf_id) for item in report.concessions],
                    [
                        (ktir.ConcessionKind.ABANDONED_STRATEGY, link),
                        (ktir.ConcessionKind.STRANDED_COST, link),
                    ],
                )
                [strategy] = report.of(ktir.ConcessionKind.ABANDONED_STRATEGY)
                self.assertIn(f"in {space} at {nbytes} bytes", strategy.detail)
                self.assertIn("no memory was held to reclaim", strategy.detail)
                [price] = report.of(ktir.ConcessionKind.STRANDED_COST)
                self.assertIn(repr(pair[0].op), price.detail)
        self.assertEqual(cases["hbm_pool"][1], 2097152)
        self.assertEqual(cases["lx"][1], 65536)

    def test_only_the_buffer_the_rewrite_deleted_is_conceded(self):
        """DECISION: derive the concessions from the RESULT, per buffer.

        So the report doubles as a check on the rewrite: a buffer listed here is
        one the rewrite chose to delete, and a rewrite that deleted something
        still needed would say so.  LIMITATION forcing the granularity: two links
        of one vector can have different fates, so a per-kernel or per-fusion
        statement could not tell them apart -- and inflating the figure with
        buffers that survive is as misleading as omitting it.

        The survivor is given a scratchpad output here, which is what a fused
        reduction whose result another op in the same kernel consumes would have.
        Without that the test cannot discriminate: with the survivor writing HBM,
        neither of its args is an owned buffer and skipping it makes no
        difference.
        """
        pair = make_absmax_pair()
        pair[1].args[-1].allocation = {"lx": 0x3000}
        report = fuse_report(pair)
        self.assertEqual({item.buf_id for item in report.concessions}, {"t0"})

    def test_a_vector_with_two_links_concedes_only_the_one_it_collapsed(self):
        """Shape C: three stages whose two links have different fates.

        DECISION: the surviving link is not this mechanism's business -- it is
        conceded nothing, because nothing was conceded about it.
        LIMITATION forcing the shape to exist at all: the surviving link crosses
        a compute op, which the backend aborts on, and this emitter has no way to
        give it an address.  So the honest report is one that mentions the
        deleted link and is silent about the other; the day the other one can be
        materialised, that silence is what has to change.
        """
        specs = make_linked_op_specs(
            ops=("exp", "abs", "max"),
            reductions=(False, False, True),
            edges=((0, 1), (1, 2)),
        )
        vector, report = _fuse(specs, _TABLE_DEFAULT)
        self.assertEqual([spec.op for spec in vector], ["exp", "absmax"])
        self.assertEqual({item.buf_id for item in report.concessions}, {"t1"})

    def test_a_kernel_no_fusion_touched_concedes_nothing_and_says_nothing(self):
        """DECISION: the report is empty when nothing was rewritten, and silent.

        LIMITATION forcing it: the summary is at ``warning`` and on by default,
        so it is seen by users who did not ask for it.  One that fired on kernels
        the table never touched would be pure noise on every compile this backend
        does, and a warning that fires unconditionally is a warning nobody reads.
        """
        report = fuse_report(make_chained_op_specs(("add", "mul")))
        self.assertEqual(report.concessions, [])
        with self.assertNoLogs(ktir.logger, level="DEBUG"):
            report.log()

    def test_a_concession_from_inside_a_loop_body_reaches_the_kernels_report(self):
        """DECISION: one report per kernel, merged across nesting levels.

        LIMITATION forcing the merge: the fuser recurses into loop bodies because
        the plan reads ops at every depth, so a report built where the fusion
        happens is one report per nest level -- and a summary line per level says
        nothing a reader can act on.
        """
        nest = LoopSpec(count=4, body=[LoopSpec(count=8, body=make_absmax_pair())])
        report = fuse_report([nest])
        self.assertEqual({item.buf_id for item in report.concessions}, {"t0"})

    def test_each_concession_is_at_debug_and_one_summary_is_at_info(self):
        """DECISION: two channels, and the summary is INFO, not WARNING.

        LIMITATION forcing the level: every concession here is raised by a
        SUCCESSFUL fusion, so the summary fires on every working absmax compile,
        and what it names costs nothing measurable -- an abandoned strategy is a
        placement this emitter never carries out for any internal buffer, so no
        memory is held and none is lost.  A warning on every working compile,
        naming nothing actionable, is how a project teaches people to ignore
        warnings.  Promote it the day an abandoned strategy is shown to change a
        placement decision elsewhere; until then this test is what pins the
        level, and its wording still must not read as a fault.
        """
        report = fuse_report(make_absmax_pair())
        with self.assertLogs(ktir.logger, level="DEBUG") as captured:
            report.log()
        messages = [record.getMessage() for record in captured.records]
        details = [record for record in captured.records if record.levelname == "DEBUG"]
        summaries = [
            record for record in captured.records if record.levelname == "INFO"
        ]
        self.assertEqual(len(details), len(report.concessions))
        self.assertEqual(len(summaries), 1)
        summary = summaries[0].getMessage()
        self.assertIn("ABANDONED_STRATEGY", summary)
        self.assertIn("STRANDED_COST", summary)
        self.assertIn("TORCH_SPYRE_PLAN_FUSION_WARN=0", summary)
        self.assertIn("Nothing is wrong with the kernel", summary)
        for blame in ("error", "invalid", "failed", "corrupt"):
            self.assertNotIn(blame, summary.lower(), messages)

    def test_the_summary_is_suppressed_by_config_and_the_detail_is_not(self):
        """DECISION: the flag silences the summary only, and changes no emission.

        LIMITATION forcing the split: whoever is debugging one buffer's placement
        needs the per-buffer lines whether or not the summary is wanted, and
        whoever silenced the summary did so because it fires on every success --
        not because they wanted the detail gone too.

        Patched on ``config`` rather than in the environment because ``config``
        reads the environment once, at import: an env patch inside a test has no
        effect and the test would pass for the wrong reason.
        """
        report = fuse_report(make_absmax_pair())
        with mock.patch.object(spyre_config, "plan_fusion_warn", False):
            with self.assertNoLogs(ktir.logger, level="WARNING"):
                with self.assertLogs(ktir.logger, level="DEBUG") as captured:
                    report.log()
        self.assertEqual(len(captured.records), len(report.concessions))
        self.assertEqual([spec.op for spec in fuse(make_absmax_pair())], ["absmax"])

    def test_the_plan_owns_the_report_for_its_own_kernel(self):
        """DECISION: the plan holds the report and logs it, once.

        LIMITATION forcing the placement: the fuser is called per spec list and
        recurses, so it cannot know a kernel is complete; the plan is the one
        object that does.  A report nobody holds is also a report step 3 cannot
        read, and materialisation is going to need exactly this list.
        """
        # Captured rather than allowed to escape, which also asserts the plan
        # logs the report exactly once: it is the only caller that may.
        with self.assertLogs(ktir.logger, level="INFO") as captured:
            plan = ktir.build_kernel_plan(make_absmax_pair())
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(
            {item.buf_id for item in plan.fusion_report.concessions}, {"t0"}
        )
        self.assertEqual(ktir.KernelPlan().fusion_report.concessions, [])


class TestPlanFusionStructure(FusionCase):
    """Where the fuser runs from, which is not observable anywhere else."""

    def test_a_span_inside_a_loop_body_is_fused(self):
        """DECISION: recurse into loop bodies.

        LIMITATION forcing it: the plan reads ops at every depth, and this runs
        once at the top of ``add_specs`` rather than being re-invoked per level
        by the step walk as the peephole it replaces was.  A span inside a
        coarse-tiled nest is therefore reached only by this recursion, and if it
        is not reached the nest keeps an op the emitter has no recipe for.

        The stages carry coordinates rather than tile advances because the
        viability predicate has to derive a reduction nest from them.
        """
        nest = LoopSpec(count=4, body=[LoopSpec(count=8, body=make_absmax_pair())])
        [result] = fuse([nest])
        self.assertEqual([spec.op for spec in result.body[0].body], ["absmax"])
        # The rebuilt bodies are lists, which is what ``LoopSpec.body`` declares.
        self.assertIsInstance(result.body, list)
        self.assertIsInstance(result.body[0].body, list)

    def test_a_divided_pair_plans_because_fusion_precedes_the_grid(self):
        """DECISION: fuse on the first line of ``add_specs``, before the grid.

        LIMITATION forcing it: ``_divisions`` insists every op in a kernel ask
        for the same per-symbol work division, symbol names included, and the two
        stages name their divisions in different namespaces -- so a divided
        vector is self-contradictory right up until the fusion deletes one of
        them.  Fusing second would refuse every divided absmax with "the ops in
        this kernel ask for different work divisions", which is the assertion
        below run on the unfused vector.

        Asserted at the plan level, because the ordering is a property of
        ``add_specs`` and nothing else can observe it.

        Would be unnecessary if: the division comparison were positional on the
        radix rather than on the symbol, which is what a multi-stage kernel will
        need anyway.
        """
        pair = make_absmax_pair(row_division=32)
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._divisions(pair)
        self.assertIn("different work divisions", str(ctx.exception))

        plan = ktir.build_kernel_plan(pair)
        self.assertEqual(plan.grid, (32,))
        self.assertEqual(plan.divisions, (ktir.Division(symbol="e1", div=32, inner=1),))
        self.assertEqual([step.op for step in plan.steps], ["absmax"])


class TestFusionDriver(FusionCase):
    """The driver, asked about entries the shipped table does not contain.

    DECISION: the table is a table -- a pattern and a result name per entry, over
    one shared rewrite.
    LIMITATION: ``PLAN_FUSIONS`` holds one entry, so every test that goes through
    it exercises one pattern and one result name.  A driver with those baked in
    would pass all of them, which is why these tests define their own entries.
    """

    def test_a_pattern_longer_than_the_vectors_tail_is_not_a_match(self):
        """DECISION: the span length is READ from the pattern, not assumed.

        LIMITATION forcing the test: the shipped pattern is a pair, so the slice
        and the advance are both 2 and a hard-coded 2 is invisible.  The vector's
        tail is where a slice that is allowed to come back short does its damage:
        a three-slot entry matching at the last two stages would be handed a span
        with a slot missing -- one the pattern never compared -- so the length
        check has to happen before anything looks at the span.
        """
        specs = make_linked_op_specs(
            ops=("exp", "abs", "max"),
            reductions=(False, False, True),
            edges=((0, 1), (1, 2)),
        )
        entry = make_plan_fusion(pattern=(("abs", False), ("max", True), ("max", True)))
        self.assertDeclined(specs, (entry,))

    def test_a_rewrite_that_declines_leaves_the_vector_untouched(self):
        """DECISION: a rewrite returning None is a decline, not an error.

        LIMITATION forcing it: the conditions live inside the rewrite, so a
        decline is the only way it can say "not this span".  If None were treated
        as a match the vector would lose a stage to a rewrite that refused to
        perform it, and the kernel would be missing a write.

        The entry is the test's own, so what is asserted is the driver's handling
        of a None rather than the shipped table's recognition: the pattern matches
        and the two stages are unlinked, which the rewrite declines whatever
        result name the entry asked for.
        """
        entry = make_plan_fusion()
        self.assertDeclined(
            make_linked_op_specs(edges=(), dangling=(0,)),
            (entry,),
            reason="read 0 time(s)",
        )

    def test_the_driver_asks_no_recipe_about_the_fused_op(self):
        """DECISION: the fused op need not be emittable for the table to produce it.

        LIMITATION forcing it: the survivor is often an op with no recipe, and
        the refusal for an unemittable result belongs to ``_steps``, whose
        message names the op.  A driver that validated the result here would turn
        a clear downstream refusal into a silent non-fusion.
        """
        self.assertNotIn("fused", ktir.KtirBuilder.RECIPES)
        [fused] = fuse(make_linked_op_specs(), (make_plan_fusion(),))
        self.assertEqual(fused.op, "fused")

    def test_each_vector_is_rewritten_by_its_own_entry(self):
        """DECISION: the pattern selects the entry; the table is a set of them.

        LIMITATION forcing the test: with one entry, "declined because the
        pattern did not match" and "declined because nothing was tried" are the
        same observation, and every decline test in this file would pass against
        an empty table.  A driver that ignored ``pattern`` and used the first
        entry would collapse a ``neg`` into an absolute-value reduction -- a
        silently wrong answer.
        """
        table = (
            make_plan_fusion(name="negmax", pattern=(("neg", False), ("max", True))),
            make_plan_fusion(name="absmax", result_op="fused_abs"),
        )
        for producer, expected in (("abs", "fused_abs"), ("neg", "fused")):
            with self.subTest(producer=producer):
                specs = make_linked_op_specs(ops=(producer, "max"))
                self.assertEqual([s.op for s in fuse(specs, table)], [expected])

    def test_a_vector_matching_no_entry_is_untouched_by_either(self):
        """DECISION: the table declines as a whole, not per entry.

        LIMITATION forcing it: one entry declining must not stop the others being
        tried, and none matching must leave the vector for the per-spec checks to
        refuse.
        """
        table = (
            make_plan_fusion(name="negmax", pattern=(("neg", False), ("max", True))),
            make_plan_fusion(),
        )
        self.assertDeclined(make_linked_op_specs(ops=("exp", "max")), table)

    def test_the_first_matching_entry_in_table_order_wins(self):
        """DECISION: overlapping entries are resolved by position, not specificity.

        LIMITATION forcing the test: anyone adding an entry has to know this. An
        earlier general entry shadows a later special one, and the failure mode
        is a fusion into the wrong result with no diagnostic at all.
        """
        first = make_plan_fusion(name="first", result_op="fused_first")
        second = make_plan_fusion(name="second", result_op="fused_second")
        for table, expected in (
            ((first, second), "fused_first"),
            ((second, first), "fused_second"),
        ):
            with self.subTest(first=table[0].name):
                vector = fuse(make_linked_op_specs(), table)
                self.assertEqual([spec.op for spec in vector], [expected])

    def test_every_shipped_entry_is_well_formed(self):
        """DECISION: nothing validates a table entry any more.

        LIMITATION replaced by it: the fields a constructor used to check are
        gone -- there is no survivor index, no thread list and no permission set
        to get wrong.  What is left cannot be validated usefully (whether the
        result name is one ``RECIPES`` defines is ``_steps``' question, and
        deliberately not asked here), so this asserts only that the shipped
        entries carry the parts the driver reads, and that an empty pattern -- the
        one shape that would match everywhere at zero length and never advance the
        walk -- is a loud bug rather than a hang.
        """
        self.assertTrue(ktir.PLAN_FUSIONS)
        for entry in ktir.PLAN_FUSIONS:
            with self.subTest(entry=entry.name):
                self.assertTrue(entry.pattern)
                self.assertTrue(entry.result_op)
                self.assertTrue(entry.why)
        with self.assertRaises(AssertionError) as ctx:
            fuse(make_linked_op_specs(), (make_plan_fusion(pattern=()),))
        self.assertIn("empty pattern", str(ctx.exception))


class TestFusionInvariants(FusionCase):
    """Properties holding for every table and every vector, over several of each.

    DECISION: outside a matched span the fuser is the identity.
    LIMITATION: it runs over the OpSpec vector of every kernel this emitter
    plans, so what almost all of them depend on is not any rewrite but that
    nothing else moved -- and a corrupted vector surfaces downstream as a wrong
    address or a missing step, never as a fusion bug.
    """

    def _vectors(self):
        """No match, a match, a match among other stages, nested, and one stage."""
        return {
            "no match": make_chained_op_specs(("add", "mul", "sub")),
            "one match": make_linked_op_specs(),
            "match in context": [
                make_op_spec("add"),
                *make_linked_op_specs(),
                make_op_spec("mul"),
            ],
            "match in a loop": [LoopSpec(count=4, body=make_linked_op_specs())],
            "single op": [make_op_spec("add")],
        }

    def _tables(self):
        """A spread of tables, including the empty one and the shipped one."""
        return {
            "empty": (),
            "probe": (make_plan_fusion(),),
            "two entries": (
                make_plan_fusion(name="other", pattern=(("neg", False), ("max", True))),
                make_plan_fusion(),
            ),
            "shipped": ktir.PLAN_FUSIONS,
        }

    def test_the_vector_never_grows_and_unmatched_entries_keep_their_identity(self):
        """A fuser may shorten a vector and rewrite its own span; nothing else.

        An extra entry, a reordering, or a copy of a spec it did not match is a
        corrupted kernel.  Identity rather than equality, because a spec rebuilt
        identically is still one the fuser had no business rebuilding: later
        passes mutate args in place and would then be editing a copy.
        """
        for table_name, table in self._tables().items():
            for vector_name, specs in self._vectors().items():
                with self.subTest(table=table_name, vector=vector_name):
                    before = list(specs)
                    vector = fuse(specs, table)
                    self.assertLessEqual(len(vector), len(before))
                    self.assertEqual(list(specs), before, "input list mutated")
                    survivors = [
                        entry for entry in vector if any(entry is x for x in before)
                    ]
                    positions = [
                        next(i for i, x in enumerate(before) if x is entry)
                        for entry in survivors
                    ]
                    self.assertEqual(positions, sorted(positions))

    def test_a_vector_no_entry_matches_is_returned_unchanged(self):
        """The overwhelmingly common case, and the one with no test per table:
        every kernel that is not a fusable span goes through this function and
        must come out the same tuple of the same objects.
        """
        for table_name, table in self._tables().items():
            with self.subTest(table=table_name):
                specs = make_chained_op_specs(("add", "mul", "sub"))
                vector = fuse(specs, table)
                self.assertEqual(len(vector), len(specs))
                for original, returned in zip(specs, vector, strict=True):
                    self.assertIs(original, returned)

    def test_no_rewrite_names_a_symbol_its_survivor_did_not(self):
        """The invariant a fused spec has to satisfy, over several vectors.

        DECISION: keep the survivor's description of every operand.
        LIMITATION forcing the assertion to be here rather than in the code: it
        used to be a post-check inside the driver, which could only ever assert
        it for entries that had declared they might need it.  It is a property of
        every rewrite, so it belongs to whatever exercises them -- and a rewrite
        that splices two namespaces produces a kernel that compiles and reads the
        wrong elements, which no layer below this notices.
        """
        cases = {
            "shipped entry": (make_absmax_pair(), ktir.PLAN_FUSIONS[0]),
            "one namespace throughout": (
                make_linked_op_specs(prefixes=("d", "d")),
                ktir.PLAN_FUSIONS[0],
            ),
            "probe": (make_linked_op_specs(), make_plan_fusion()),
        }
        for name, (specs, entry) in cases.items():
            with self.subTest(case=name):
                survivor = specs[len(entry.pattern) - 1]
                allowed = _symbols_of(survivor) | set(survivor.iteration_space)
                [fused] = fuse(specs, (entry,))
                self.assertTrue(_symbols_of(fused))
                self.assertLessEqual(_symbols_of(fused), allowed)


class TestGenuineAbsmaxRecipe(unittest.TestCase):
    """``RECIPES['absmax']`` has a caller that is not the fusion table.

    ``torch.any`` lowers to a real ``absmax`` reduction (``lower_any_dim`` names
    the reduction type itself), so the frontend can hand this emitter a single
    ``absmax`` OpSpec with no fusion anywhere in sight.  That path cannot
    currently be run end to end, which is exactly why it needs a test here:
    otherwise the recipe looks like a private detail of the peephole and would go
    with it.
    """

    def test_a_standalone_absmax_reduction_plans_without_any_fusion(self):
        rows, reduced = sympy.symbols("c0 c1")
        stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
        spec = make_op_spec(
            "absmax",
            is_reduction=True,
            inputs=1,
            sizes=[[2, 256, 64], [1, 256, 64]],
            coords_per_arg=[
                [stick, rows, lane],
                [sympy.Integer(0), rows, sympy.Integer(0)],
            ],
            space={rows: (256, 1), reduced: (128, 1)},
        )
        self.assertIn("absmax", ktir.KtirBuilder.RECIPES)
        plan = ktir.build_kernel_plan([spec])
        [step] = plan.steps
        self.assertEqual(step.op, "absmax")
        # A reduction's recipe must accumulate, and on-stick is the generic form.
        self.assertIs(
            ktir.KtirBuilder.RECIPES["absmax"].arm(FP16).kind,
            ktir.BindingKind.COMBINER,
        )
        self.assertIs(step.surface, ktir.Surface.GENERIC)


# ---------------------------------------------------------------------------
# What we generate
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

    def test_a_fused_arrangement_is_a_type_and_not_a_stride(self):
        """DECISION pinned: ``EXX2`` selects an element TYPE, nothing else.

        A buffer holding two statistics to a stick keeps the rank, extent and
        row-major strides its ``device_size`` states, because the pair is one
        element of ``!spyreop.fp16_fused`` rather than two of ``f16``.

        LIMITATION forcing it: nothing here can spell a stagger.  The staggered
        arrangements next door are refused for exactly that reason, so an
        arrangement that passes through has to be one whose element ORDER is the
        standard one -- which the fused pair's is, MEASURED against
        ``ktir-spyreop-exx2.mlir`` (``memref<256x64x!spyreop.fp16_fused>``, the
        strides an f16 output of that reduction would have).
        """
        extent, strides = (256, 64), (64, 1)
        self.assertEqual(
            ktir._arrangement_layout(ElementArrangement.EXX2, extent, strides),
            (extent, strides),
        )

    def test_every_label_is_greppable_and_uniquely_owned(self):
        """Each label is raised from exactly one site, so grepping it is exact."""
        source = inspect.getsource(ktir)
        labels = re.findall(r'_unimplemented\(\s*\n?\s*"([^"]+)"', source)
        self.assertEqual(sorted(labels), sorted(set(labels)))
        self.assertEqual(sorted(labels), ["staggered-element-arrangement"])

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
        self.assertEqual(len(messages), 1)
        for message in messages:
            with self.subTest(message=message[:40]):
                for blame in ("dbo-opt", "no consumer", "nothing lowers", "scheduler"):
                    self.assertNotIn(blame, message)


class TestBroadcastOperands(unittest.TestCase):
    """An operand that does not walk every axis of the output.

    DECISION pinned: the operand's map row is READ OFF ITS COORDINATES, and an
    axis it does not walk becomes a CONSTANT result position -- ``None`` in an
    ``Indexing`` row, ``0`` in the emitted affine map.  Any such operand forces
    ``Surface.GENERIC``, because a named ``linalg`` op states its own indexing.

    LIMITATION forcing it: an ``Indexing`` row used to be bare dim indices, so a
    constant position could not be spelled at all, and the three maps the
    hand-written chain needs are all constant positions.

    ``align_reshape_plan`` stays the SWITCH between this and the identity fast
    path, and that is deliberate: deriving the row for an operand alignment says
    already matches would make ``add``'s emitted form hostage to the dim-reuse
    rule ``reduction_indexing`` needs, silently turning a ``linalg.add`` into a
    ``linalg.generic``.
    """

    def test_each_form_derives_the_maps_the_chain_needs(self):
        """The three rows, against the hand-written chain's three maps."""
        for form, maps in (
            ("row", ((0, 1, 2), (0, None, 2), (0, 1, 2))),
            ("stat", ((0, 1, 2), (1, None), (0, 1, 2))),
            ("splat", ((0, None), (0, 1))),
        ):
            with self.subTest(form=form):
                plan = ktir.build_kernel_plan([make_broadcast_op_spec(form)])
                [step] = plan.steps
                self.assertIs(step.surface, ktir.Surface.GENERIC)
                self.assertEqual(step.indexing.maps, maps)
                # Pointwise: every iteration dim is the output's, all parallel.
                self.assertEqual(
                    step.indexing.iters, ("parallel",) * len(step.out.extent)
                )

    def test_an_aligned_operand_still_reaches_the_named_form(self):
        """The fast path, asserted where the derivation would also have applied:
        a plain ``add`` states no indexing at all and comes out as a named op."""
        plan = ktir.build_kernel_plan([make_op_spec()])
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.BARE)
        self.assertIsNone(step.indexing)

    def test_a_broadcast_operand_beside_an_aligned_one_states_both_rows(self):
        """``indexing_maps`` is one attribute, so the aligned operand's identity
        row is stated too -- derived rather than assumed, and equal either way."""
        plan = ktir.build_kernel_plan([make_broadcast_op_spec("row")])
        [step] = plan.steps
        self.assertEqual(step.indexing.maps[0], (0, 1, 2))


class TestArityBeyondTwoAndPerOperandElementTypes(unittest.TestCase):
    """An op with five operands, and one of them read at another element type.

    DECISION pinned: arity is generic (nothing in the plan counts to two), and the
    element type an access reads a buffer AT belongs to the RECIPE
    (``Recipe.unfused``), positions being the inputs in order and then the result.

    LIMITATION forcing the second half: ``element_arrangement`` cannot answer it.
    MEASURED on the real normalisation vector, the flag is propagated to every arg
    naming a statistic buffer -- so it says "this buffer holds two values to a
    stick", which is true, and not "this operand reads them as a pair", which is
    false for two of the four args that carry it.
    """

    @staticmethod
    def _five_inputs(arrangements=None):
        return make_op_spec(
            "layernormnorm",
            inputs=5,
            size=[12, 64, 64],
            dtype=DataFormats.IEEE_FP32,
            arrangements=arrangements,
        )

    def test_five_operands_plan_with_one_map_each_and_one_for_the_result(self):
        plan = ktir.build_kernel_plan([self._five_inputs()])
        [step] = plan.steps
        self.assertEqual(len(step.ins), 5)
        self.assertEqual(len(plan.parameters), 6)
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(step.indexing.maps, ((0, 1, 2),) * 6)

    def test_the_recipe_and_not_the_arrangement_types_an_operand(self):
        """Two operands whose buffers both hold fused pairs; the recipe says only
        ONE of them is read as a pair, and that is the one that is."""
        recipe = ktir.KtirBuilder.RECIPES["layernormnorm"]
        self.assertEqual(recipe.unfused, (1,))  # squares, read as f16
        fused = ElementArrangement.EXX2
        plan = ktir.build_kernel_plan(
            [self._five_inputs([None, fused, fused, None, None, None])]
        )
        [step] = plan.steps
        self.assertEqual(step.ins[1][1].elems.storage, "f32")
        self.assertEqual(step.ins[2][1].elems.storage, "!spyreop.fp32_fused")

    def test_the_result_can_be_the_unfused_position(self):
        """``layernormscale_fused`` returns a plain float out of a buffer the
        frontend flags fused, so the position named is the result."""
        self.assertEqual(ktir.KtirBuilder.RECIPES["layernormscale"].unfused, (1,))
        plan = ktir.build_kernel_plan(
            [
                make_op_spec(
                    "layernormscale",
                    inputs=1,
                    arrangements=[ElementArrangement.EXX2, ElementArrangement.EXX2],
                )
            ]
        )
        [step] = plan.steps
        self.assertEqual(step.ins[0][1].elems.storage, "!spyreop.fp16_fused")
        self.assertEqual(step.out.elems.storage, "f16")

    def test_an_unfused_position_the_op_does_not_have_is_a_typo(self):
        """A recipe is source, so a position past the result fails where it is
        written rather than typing some other operand by accident."""
        with self.assertRaises(ValueError) as ctx:
            ktir.Recipe(
                arity=1,
                arms=ktir.Arm(
                    kind=ktir.BindingKind.PAYLOAD, binding=lambda: None, dtypes=()
                ),
                unfused=(2,),
            )
        self.assertIn("does not have", str(ctx.exception))


class TestReadingAStatisticAtTheHeadOfItsStick(unittest.TestCase):
    """A pointwise stage reading what a reduction wrote.

    DECISION pinned, two halves of one rewrite:

    1. the read is SQUEEZED the way the producer's output was, so the reader's
       access has the rank of the buffer the producer registered;
    2. the read TILES one element on the innermost axis while the VIEW keeps the
       whole stick.

    LIMITATION forcing each.  (1) is the frontend describing one buffer two ways:
    MEASURED, the producer writes ``(256, 64)`` and the consumer's spec says
    ``(1, 256, 64)``, and an access of the wrong rank cannot tile the registered
    view at all.  (2) is a backend constraint with its own negative test,
    ``ktir-spyreop-layernormscale-full-stick.mlir``: a tile covering the whole
    innermost dimension is ``error: the tile covers more than the first element of
    its innermost dimension``, because the mean of squares sits sixteen bytes along
    the mean and a wider tile puts that offset on the next statistic.
    """

    def test_the_reader_has_the_rank_the_producer_registered(self):
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        produce, consume = plan.steps
        [(_x1, _full), (link, statistic)] = consume.ins
        self.assertEqual(link, "buf0")
        self.assertEqual(len(statistic.extent), len(produce.out.extent))
        self.assertEqual(plan.buffers["buf0"].layout.extent, (256, 64))

    def test_the_tile_is_the_stick_head_and_the_view_is_the_whole_stick(self):
        """The negative test's constraint, stated as the two numbers it is about:
        a wider tile is a backend error, and a narrower view would name the wrong
        buffer."""
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        _produce, consume = plan.steps
        [_full, (_link, statistic)] = consume.ins
        self.assertEqual(statistic.extent, (256, 1))
        self.assertEqual(plan.buffers["buf0"].layout.extent[-1], 64)

    def test_the_producer_still_writes_the_whole_stick(self):
        """The asymmetry: the reduction's own output keeps all 64 lanes, which is
        what the opaque reduction needs to get the pair to element 0."""
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        produce, _consume = plan.steps
        self.assertEqual(produce.out.extent, (256, 64))

    def test_the_squeezed_read_derives_the_statistic_map(self):
        """And the two capabilities meet: a rank-reduced operand at a one-element
        tile is what makes the row ``(d1, 0)``."""
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        _produce, consume = plan.steps
        self.assertEqual(consume.indexing.maps, ((0, 1, 2), (1, None), (0, 1, 2)))


class TestAReducingBodyThatIgnoresItsAccumulator(unittest.TestCase):
    """A reduction registered ``COMBINER`` whose binding ignores ``accumulated``.

    DECISION pinned: this needs no mechanism.  The kind says only that the op
    REDUCES -- which is the bit ``_stages`` compares against ``is_reduction`` --
    so a binding that does not use its accumulator reaches the ordinary reducing
    generic, and the plan records exactly what a bare ``sum`` over the stick
    records.

    LIMITATION forcing it: what the body is FOR is a device pattern that replaces
    the whole generic.  Nothing here can check that the pattern matches, so the
    recipe is the only place that can say so, and it does.
    """

    def test_it_reduces_and_is_registered_as_a_combiner(self):
        """Both statements of the one bit agree, so the equality check passes."""
        recipe = ktir.KtirBuilder.RECIPES["exx2"]
        self.assertEqual(recipe.arity, 1)
        self.assertIs(recipe.arm(FP16).kind, ktir.BindingKind.COMBINER)

    def test_the_plan_is_the_ordinary_on_stick_reduction(self):
        specs = make_onstick_sum_specs(
            "exx2", arrangements=[ElementArrangement.STANDARD, ElementArrangement.EXX2]
        )
        plan = ktir.build_kernel_plan(specs)
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(
            step.indexing.iters, ("reduction", "parallel", "reduction", "parallel")
        )
        self.assertEqual(step.indexing.maps, ((0, 1, 2), (1, 3)))
        # The accumulator's type is the output buffer's, and that is the pair:
        # ``exx2_fused`` returns it, so nothing else has to be told.
        self.assertEqual(step.out.elems.value, "!spyreop.fp16_fused")
        self.assertEqual(step.out.extent, (256, 64))


class TestFusedElementType(unittest.TestCase):
    """One buffer's ``element_arrangement`` decides its element TYPE.

    DECISION pinned: the fused pair is a two-key lookup,
    ``(device_dtype, element_arrangement) -> MLIR spelling``, so ``EXX2`` at
    either fp16 format is ``!spyreop.fp16_fused``.

    LIMITATION forcing it: the frontend has no dtype for the pair.  It carries the
    fact as ``ElementArrangement.EXX2`` ("reduction mode: two values per stick"),
    which is why the arrangement is read here and not only in the layout rule.
    """

    def test_exx2_is_the_fused_spelling_of_its_dtype(self):
        for dtype, spelling in (
            (DataFormats.SEN169_FP16, "!spyreop.fp16_fused"),
            (DataFormats.IEEE_FP16, "!spyreop.fp16_fused"),
            (DataFormats.IEEE_FP32, "!spyreop.fp32_fused"),
        ):
            with self.subTest(dtype=dtype):
                fused = ktir.ElemTypes.of(dtype, ElementArrangement.EXX2)
                self.assertEqual((fused.storage, fused.value), (spelling, spelling))
                # The same dtype in the standard order is the plain float: the
                # arrangement is what selects the table.
                self.assertNotEqual(ktir.ElemTypes.of(dtype).storage, spelling)

    def test_a_dtype_with_no_fused_spelling_is_refused(self):
        """An integer pair has no spelling in the dialect, so it is not guessed:
        reading a pair as one plain element is the wrong half of a statistic, and
        it would compile."""
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.ElemTypes.of(DataFormats.IEEE_INT32, ElementArrangement.EXX2)
        self.assertIn("EXX2", str(ctx.exception))

    def test_the_buffer_and_the_access_both_take_the_fused_type(self):
        """The view and the tile agree, because one derivation answers both."""
        specs = [
            make_op_spec(
                "layernormscale",
                inputs=1,
                arrangements=[ElementArrangement.EXX2, ElementArrangement.STANDARD],
            )
        ]
        plan = ktir.build_kernel_plan(specs)
        [step] = plan.steps
        [(_buf_id, source)] = step.ins
        self.assertEqual(source.elems.storage, "!spyreop.fp16_fused")
        self.assertEqual(plan.buffers["arg0"].elems.storage, "!spyreop.fp16_fused")
        # The pair is one element, so the extent is the arg's own device_size.
        self.assertEqual(plan.buffers["arg0"].layout.extent, tuple(ADD_SIZE))
        # And the OUTPUT of this op is not fused: nothing propagates the flag.
        self.assertEqual(step.out.elems.storage, "f16")

    def test_layernormscale_binds_the_fused_form_at_arity_one(self):
        """The frontend hands this op the pair as ONE operand.

        The two-operand ``spyreop.layernormscale`` is what the backend's own
        ``dbo-unfuse-layernormscale`` produces below us, so binding it here would
        need a second operand nobody supplies.
        """
        self.assertEqual(ktir.KtirBuilder.RECIPES["layernormscale"].arity, 1)


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
            layout, _ = fresh._solve_layout(make_op_spec().args[0], [])
            self.assertEqual(layout.extent, (16, 512, 64))

    def test_emission_is_what_needs_the_dialect(self):
        # A *valid* request gets as far as the builder and no further.
        with self._blocked() as fresh, self.assertRaises(ImportError):
            fresh.generate_ktir("k", [make_op_spec()])


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

        # Every explicit raise on the emission path is a malformed-plan assertion:
        # no NotImplementedError and no Unimplemented, which is what a refusal is
        # spelled as.  So none of the functions that *can* refuse -- the labelled
        # guard `_unimplemented`, and the derivations `_levels`, `_solve_layout`
        # and `_access` that call it or raise directly -- is reachable from here.
        #
        # Stated over the raise kinds rather than over those names: a name would
        # have to be kept in step with the source, and two of them once were not,
        # so they silently asserted nothing for as long as that lasted.
        self.assertTrue(raised, "expected the plan-bug assertions to be found")
        self.assertEqual({kind for _, kind in raised}, {"AssertionError"})


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
