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

"""OpSpec -> KTIR emitter.

``generate_ktir`` is an OpSpec consumer: it consumes the finished
``list[OpSpec | LoopSpec]`` kernel contract (the same contract the SDSC bundle
emitter ``generate_bundle`` consumes) and emits **KTDP-dialect MLIR** directly.
The module is built with the ``mlir_ktdp`` Python builders, so the returned
``str(module)`` is canonical, verifier-checked MLIR that the golden snapshot
test consumes without drift.

It uses the OpSpec-reading helpers from ``opspec_utils`` to adapt the OpSpec
information to generate_ktir.

Base addresses are emitted either as func arguments or as baked
``arith.constant``s, selected by the ``bake_addresses`` option.  The baked
form is a temporary dataflow-scheduler#65 workaround, to be reverted when the
backend accepts symbolic addresses.

Structure
---------

``generate_ktir`` is three steps, in this order:

1. ``validate(specs)`` -- a **pure** recursive walk that raises every
   ``NotImplementedError`` the emitter can raise, runs the derivations, and
   returns a ``BufferTable`` of what they produced.  It imports nothing from
   ``mlir_ktdp``, so every rejection is reachable and testable where the dialect
   build is absent.
2. ``KtirBuilder.create(table)`` -- the single ``mlir_ktdp`` import site; owns
   the context and the per-module state.
3. ``emit_specs(b, specs)`` -- a recursive walk over the same spec tree,
   dispatching each ``OpSpec`` through ``KtirBuilder.RECIPES``.

Adding a pointwise op is one ``RECIPES`` entry.  Enabling counted loops is
dropping the ``LoopSpec`` rejection in ``_validate_list`` and threading the
enclosing chain through the two walks: the derivations, ``emit_loop`` and the
per-level access indices are in place, and ``STATUS_TABLE`` says which parts of
them a consumer accepts.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.codegen.opspec_utils import (
    _align_reshape_plan,
    _buf_id,
    _row_major_strides,
)
from torch_spyre._inductor.constants import STAGGERED_EAS
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.pass_utils import coeff_through_floor

# The dialect handles: one module-level name each, None until _load_dialects()
# binds them.  Under TYPE_CHECKING they are the real imports, so `ir.Module` and
# `linalg.add` carry types; at runtime the block does not execute, so importing
# this module requires no dialect build.
if TYPE_CHECKING:
    from mlir_ktdp import ir
    from mlir_ktdp.dialects import arith, func, ktdp, linalg, scf, tensor
else:
    ir = arith = func = ktdp = linalg = scf = tensor = None


def _load_dialects() -> None:
    """Bind the dialect handles into this module, once.  The only import site."""
    global ir, arith, func, ktdp, linalg, scf, tensor
    if ir is not None:
        return
    from mlir_ktdp import ir as _ir
    from mlir_ktdp.dialects import arith as _arith
    from mlir_ktdp.dialects import func as _func
    from mlir_ktdp.dialects import ktdp as _ktdp
    from mlir_ktdp.dialects import linalg as _linalg
    from mlir_ktdp.dialects import scf as _scf
    from mlir_ktdp.dialects import tensor as _tensor

    ir, arith, func, ktdp, linalg, scf, tensor = (
        _ir,
        _arith,
        _func,
        _ktdp,
        _linalg,
        _scf,
        _tensor,
    )


def dialect_available() -> bool:
    """True when the bindings ``_load_dialects`` needs are importable."""
    try:
        _load_dialects()
    except ImportError:
        return False
    return True


# Supported device dtype -> the *name* of the ``mlir_ktdp.ir`` type builder for
# it.  Names, not builder references, so this table stays importable without the
# dialect: ``_elem_types`` reads it (making it the supported-dtype predicate) and
# puts the name in an ``ElemTypes`` record, which is where the builder's
# ``named_type`` resolves it against the imported ``ir``.  The two fp16 device
# formats both map to ``f16``; extend this map (never fall through silently) as
# new dtypes are supported.
_MLIR_ELT_TYPE_NAMES: dict[DataFormats, str] = {
    DataFormats.IEEE_FP16: "F16Type",
    DataFormats.SEN169_FP16: "F16Type",
    DataFormats.IEEE_FP32: "F32Type",
    DataFormats.BFLOAT16: "BF16Type",
}


# ---------------------------------------------------------------------------
# Three kinds of "not supported", kept apart
# ---------------------------------------------------------------------------
#
# 1. *dialect-illegal* -- the KTDP verifiers forbid the construct.  Permanent and
#    structural; the emitter never tries.
# 2. *downstream-guarded* -- KTDP expresses it and this emitter derives it, but a
#    consumer of the emitted KTIR refuses it today.  Raised by
#    ``_downstream_unsupported``: a thin guard in front of a working derivation,
#    deletable on its own without touching the derivation behind it.
# 3. *unspecified* -- nobody has yet defined what should be emitted, so there is
#    no output to guard.  Raised by ``_unspecified``.  There is exactly one such
#    item (see ``STATUS_TABLE``), and it is the only place in this module where a
#    derivation is missing rather than merely fenced off.
#
# Both helpers take a *label* -- a stable token that is also the join key between
# the raise, its row in ``STATUS_TABLE`` and its test.  Grepping one label finds
# all three.  Neither the label nor the message names a consumer, a compiler
# pass, a file or a version: the guard says what the *value* is, and the status
# table says what its state is.


class DownstreamUnsupported(NotImplementedError):
    """A derivation produced a KTIR-expressible answer a consumer refuses today."""


class Unspecified(NotImplementedError):
    """What to emit has never been defined, so no derivation exists yet."""


def _downstream_unsupported(label: str, message: str) -> NoReturn:
    """Refuse an emission that is legal KTDP but not accepted downstream today."""
    raise DownstreamUnsupported(f"OpSpec->KTIR [{label}]: {message}")


def _unspecified(label: str, message: str) -> NoReturn:
    """Refuse an emission nobody has specified.  See ``STATUS_TABLE``."""
    raise Unspecified(f"OpSpec->KTIR [{label}]: {message}")


class Status(enum.Enum):
    """The state of one labelled capability."""

    COMPLETE = "complete"
    DOWNSTREAM_GUARDED = "downstream-guarded"
    UNSPECIFIED = "unspecified"
    INFORMATIONAL = "informational"


@dataclasses.dataclass(frozen=True)
class StatusRow:
    label: str
    status: Status
    derivation: str
    note: str


# THE status table.  One location: there are no "not supported yet" notes
# scattered through the derivations, so what this emitter can and cannot do is
# read here rather than reconstructed by grepping.
#
# It covers *labelled capabilities* only.  A derivation that rejects a specific
# value of its own input -- an unsupported dtype, a tile advance that is not a
# lattice point of its view -- raises about that value at its own site and needs
# no row: the input is wrong (or absent), not the emitter.
STATUS_TABLE: tuple[StatusRow, ...] = (
    StatusRow(
        label="static-view-extent",
        status=Status.COMPLETE,
        derivation="_layout",
        note="integer buffer extents, row-major strides over the solved extent",
    ),
    StatusRow(
        label="dynamic-view-extent",
        status=Status.DOWNSTREAM_GUARDED,
        derivation="_layout",
        note=(
            "a symbolic extent is legal KTDP (construct_memory_view takes dynamic "
            "sizes and strides with matching SSA operands) and _layout derives it "
            "under symbolic_extent='dynamic'; the default arm guards it because "
            "no consumer lowers it yet"
        ),
    ),
    StatusRow(
        label="max-view-extent",
        status=Status.COMPLETE,
        derivation="_layout",
        note=(
            "symbolic_extent='max' bakes OpSpec.symbolic_dim_bounds' upper bound "
            "into a static extent; opt-in because it over-allocates the view"
        ),
    ),
    StatusRow(
        label="standard-element-arrangement",
        status=Status.COMPLETE,
        derivation="_layout",
        note="STANDARD and QFP8CH are plain row-major of the solved extent",
    ),
    StatusRow(
        label="staggered-element-arrangement",
        status=Status.UNSPECIFIED,
        derivation="_layout",
        note=(
            "THE ONE UNSPECIFIED ITEM.  A staggered arrangement is an element "
            "*order* within the stick, so it is a (rank, extent, strides) "
            "selector like any other layout -- but the permutation it leaves "
            "behind has never been written down as numbers, so there is nothing "
            "to emit.  Not a guard in front of working code: the code is absent"
        ),
    ),
    StatusRow(
        label="loop-levels",
        status=Status.COMPLETE,
        derivation="_levels",
        note="integer loop counts, one level per entry of OpSpec.tiled_symbols",
    ),
    StatusRow(
        label="symbolic-loop-count",
        status=Status.DOWNSTREAM_GUARDED,
        derivation="_levels",
        note=(
            "a symbolic LoopSpec.count is expressible as an scf.for bound taken "
            "from a runtime argument; guarded because no consumer accepts one yet"
        ),
    ),
    StatusRow(
        label="tile-advance-decomposition",
        status=Status.COMPLETE,
        derivation="_advance",
        note=(
            "inverts the linearized device_tile_advance_expr into per-dim steps "
            "against the view's own strides; a coefficient that is not a lattice "
            "point of the view, or two levels that cannot be told apart, is "
            "reported at the raise site as the value it is"
        ),
    ),
    StatusRow(
        label="access-tile-offsets",
        status=Status.COMPLETE,
        derivation="_access",
        note=(
            "per-view-dim index expressions from the per-level steps; with no "
            "enclosing levels every expression is the function-entry zero"
        ),
    ),
    StatusRow(
        label="coordinate-set",
        status=Status.INFORMATIONAL,
        derivation="_layout / _access (emitted by the builder)",
        note=(
            "the per-dim bounding integer set on construct_memory_view and "
            "construct_access_tile.  No known reader, and it is in the committed "
            "KTIR goldens, so it keeps being emitted; recorded here so that "
            "'nothing reads it' is written down once instead of being rediscovered"
        ),
    ),
)


def status_of(label: str) -> StatusRow:
    """The one ``STATUS_TABLE`` row for ``label``, or raise."""
    for row in STATUS_TABLE:
        if row.label == label:
            return row
    raise KeyError(f"no STATUS_TABLE row for label {label!r}")


# ---------------------------------------------------------------------------
# Records: what the derivations produce and the builders consume
# ---------------------------------------------------------------------------
#
# Every record is dialect-free -- ints, strings and sympy Exprs -- so the whole
# derivation layer is exercised (and unit-tested) without an ``mlir_ktdp`` build.
# The builders are the only code that turns a record into an ``ir`` object.


@dataclasses.dataclass(frozen=True)
class ElemTypes:
    """The two element types one buffer access involves.

    ``storage`` types the memref (the view), ``value`` types the tensor a load
    produces or a store consumes.  KTDP compares neither against the other --
    ``LoadOp``/``StoreOp`` verify shapes only -- so they are two fields rather
    than one, and today's derivation returns them equal.  Held as the *names* of
    the ``mlir_ktdp.ir`` type builders, so the record stays dialect-free.
    """

    storage: str
    value: str


@dataclasses.dataclass(frozen=True)
class Level:
    """One enclosing loop level.

    ``symbols`` is that level's entry of ``OpSpec.tiled_symbols`` (possibly
    empty: a level that does not tile this op), ``trip`` its trip count, ``iv``
    the ``scf.for`` induction variable once one exists (``None`` while the level
    is being reasoned about outside an emission).
    """

    symbols: tuple[Any, ...]
    trip: int
    iv: Any | None = None


@dataclasses.dataclass(frozen=True)
class Layout:
    """A buffer's device extent and strides, in elements.

    ``extent`` entries are ``int``, or a sympy ``Expr`` for a dynamic extent.
    """

    extent: tuple[Any, ...]
    strides: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class Buffer:
    """One unique buffer referenced by the kernel; sole input to a memory view."""

    buf_id: str  # opspec_utils._buf_id(arg)
    arg_index: int  # position in the kernel call; -1 => not a kernel argument
    elems: ElemTypes
    layout: Layout
    base_elements: int | None  # ELEMENTS for the baked form; None => func arg
    space: str = "HBM"


@dataclasses.dataclass(frozen=True)
class Access:
    """One (OpSpec, TensorArg) access; sole input to an access tile.

    ``extent`` is the tile's own extent, which is ``device_size``: the tile
    extent is what tiling already baked into ``device_size``, while the
    *buffer* extent grows back out of it in ``_layout``.

    ``index_coeffs[i][l]`` is the step level ``l`` takes along view dim ``i``,
    so the index handed to the access tile for dim ``i`` is
    ``sum_l index_coeffs[i][l] * indices[l]``.  This is the design's ``base_map``
    as a matrix; the builder spells it the way the committed loop fixtures do --
    an identity ``base_map`` with one index expression per view dim -- rather
    than as a non-identity map over the induction variables.  The matrix is the
    same either way, so the spelling is one builder function.

    ``elems`` is the access's own element type pair: a tile of an internal buffer
    has no ``Buffer`` to read one from, and a load that reinterprets would differ
    from its buffer's storage type anyway.

    ``buffer`` is what the access is a tile *of*, so a record carries its own way
    back to the view; ``None`` for an internal (threaded) buffer, which has no
    view because it never reaches memory.
    """

    extent: tuple[int, ...]
    index_coeffs: tuple[tuple[int, ...], ...]
    indices: tuple[Any, ...]  # per level, outermost-first
    elems: ElemTypes
    buffer: Buffer | None = None  # None for an internal (threaded) buffer


# ---------------------------------------------------------------------------
# Derivations: one owner per OpSpec / TensorArg field
# ---------------------------------------------------------------------------


def _static(value) -> Any:
    """``value`` as a Python ``int`` when it is one, else ``value`` unchanged."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _mul(lhs, rhs) -> Any:
    """``lhs * rhs``, an ``int`` when both are."""
    return _static(lhs * rhs)


def _elem_types(arg: TensorArg) -> ElemTypes:
    """The storage/value element type pair for ``arg``.

    The only reader of ``device_dtype``, so the unsupported-dtype raise lives
    here.  One ``device_dtype`` means one type on both sides today; a load that
    reinterprets is why the record has two fields.
    """
    name = _MLIR_ELT_TYPE_NAMES.get(arg.device_dtype)
    if name is None:
        raise NotImplementedError(
            f"OpSpec->KTIR: unsupported device dtype {arg.device_dtype!r}"
        )
    return ElemTypes(storage=name, value=name)


def _levels(spec: OpSpec, loops: Sequence[tuple[LoopSpec, Any]] = ()) -> list[Level]:
    """The enclosing loop levels for ``spec``, outermost-first.

    ``loops`` is the enclosing ``(LoopSpec, induction variable)`` chain the walk
    is inside, outermost-first; ``()`` at function level.  ``OpSpec.tiled_symbols``
    is innermost-first with one entry per enclosing level, so this is that list
    reversed and zipped against the loops -- the one place the two orderings
    meet, and therefore the place a mismatch between them is reported.

    Labels: ``loop-levels``, ``symbolic-loop-count``.

    With no enclosing loops the result is ``[]`` because ``tiled_symbols`` is
    empty and there is nothing to zip -- the general answer for a nest of depth
    zero, not a placeholder.
    """
    by_level = list(reversed(list(spec.tiled_symbols)))  # outermost-first
    if len(loops) != len(by_level):
        raise NotImplementedError(
            f"OpSpec->KTIR: op {spec.op!r} carries {len(by_level)} tiled_symbols "
            f"level(s) inside {len(loops)} enclosing loop(s); every enclosing "
            "level must have an entry, even an empty one"
        )
    levels: list[Level] = []
    for (loop, iv), symbols in zip(loops, by_level, strict=True):
        trip = _static(loop.count)
        if not isinstance(trip, int):
            _downstream_unsupported(
                "symbolic-loop-count",
                f"loop trip count {loop.count} is symbolic; trip counts must be "
                "integers to be emitted today",
            )
        for symbol in symbols:
            declared = _static(spec.tiled_symbol_trip_counts.get(symbol, trip))
            if declared != trip:
                raise NotImplementedError(
                    f"OpSpec->KTIR: symbol {symbol} is declared with trip count "
                    f"{declared} but its loop level runs {trip} times"
                )
        levels.append(Level(symbols=tuple(symbols), trip=trip, iv=iv))
    return levels


def _advance_coeffs(arg: TensorArg, levels: Sequence[Level]) -> tuple[int, ...]:
    """Per-level linearized device-element step for ``arg``, one per level.

    ``device_tile_advance_expr`` is a single sum over the per-level symbols, so
    a level's own coefficient is the sum of its symbols' coefficients (a level
    with no symbols does not move this arg, hence ``0``).  ``None`` means the arg
    is not tiled at all: every level's step is ``0``.
    """
    expr = arg.device_tile_advance_expr
    if expr is None:
        return tuple(0 for _ in levels)
    coeffs: list[int] = []
    for level in levels:
        total = 0
        for symbol in level.symbols:
            coeff = _static(coeff_through_floor(expr, symbol))
            if not isinstance(coeff, int):
                raise NotImplementedError(
                    f"OpSpec->KTIR: tile-advance coefficient {coeff} for symbol "
                    f"{symbol} in {expr} is not an integer element count"
                )
            total += coeff
        coeffs.append(total)
    return tuple(coeffs)


def _advance(
    arg: TensorArg, levels: Sequence[Level], strides: Sequence[Any]
) -> list[tuple[int, ...]]:
    """``q[l][i]``: level ``l``'s step along view dim ``i``, in elements.

    The consumer of KTIR linearizes per-dim indices with the view's strides, and
    ``device_tile_advance_expr`` arrives already linearized, so this is that
    linearization's inverse against ``strides``: level ``l``'s coefficient
    ``c_l`` becomes the digit ``c_l / S_i`` on the one dim ``i`` it lands on.

    Coefficients are matched to dims smallest-first, dims innermost-first,
    excluding the trailing dim (a stick dim is never coarse-tiled, so nothing
    steps along it).  One dim per level: a level whose coefficient no remaining
    dim divides is left unassigned, which ``_solve_layout`` reports.  ``strides``
    entries that are not ``int`` are dims whose stride is not solved yet and are
    skipped, which is what makes the joint inner-to-outer solve possible.

    Label: ``tile-advance-decomposition``.

    With no levels the result is ``[]``: there is nothing to decompose.
    """
    coeffs = _advance_coeffs(arg, levels)
    rank = len(strides)
    q = [[0] * rank for _ in levels]
    available = [
        i for i in range(rank - 2, -1, -1) if isinstance(strides[i], int)
    ]  # ascending stride
    for coeff, level_index in sorted((c, l) for l, c in enumerate(coeffs) if c):
        if coeff < 0:
            raise NotImplementedError(
                f"OpSpec->KTIR: negative tile advance {coeff} for level "
                f"{level_index} of {arg.name!r}; a view dim is walked backwards"
            )
        for position, dim in enumerate(available):
            if strides[dim] and coeff % strides[dim] == 0:
                q[level_index][dim] = coeff // strides[dim]
                del available[position]
                break
    return [tuple(row) for row in q]


def _grown_extent(tile: Any, levels: Sequence[Level], steps: Sequence[int]) -> Any:
    """One dim's buffer extent: the tile extent plus what the levels walk over.

    ``E_i = A_i + sum_l q[l][i] * (T_l - 1)``.  The one implementation of that
    formula: ``_solve_layout`` uses it while solving strides and ``_layout`` uses
    it to build the record, so the two cannot disagree.
    """
    extent = tile
    for level, step in zip(levels, steps, strict=True):
        if step:
            extent = extent + step * (level.trip - 1)
    return _static(extent)


def _row_major(extent: Sequence[Any]) -> tuple[Any, ...]:
    """Row-major strides over ``extent``, symbolic entries carried through."""
    if all(isinstance(e, int) for e in extent):
        return tuple(_row_major_strides([int(e) for e in extent]))
    strides: list[Any] = [1] * len(extent)
    for i in range(len(extent) - 2, -1, -1):
        strides[i] = _mul(extent[i + 1], strides[i + 1])
    return tuple(strides)


def _resolve_extent(extent: Any, mode: str, bounds: dict | None) -> Any:
    """One extent under the requested ``symbolic_extent`` mode.

    Labels: ``static-view-extent``, ``dynamic-view-extent``, ``max-view-extent``.
    """
    if isinstance(extent, int):
        return extent
    if mode == "max":
        bounds = bounds or {}
        names = {str(s) for s in getattr(extent, "free_symbols", ())}
        resolved = extent
        for name in names:
            if name not in bounds:
                raise NotImplementedError(
                    f"OpSpec->KTIR: extent {extent} has no bound for {name!r} in "
                    "OpSpec.symbolic_dim_bounds"
                )
            resolved = resolved.subs({name: int(bounds[name][0])})
        resolved = _static(resolved)
        if not isinstance(resolved, int):
            raise NotImplementedError(
                f"OpSpec->KTIR: extent {extent} does not become an integer under "
                f"its bounds {bounds!r}"
            )
        return resolved
    if mode == "dynamic":
        # The dynamic form itself: the extent stays symbolic in the record and
        # the builder spells it as a dynamic memref dim with an SSA size operand.
        return extent
    if mode != "static":
        raise ValueError(f"OpSpec->KTIR: unknown symbolic_extent mode {mode!r}")
    _downstream_unsupported(
        "dynamic-view-extent",
        f"view extent {extent} is symbolic; pass symbolic_extent='dynamic' to "
        "emit the dynamic view (which nothing lowers today) or "
        "symbolic_extent='max' to bake its upper bound",
    )


def _arrangement_layout(
    arrangement: Any, extent: tuple[Any, ...], strides: tuple[Any, ...]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """``(extent, strides)`` adjusted for the element order within a stick.

    ``element_arrangement`` is an element *order*, not a dtype conversion (one
    ``device_dtype`` covers the data type), so it is a layout fact: a rank and
    stride selector, of the shape the SDSC path already uses for a stick split.

    Labels: ``standard-element-arrangement``, ``staggered-element-arrangement``.
    """
    if arrangement in (None, ElementArrangement.STANDARD, ElementArrangement.QFP8CH):
        return extent, strides
    if arrangement in STAGGERED_EAS:
        _unspecified(
            "staggered-element-arrangement",
            f"{arrangement!r} records a non-sequential element order inside the "
            "stick; the permutation has never been written down as numbers, so "
            "there is no rank/stride pair to emit for it",
        )
    raise NotImplementedError(
        f"OpSpec->KTIR: element arrangement {arrangement!r} has no layout rule"
    )


def _layout(
    arg: TensorArg,
    levels: Sequence[Level],
    q: Sequence[Sequence[int]],
    *,
    symbolic_extent: str = "static",
    bounds: dict | None = None,
) -> Layout:
    """``arg``'s buffer extent and strides, given the per-level steps ``q``.

    The buffer extent expands out of the tile extent by what the levels walk
    over (``_grown_extent``); strides are row-major of that extent.  The only
    place an extent becomes a memref dim, so the extent modes and the element
    arrangement are decided here.

    ``bounds`` is ``OpSpec.symbolic_dim_bounds`` (needed only by
    ``symbolic_extent='max'``); it is a parameter rather than a field of ``arg``
    because it lives on the OpSpec.
    """
    tile = [_static(s) for s in arg.device_size]
    extent = tuple(
        _resolve_extent(
            _grown_extent(tile[i], levels, [row[i] for row in q]),
            symbolic_extent,
            bounds,
        )
        for i in range(len(tile))
    )
    extent, strides = _arrangement_layout(
        getattr(arg, "element_arrangement", None), extent, _row_major(extent)
    )
    return Layout(extent=extent, strides=strides)


def _solve_layout(
    arg: TensorArg,
    levels: Sequence[Level],
    *,
    symbolic_extent: str = "static",
    bounds: dict | None = None,
) -> tuple[Layout, list[tuple[int, ...]]]:
    """``(Layout, q)`` for ``arg``: extents and per-level steps, solved together.

    They are mutually dependent -- a step is a multiple of a stride, a stride is
    a product of extents, an extent grows by a step -- so they cannot be given
    separate owners.  The solve runs innermost dim outward, which terminates
    because the trailing dim is a stick dim and is never coarse-tiled: its
    extent is ``device_size``' own, giving the first stride, and each further
    dim's extent is settled before the next stride needs it.

    With no levels this is one pass with nothing to decompose: the extent is
    ``device_size`` and the strides are row-major of it.
    """
    tile = [_static(s) for s in arg.device_size]
    rank = len(tile)
    strides: list[Any] = [None] * rank
    extent = list(tile)
    q: list[tuple[int, ...]] = [tuple([0] * rank) for _ in levels]
    for i in range(rank - 1, -1, -1):
        strides[i] = 1 if i == rank - 1 else _mul(extent[i + 1], strides[i + 1])
        q = _advance(arg, levels, strides)
        extent[i] = _grown_extent(tile[i], levels, [row[i] for row in q])
    coeffs = _advance_coeffs(arg, levels)
    seen: dict[int, int] = {}
    for level_index, coeff in enumerate(coeffs):
        if not coeff:
            continue
        if coeff in seen:
            raise NotImplementedError(
                f"OpSpec->KTIR: levels {seen[coeff]} and {level_index} of "
                f"{arg.name!r} both advance by {coeff} elements, so which view "
                "dim each walks cannot be told apart from the linearized advance"
            )
        seen[coeff] = level_index
        if not any(q[level_index]):
            raise NotImplementedError(
                f"OpSpec->KTIR: tile advance {coeff} elements (level "
                f"{level_index} of {arg.name!r}) is not a whole number of steps "
                f"along any dim of a view with strides {tuple(strides)}"
            )
    return (
        _layout(arg, levels, q, symbolic_extent=symbolic_extent, bounds=bounds),
        q,
    )


def _access(
    arg: TensorArg,
    levels: Sequence[Level],
    q: Sequence[Sequence[int]],
    layout: Layout,
    buffer: Buffer | None = None,
) -> Access:
    """The access record for one ``(OpSpec, TensorArg)``.

    The tile extent is ``device_size`` (tiling is already baked into it), and
    the per-dim index expression is level ``l``'s step along dim ``i`` times
    that level's induction variable, summed over levels.

    Label: ``access-tile-offsets``.

    With no levels every row is empty, so every index expression is the empty
    sum -- zero -- which is why an untiled access sits at the view's origin.
    """
    extent = tuple(_static(s) for s in arg.device_size)
    for value in extent:
        if not isinstance(value, int):
            raise NotImplementedError(
                f"OpSpec->KTIR: access tile extent {value} of {arg.name!r} is "
                "symbolic; a tile is sized in whole elements"
            )
    if len(extent) != len(layout.extent):
        raise AssertionError(
            f"access rank {len(extent)} != buffer rank {len(layout.extent)}"
        )
    index_coeffs = tuple(
        tuple(int(q[l][i]) for l in range(len(levels))) for i in range(len(extent))
    )
    return Access(
        extent=extent,
        index_coeffs=index_coeffs,
        indices=tuple(level.iv for level in levels),
        elems=_elem_types(arg),
        buffer=buffer,
    )


def _solve_access(
    arg: TensorArg, levels: Sequence[Level], table: BufferTable | None = None
) -> Access:
    """``arg``'s access record at nest depth ``levels``: the derivations, in order.

    The one composite the walk calls per operand.  ``_solve_layout`` settles the
    view the tile indexes and the per-level steps together, and ``_access`` turns
    those steps into per-dim index expressions; ``table`` supplies the ``Buffer``
    the access is a tile of, which is how the record carries its own way back to
    the emitted view.  Named like ``_solve_layout`` because it is the same kind of
    thing: several derivations whose results depend on each other.
    """
    layout, q = _solve_layout(arg, levels)
    buffer = table.buffers.get(_buf_id(arg)) if table is not None else None
    return _access(arg, levels, q, layout, buffer)


# ---------------------------------------------------------------------------
# BufferTable: the one record the walk needs up front
# ---------------------------------------------------------------------------
#
# The func signature must be known before the body is emitted, and its parameter
# count depends on the address form, so the buffers -- the ``Buffer`` records the
# derivations produce -- are collected by the validation walk into a flat table
# keyed by ``_buf_id``.  The per-access ``Access`` records are not in the table:
# they depend on the enclosing loop levels, so they are derived where the walk
# knows them, from the same pure derivations the table was built with.


def is_internal(arg: TensorArg) -> bool:
    """Whether the kernel produces this buffer only for its own later ops.

    An internal buffer is threaded as an SSA value: not stored, and read back as
    a value rather than loaded.  SDSC says the same thing by giving the buffer an
    LX or HBM-pool allocation, which KTIR does not want -- the scheduler owns
    buffering -- so the KTIR form needs a spec-level flag instead.

    OpSpec does not carry one yet, so this reads a field that does not exist and
    nothing is internal.  When ``TensorArg`` grows the flag, the body becomes
    ``return arg.is_internal`` and no caller changes.
    """
    return bool(getattr(arg, "is_internal", False))


def _buffer(
    arg: TensorArg,
    layout: Layout,
    elems: ElemTypes,
    *,
    bake_addresses: bool = False,
) -> Buffer:
    """``arg``'s buffer record, rejecting what the emitter cannot address.

    The one place a ``TensorArg`` becomes a ``Buffer``, so every buffer-level
    rejection is here and the record the table holds is the record the view is
    emitted from.  ``layout`` and ``elems`` are the other derivations' answers,
    passed in rather than re-derived.
    """
    # ``arg_index`` stays -1 for buffers the frontend does not pass to the
    # kernel, which today means an LX or HBM-pool allocation.  This emitter
    # constructs HBM memory views only.
    if arg.arg_index < 0:
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} is not a kernel argument "
            f"(allocation={arg.allocation!r}); only HBM buffers are supported"
        )
    return Buffer(
        buf_id=_buf_id(arg),
        arg_index=arg.arg_index,
        elems=elems,
        layout=layout,
        # Resolved only for the baked form: the symbolic form takes its bases
        # from func arguments and never reads ``allocation["hbm"]``, whose units
        # differ between the two forms.
        base_elements=_base_address_elements(arg) if bake_addresses else None,
        space="HBM",
    )


class BufferTable:
    """Unique ``Buffer`` records in first-seen order, keyed by ``buf_id``.

    A container of records: it reads no ``TensorArg``, and ``_buffer`` is what
    turns one into an entry.  Carries the address form because that is what
    ``_buffer`` needs to resolve a base.
    """

    def __init__(self, *, bake_addresses: bool = False) -> None:
        self.buffers: dict[str, Buffer] = {}
        self.bake_addresses = bake_addresses

    def add(self, buffer: Buffer) -> None:
        """Register ``buffer``, keeping the first record seen for its ``buf_id``."""
        self.buffers.setdefault(buffer.buf_id, buffer)

    @property
    def param_entries(self) -> list[Buffer]:
        """External buffers in ascending ``arg_index``.

        Ascending ``arg_index`` matches the positional order ``call_kernel``
        passes to ``.run(...)``, so the emitted func signature lines up with
        that binding.
        """
        return sorted(
            (e for e in self.buffers.values() if e.arg_index >= 0),
            key=lambda e: e.arg_index,
        )


def _base_address_elements(arg: TensorArg) -> int:
    """``arg``'s buffer base address in ELEMENTS, for the baked form only.

    Read from ``allocation["hbm"]``, the same field the SDSC path resolves into
    the bundle start address (``superdsc.py:774`` -> ``startAddressCoreCorelet_``).
    Its units follow ``config.bundle_symbolic_args``: baked gives a byte address
    (arg 1 -> ``{'hbm': 17179869184}``), symbolic a bare sentinel ``arg_index``
    (arg 1 -> ``{'hbm': 1}``).  A memref offset indexes the *element* type, so
    the byte address is scaled down by the element size.
    """
    allocation = arg.allocation or {}
    # Key presence, not truthiness: a legitimate 'hbm' address of 0 exists.
    if "hbm" not in allocation:
        space = next(iter(allocation), None)
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} is not HBM-allocated "
            f"(allocation={allocation!r}); the emitter only emits HBM memory "
            f"views, so {space!r} allocations are out of scope"
        )
    byte_offset = allocation["hbm"]
    if byte_offset is None:
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} has an unassigned 'hbm' "
            "address (None); memory planning must run before KTIR emission"
        )
    return int(byte_offset) // num_bytes(arg.device_dtype)


# ---------------------------------------------------------------------------
# validate: every rejection, with no mlir_ktdp
# ---------------------------------------------------------------------------


def validate(
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
    *,
    bake_addresses: bool = False,
) -> BufferTable:
    """Pure. Raises every ``NotImplementedError`` the emitter can raise.

    Imports nothing from ``mlir_ktdp`` (the dialect import is lazy, inside
    ``KtirBuilder.create``), so it is testable wherever ``import ktir`` works --
    which is everywhere.  Afterwards the emission path holds no ``raise`` but
    the ``AssertionError`` in ``emit_specs``, which only fires on a validation
    bug.
    """
    # Multi-core work division is future work; the emitted grid is hard-coded to
    # a single core, so reject anything else rather than silently emitting a
    # single-core grid on a multi-core request.
    if _spyre_config.sencores != 1:
        raise NotImplementedError(
            "OpSpec->KTIR: multi-core work division is not supported yet "
            f"(SENCORES={_spyre_config.sencores}, only 1 is supported)"
        )
    table = BufferTable(bake_addresses=bake_addresses)
    _validate_list(specs, table)
    if not table.buffers:
        raise NotImplementedError("OpSpec->KTIR: no OpSpec to emit")
    return table


def _validate_list(specs, table: BufferTable, loops: Sequence = ()) -> None:
    """Recursive: validate one spec list. Mirrors ``emit_specs``' structure.

    ``loops`` is the enclosing ``(LoopSpec, induction variable)`` chain, the same
    value ``emit_specs`` reaches an op with, except that no induction variable
    exists outside an emission and every entry's is ``None``.  The derivations
    read only ``LoopSpec.count`` and the level ordering, so they answer the same
    here as they do at emit time.
    """
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            raise NotImplementedError(f"OpSpec->KTIR: unimplemented op {entry.op!r}")
        if isinstance(entry, LoopSpec):
            raise NotImplementedError(
                "OpSpec->KTIR: counted loops (LoopSpec) are not supported yet"
            )
            # When loops land, this becomes:
            #     _validate_list(entry.body, table, [*loops, (entry, None)])
        if not isinstance(entry, OpSpec):
            raise NotImplementedError(
                f"OpSpec->KTIR: unexpected spec entry {type(entry).__name__}"
            )
        if entry.is_reduction:
            raise NotImplementedError("OpSpec->KTIR: reductions are not supported yet")
        if entry.op not in KtirBuilder.RECIPES:
            raise NotImplementedError(
                f"OpSpec->KTIR: op {entry.op!r} is not supported yet "
                f"(registered: {sorted(KtirBuilder.RECIPES)})"
            )
        _validate_op(entry, table, loops)


def _validate_op(spec: OpSpec, table: BufferTable, loops: Sequence = ()) -> None:
    """Per-op checks: roles/arity, in-place aliasing, operand alignment, buffers.

    The buffer records are built here by the same derivations the emission path
    calls, so every raise those derivations own is reached before the dialect is
    imported and the table cannot describe a buffer differently from the view
    that gets emitted for it.
    """
    out, inputs = validated_roles(spec)
    out_extents = [int(s) for s in out.device_size]
    for arg in inputs:
        # In-place (input buffer aliases the output) is not supported yet.
        if _buf_id(arg) == _buf_id(out):
            raise NotImplementedError(
                "OpSpec->KTIR: in-place ops (input aliases output) not supported"
            )
        # Reject broadcast / transpose operands: only operands whose device
        # axes already match the output tile exactly are supported.
        plan = _align_reshape_plan(
            list(arg.device_coordinates),
            [int(s) for s in arg.device_size],
            list(out.device_coordinates),
            out_extents,
        )
        if plan is not None:
            raise NotImplementedError(
                "OpSpec->KTIR: broadcast / reshape operands not supported yet"
            )
    # The derivations run here, so every raise they own -- unsupported dtype, an
    # extent that stays symbolic, a tile advance no view dim divides, a buffer
    # with no address -- is reached before the dialect is imported.  Their results
    # go into the table, so the table describes each buffer exactly as the view
    # emitted for it will.
    levels = _levels(spec, loops)
    for arg in spec.args:
        layout, _ = _solve_layout(arg, levels)
        elems = _elem_types(arg)
        # An internal buffer never reaches memory: no func parameter, no memory
        # view, no address.  It is threaded as a value instead, so it gets no
        # table entry.
        if is_internal(arg):
            continue
        table.add(_buffer(arg, layout, elems, bake_addresses=table.bake_addresses))


def validated_roles(spec: OpSpec) -> tuple[TensorArg, list[TensorArg]]:
    """``(output, inputs)`` for ``spec``, or raise. Pure; shared with ``validate``.

    Handlers call this instead of re-deriving the roles, so the arity and
    single-output rejections have exactly one implementation.
    """
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(outputs) != 1:
        raise NotImplementedError(
            f"OpSpec->KTIR: expected exactly one output, got {len(outputs)}"
        )
    arity = KtirBuilder.RECIPES[spec.op].arity
    if len(inputs) != arity:
        raise NotImplementedError(
            f"OpSpec->KTIR: {spec.op!r} expects {arity} inputs, got {len(inputs)}"
        )
    return outputs[0], inputs


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
#
# A recipe declares one op: how many inputs it takes, which emission family it
# belongs to, and the dialect builder that implements it.  The recipes live on
# ``KtirBuilder`` beside the family methods that execute them.
#
# ``binding`` returns the builder rather than being it.  The call defers the
# dialect reference to emit time, keeping this module importable without a
# dialect build, and keeps the reference a literal that tooling can resolve.


class Family(enum.Enum):
    """An emission shape.  A recipe declares the one its binding implements;
    ``Family.of`` reads the one a spec asks for."""

    ELEMENTWISE = enum.auto()
    REDUCTION = enum.auto()

    @classmethod
    def of(cls, spec: OpSpec) -> Family:
        """The family ``spec`` asks for, read from the spec rather than the op
        name: the same binding can be wanted in more than one shape."""
        return cls.REDUCTION if spec.is_reduction else cls.ELEMENTWISE


@dataclasses.dataclass(frozen=True)
class Recipe:
    arity: int
    family: Family
    binding: Callable[[], Any]

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError(f"OpSpec->KTIR: arity must be >= 1, got {self.arity}")


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def emit_specs(b: KtirBuilder, specs) -> None:
    """Emit a spec list into the current insertion point. Recursive."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            emit_loop(b, entry)
        elif isinstance(entry, OpSpec):
            _emit_op(b, entry)
        else:
            # validate() already rejected UnimplementedOp and anything else, so
            # reaching here is a validation bug, not an unsupported request.
            # AssertionError (not TypeError) says exactly that.
            raise AssertionError(  # noqa: TRY004
                f"unvalidated spec entry {type(entry).__name__}"
            )


def _emit_op(b: KtirBuilder, spec: OpSpec) -> None:
    """Translate one ``OpSpec`` into records, then emit it.

    THE boundary between the spec tree and the builder.  Reading the iteration
    space needs the enclosing loop nest, which only the walk knows, so the walk
    derives every ``Access`` here -- against the nest the insertion point is
    inside, taken from the builder's own scope stack -- and hands the builder
    records, values and primitives.  Nothing below this line sees an ``OpSpec``.
    """
    recipe = KtirBuilder.RECIPES[spec.op]
    family = Family.of(spec)
    if family is not Family.ELEMENTWISE:
        # validate() rejects every family without a builder method, so reaching
        # here is a validation bug rather than an unsupported request.
        raise AssertionError(f"no emission for family {family} of {spec.op!r}")
    out, inputs = validated_roles(spec)
    levels = _levels(spec, b.env.loops())
    ins = [
        b.operand(_buf_id(arg), _solve_access(arg, levels, b.table)) for arg in inputs
    ]
    out_access = _solve_access(out, levels, b.table)
    value = b.elementwise(recipe, ins, out_access)
    # An internal buffer is threaded rather than stored, so it is handed no
    # access to store through.  The extent and element type the compute needed
    # still come from its own access record.
    b.result(_buf_id(out), None if is_internal(out) else out_access, value)


def emit_loop(b: KtirBuilder, loop: LoopSpec) -> None:
    """``scf.for`` over ``loop.count``, body emitted inside the loop's region.

    Not supported yet -- ``_validate_list`` rejects ``LoopSpec``, so this is
    unreachable.  The shape is here so that enabling loops is a local change
    rather than a restructure: the nesting in the emitted IR comes from
    ``ir.InsertionPoint`` plus ``b.env.scope(...)``, both scoped by ``with``, so
    the walk needs no parallel structure to know where it is.
    """
    lo, step = b.icst_index(0), b.icst_index(1)
    hi = b.trip_count(loop.count)  # int today; sympy later
    for_op = scf.ForOp(lo, hi, step)
    with (
        ir.InsertionPoint(for_op.body),
        b.env.scope(iv=for_op.induction_variable, loop=loop),
    ):
        emit_specs(b, loop.body)  # the recursion point
        # scf.for regions are not implicitly terminated by the builders.  No
        # operands: the loop carries no iter_args, because every value it
        # produces is stored to memory inside the body.
        scf.YieldOp([])


# ---------------------------------------------------------------------------
# What the walk carries in scope
# ---------------------------------------------------------------------------


class ScopeStack:
    """Builder-owned lexical scope: loop levels and live values.

    Pushed and popped by the walk via ``with``.  A base frame is always present
    so values produced at function level have somewhere to live.  A frame that
    carries a ``LoopSpec`` and its induction variable is a loop level; the base
    frame carries neither.
    """

    def __init__(self) -> None:
        # (LoopSpec or None, induction variable or None, {buf_id: Value}),
        # innermost last.
        self._frames: list[tuple[Any, Any, dict[str, Any]]] = [(None, None, {})]

    @contextlib.contextmanager
    def scope(self, iv: Any = None, loop: Any = None) -> Iterator[None]:
        self._frames.append((loop, iv, {}))
        try:
            yield
        finally:
            self._frames.pop()

    def produced(self, buf_id: str):
        """The ``Value`` a live node produced for ``buf_id``, else ``None``."""
        for *_, produced in reversed(self._frames):
            if buf_id in produced:
                return produced[buf_id]
        return None

    def bind_produced(self, buf_id: str, value) -> None:
        self._frames[-1][2][buf_id] = value

    def ivs(self) -> list:
        """Enclosing induction variables, innermost last."""
        return [iv for _, iv, _ in self._frames if iv is not None]

    def loops(self) -> list[tuple[Any, Any]]:
        """Enclosing ``(LoopSpec, induction variable)`` levels, outermost-first.

        The value ``_levels`` zips ``OpSpec.tiled_symbols`` against, so the loop
        nest the walk is physically inside is the one the derivations reason
        about -- there is no second record of the nesting to fall out of step.
        """
        return [(loop, iv) for loop, iv, _ in self._frames if loop is not None]


# ---------------------------------------------------------------------------
# KtirBuilder
# ---------------------------------------------------------------------------


class KtirBuilder:
    """Owns the MLIR context, the dialect handles and per-module state.

    No method takes an ``OpSpec`` or a ``TensorArg``: the arguments are records
    (``Buffer``, ``Layout``, ``ElemTypes``, ``Access``), SSA values and
    primitives.  Deriving an access needs the enclosing loop nest, so that
    reading of the iteration space is the walk's (``_emit_op``'s) job and the
    builder receives the answer.  The builder does hold the ``BufferTable``,
    which is a table of records, so the walk can reach it through ``b``.

    Every ktdp shape method returns an SSA ``Value``, so ``val()`` does not
    appear at call sites.
    """

    def __init__(self, stack, table: BufferTable):
        self._stack = stack
        self.table = table
        self.env = ScopeStack()
        # Requires the live context entered by create().
        self.index_t = ir.IndexType.get()
        self.block_args: list = []
        self.views: dict[str, Any] = {}
        self.c0 = None
        self._text: str | None = None

    @classmethod
    def create(cls, table: BufferTable) -> KtirBuilder:
        """THE single lazy-import site, and the owner of the MLIR context.

        Module level stays ``mlir_ktdp``-free, so ``import ktir`` -- and
        therefore ``validate`` -- works where the dialect build is absent.

        The context is entered here rather than in ``module()`` because
        ``_func_param_types`` builds ``ir`` types and is called before the module
        is opened; ``module()`` closes it on the way out.
        """
        _load_dialects()

        stack = contextlib.ExitStack()
        try:
            ctx = stack.enter_context(ir.Context())
            stack.enter_context(ir.Location.unknown())
            ktdp.register_dialects(ctx)
            return cls(stack, table)
        except BaseException:
            stack.close()
            raise

    # -- generic helpers ---------------------------------------------------

    @staticmethod
    def val(x):
        """The SSA ``Value`` of a builder result (builders return ``OpView`` or ``Value``)."""
        return x.result if hasattr(x, "result") else x

    @staticmethod
    def named_type(name: str):
        """The ``ir`` type for one ``ElemTypes`` entry (a type-builder name)."""
        return getattr(ir, name).get()

    def icst_index(self, value: int):
        """A fresh ``arith.constant <value> : index``."""
        return self.val(arith.ConstantOp(self.index_t, int(value)))

    def trip_count(self, count):
        """``count`` as an index SSA value. ``int`` today; sympy later."""
        return self.icst_index(int(count))

    def scaled(self, value, factor: int):
        """``factor * value`` as an index SSA value, without a ``* 1``."""
        if factor == 1:
            return value
        return self.val(arith.MulIOp(value, self.icst_index(factor)))

    def summed(self, terms: Sequence):
        """``sum(terms)`` as an index SSA value; the empty sum is ``%c0``.

        The function-entry ``%c0`` is reused rather than re-materialised, so an
        access that does not move along a dim indexes it with the one zero.
        """
        if not terms:
            return self.c0
        return functools.reduce(
            lambda lhs, rhs: self.val(arith.AddIOp(lhs, rhs)), terms
        )

    # -- module scaffolding ------------------------------------------------

    @contextlib.contextmanager
    def module(self, kernel_name: str, grid: list[int], params: list) -> Iterator[None]:
        """Open ``module { func.func @kernel_name(params) }`` and emit into it."""
        try:
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                # [] is the result list: a KTIR kernel returns nothing.
                fn = func.FuncOp(kernel_name, ir.FunctionType.get(params, []))
                i64 = ir.IntegerType.get_signless(64)
                fn.attributes["grid"] = ir.ArrayAttr.get(
                    [ir.IntegerAttr.get(i64, int(g)) for g in grid]
                )
                block = fn.add_entry_block()
                self.block_args = list(block.arguments)
                with ir.InsertionPoint(block):
                    self.c0 = self.icst_index(0)
                    yield
                    func.ReturnOp([])  # no operands, matching the signature
            # Printed while the context is still alive.
            self._text = str(module)
        finally:
            self._stack.close()

    def finish(self) -> str:
        """The canonical MLIR text of the module built by ``module()``."""
        if self._text is None:
            raise AssertionError("KtirBuilder.finish() before module() completed")
        return self._text

    # -- ktdp shapes -------------------------------------------------------

    def memory_view(self, base, buffer: Buffer):
        """``ktdp.construct_memory_view`` for one buffer, at base address ``base``.

        Extent and strides come from ``buffer.layout``, the record ``_layout``
        derived, so the view says what the table says.  Both are whole element
        counts: a symbolic extent reaches no view, because the extent mode that
        would let one through is guarded in ``_resolve_extent``.
        """
        sizes = [int(e) for e in buffer.layout.extent]
        strides = [int(s) for s in buffer.layout.strides]
        memref_t = ir.MemRefType.get(sizes, self.named_type(buffer.elems.storage))
        # No Python builder is exposed for the ``spyre_memory_space`` enum
        # attribute (only the ktdp *types* have getters), so this small enum
        # literal is the one unavoidable textual attribute.
        memory_space = ir.Attribute.parse(f"#ktdp.spyre_memory_space<{buffer.space}>")
        return self.val(
            ktdp.construct_memory_view(
                result=memref_t,
                offset=base,
                # SSA operands for dynamic extents, of which there are none:
                # every size and stride is a literal, passed as static_* below.
                sizes=[],
                strides=[],
                static_sizes=sizes,
                static_strides=strides,
                memory_space=memory_space,
                coordinate_set=self.coord_set(sizes),
            )
        )

    def bind_view(self, buf_id: str, view) -> None:
        self.views[buf_id] = view

    def view(self, buf_id: str):
        """The memory view bound for ``buf_id``."""
        return self.views[buf_id]

    def tile_indices(self, access: Access) -> list:
        """One index SSA value per view dim: ``sum_l coeffs[i][l] * indices[l]``."""
        return [
            self.summed(
                [
                    self.scaled(iv, coeff)
                    for coeff, iv in zip(coeffs, access.indices, strict=True)
                    if coeff
                ]
            )
            for coeffs in access.index_coeffs
        ]

    def access_tile(self, view, access: Access):
        """``ktdp.construct_access_tile`` for ``access`` into ``view``."""
        sizes = list(access.extent)
        rank = len(sizes)
        at_t = ktdp.AccessTileType.get(sizes, ir.IndexType.get())
        identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))
        return self.val(
            ktdp.construct_access_tile(
                result=at_t,
                base=view,
                # How the view is indexed, and the order of the tile's own axes.
                # Both identity: the tile covers the view one-to-one.
                base_map=identity,
                access_tile_order=identity,
                indices=self.tile_indices(access),
                # SSA operands for symbols in base_map; it uses none.
                symbol_operands=[],
                access_tile_set=self.coord_set(sizes),
            )
        )

    def operand(self, buf_id: str, access: Access):
        """The value of an input operand: a live produced value, or a fresh load.

        Reusing a produced value is what register-threaded fused intermediates
        will need; they are rejected today, so ``produced`` is always ``None``
        and this always loads.
        """
        produced = self.env.produced(buf_id)
        return produced if produced is not None else self.load(access)

    def load(self, access: Access):
        """An access tile + ``ktdp.load`` of the tile ``access`` describes."""
        tensor_t = ir.RankedTensorType.get(
            list(access.extent), self.named_type(access.elems.value)
        )
        tile = self.access_tile(self.view(access.buffer.buf_id), access)
        return self.val(ktdp.load(result=tensor_t, access_tile=tile))

    def result(self, buf_id: str, access: Access | None, value) -> None:
        """Dispose of an op's result: thread it, or materialise it.

        The mirror of ``operand`` on the way out.  ``access is None`` is an
        internal buffer: it has no view to store through, so the value is bound
        in scope for a later op to consume.
        """
        if access is None:
            self.env.bind_produced(buf_id, value)
        else:
            self.store_to(access, value)

    def store_to(self, access: Access, value) -> None:
        """An access tile + ``ktdp.store`` of ``value`` into the tile ``access``."""
        tile = self.access_tile(self.view(access.buffer.buf_id), access)
        ktdp.store(data_tile=value, access_tile=tile)

    # -- compute -----------------------------------------------------------
    #
    # One method per emission family, not per op: the op contributes only its
    # dialect builder, via ``recipe.binding()``.
    #
    # Bindings are dialect *functions*, not OpView classes: ``linalg.AddOp``
    # constructed directly leaves the named op's body region empty and fails
    # verification, while the OpDSL function generates that body.
    #
    # A repeated key here is ruff F601, so an op cannot be declared twice.
    RECIPES: ClassVar[dict[str, Recipe]] = {
        "add": Recipe(arity=2, family=Family.ELEMENTWISE, binding=lambda: linalg.add),
        "mul": Recipe(arity=2, family=Family.ELEMENTWISE, binding=lambda: linalg.mul),
    }

    def elementwise(self, recipe: Recipe, ins: Sequence, out: Access):
        """Apply ``recipe``'s builder to ``ins``, shaped by the result's tile.

        Returns the result value; what becomes of it is the caller's decision.
        Every operand and the result share the result tile's extents.

        The destination is an uninitialised ``tensor.empty``, valid because an
        elementwise op writes every element of it.  An accumulating op reads its
        destination and needs it filled with an identity instead.
        """
        extents = list(out.extent)
        elt_t = self.named_type(out.elems.value)
        dest = self.val(tensor.EmptyOp(extents, elt_t))
        return recipe.binding()(
            *ins,
            outs=[dest],
            result_tensors=[ir.RankedTensorType.get(extents, elt_t)],
        )

    # -- attributes --------------------------------------------------------

    @staticmethod
    def coord_set(sizes: list[int]):
        """Per-dim bounding integer set ``(0 <= d_i <= size_i - 1)`` as an attribute.

        Built with ``ir.IntegerSet`` from ``AffineExpr`` constraints (no textual
        round-trip): for each dim ``i`` two inequalities ``d_i >= 0`` and
        ``-d_i + (size_i - 1) >= 0``, matching the ``affine_set`` MLIR prints.
        """
        exprs = []
        eq_flags: list[bool] = []
        for i, s in enumerate(sizes):
            dim = ir.AffineExpr.get_dim(i)
            # d_i >= 0
            exprs.append(dim)
            eq_flags.append(False)
            # -d_i + (size_i - 1) >= 0
            neg_dim = ir.AffineExpr.get_mul(ir.AffineExpr.get_constant(-1), dim)
            upper = ir.AffineExpr.get_add(
                neg_dim, ir.AffineExpr.get_constant(int(s) - 1)
            )
            exprs.append(upper)
            eq_flags.append(False)
        integer_set = ir.IntegerSet.get(len(sizes), 0, exprs, eq_flags)
        return ir.IntegerSetAttr.get(integer_set)


# ---------------------------------------------------------------------------
# The two address forms
# ---------------------------------------------------------------------------
#
# ``BufferTable.bake_addresses`` selects where a buffer's base address comes
# from: a func argument (symbolic) or an ``arith.constant`` in elements (baked).
# The SDSC path makes the same choice from ``config.bundle_symbolic_args``.
#
# The baked form exists because ``ktdp.load`` requires a static memref offset,
# which a constant base gives only when the consumer is a ``linalg`` op.  It is
# the dataflow-scheduler#65 workaround: deleting the baked arm of the two
# functions below reverts it.


def _func_param_types(b: KtirBuilder, table: BufferTable) -> list:
    """Baked bases need no func arguments; symbolic ones need an index each."""
    if table.bake_addresses:
        return []
    return [b.index_t] * len(table.param_entries)


def _bind_views(b: KtirBuilder, table: BufferTable) -> None:
    """One memory view per buffer, based on a constant or on a func argument."""
    baked = table.bake_addresses
    for position, buffer in enumerate(table.param_entries):
        base = b.icst_index(buffer.base_elements) if baked else b.block_args[position]
        b.bind_view(buffer.buf_id, b.memory_view(base, buffer))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_ktir(
    kernel_name: str,
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
    **options,
) -> str:
    """Build a KTDP-dialect MLIR module for ``specs`` and return ``str(module)``.

    ``specs`` is the finished OpSpec kernel contract (the same value
    ``call_kernel`` passes positionally to ``.run(...)``).  Func parameters are
    the unique operand buffers in ascending ``arg_index`` order so the emitted
    signature matches that positional binding (or, in the baked form, no
    parameters at all and one ``arith.constant`` base address per buffer).
    """
    # Every rejection lives in validate(), and validate() completes before
    # KtirBuilder.create(), so an unsupported request fails fast -- and is
    # testable -- whether or not mlir_ktdp is installed.
    # Emission options, defaulted here so callers pass only what they need.
    #
    # bake_addresses: emit each base as an arith.constant instead of a func
    # argument.  Canonical KTIR is symbolic; baking is what dbo-opt requires
    # (dataflow-scheduler#65), so the caller that runs dbo-opt asks for it.
    bake_addresses = bool(options.pop("bake_addresses", False))
    if options:
        raise TypeError(f"generate_ktir: unknown option(s) {sorted(options)}")

    table = validate(specs, bake_addresses=bake_addresses)
    b = KtirBuilder.create(table)
    # Single-core (SENCORES=1) grid; work-division scaling is future work.
    with b.module(kernel_name, grid=[1], params=_func_param_types(b, table)):
        _bind_views(b, table)
        emit_specs(b, specs)
    return b.finish()
