# Copyright 2025 The Torch-Spyre Authors.
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

import os
import sys
from typing import Literal

from torch.utils._config_module import install_config_module
from .logging_utils import _get_env_bool

lx_planning: bool = os.environ.get("LX_PLANNING", "1") == "1"
co_optimizing_lx_planning: bool = (
    os.environ.get("CO_OPTIMIZING_LX_PLANNING", "0") == "1"
)
hbm_pool_planning: bool = _get_env_bool("HBM_POOL_PLANNING", True)

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

# Opt-in OpSpec->KTIR emitter (experimental, #3380). When enabled the scheduler
# emits ``async_compile.ktir(...)`` instead of the SDSC bundle, and
# ``create_tensor_arg`` populates the op-spec buffer name so the emitter has a
# stable per-buffer identity. Inert by default: the SDSC/flex path is unchanged.
ktir_emitter: bool = os.environ.get("TORCH_SPYRE_KTIR", "0") == "1"

# Device description passed to the KTIR backend compiler as ``--device=<file>``.
# It has no default of its own and rejects a module that carries no
# ``ktdf_arch.device`` op, which the emitter does not produce, so this must name
# a .mlir declaring the target device. Only read on the KTIR/dbo path.
ktir_device_mlir: str = os.environ.get("KTIR_DEVICE_MLIR", "")

# Colon-separated library directories prepended to LD_LIBRARY_PATH *for the
# backend-compiler subprocess only*. The backend binary ships without an RPATH,
# so it cannot resolve its own deeptools libraries unaided. This must never be
# applied to this process: the runtime needs the installed libraries resolved
# first, and shadowing them makes a byte-correct program return garbage.
dbo_lib_paths: str = os.environ.get("DBO_LIB_PATHS", "")

allow_all_ops_in_lx_planning: bool = False

dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

sencores: int = int(os.getenv("SENCORES", "32"))

# Symbolic-dim knobs consumed by compute_granularity in pass_utils.py.
# The pointwise work-division PR (#2499) wires that helper into the
# compilation pipeline; until then these knobs are read only by the
# helper and its unit tests. See #2284, #2287 for the design.

# Cap on bucket count (= max_size / granularity).
# TODO: confirm the default with the Deeptools team.
max_buckets: int = int(os.getenv("MAX_BUCKETS", "32"))

# Soft floor on the auto-derived granularity when mark_dynamic(min=...)
# is not provided. Keeps the picked granularity from collapsing to a
# very small divisor when max_size has many of them.
min_default_granularity: int = int(os.getenv("MIN_DEFAULT_GRANULARITY", "4"))

ignore_work_division_hints: bool = (
    os.environ.get("SPYRE_INDUCTOR_IGNORE_HINTS", "0") == "1"
)

ignore_wsr_hints: bool = os.environ.get("SPYRE_INDUCTOR_IGNORE_HINTS", "0") == "1"

# Per-pass operation logging for CustomPreSchedulingPasses.
# Set to "all" or "1" to log after every pass, or a comma-separated list of
# pass function names (e.g., "split_multi_ops,insert_restickify") to log only
# after specific passes. Set via SPYRE_LOG_PASSES env var or programmatically.
log_passes: str = os.environ.get("SPYRE_LOG_PASSES", "")

# Disable compiler-generated span-overflow coarse-tiling hints.  The global
# SPYRE_INDUCTOR_IGNORE_HINTS flag also disables these so one switch can still
# suppress all WSR/coarse-tiling hint paths.
#
# Defaults to disabled (opt-in): span-overflow auto-tiling can synchronize
# compatible contiguous pointwise groups, but incompatible producer/consumer
# groups and reduction-dim tiling still need broader support. Set
# SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS=0 to opt in;
# tests exercising this path directly should override via
# config.patch({"ignore_span_overflow_hints": False}).
ignore_span_overflow_hints: bool = (
    ignore_wsr_hints
    or os.environ.get("SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS", "1") == "1"
)

# Enable reduction-dim (Lk-style) coarse tiling. Defaults to enabled — this
# capability is exercised by passing tests today. Disabling it (or a future
# hardware limitation that can't support it) makes planning treat any op
# whose group requests reduction-dim tiling as unsupported, raising
# Unsupported rather than attempting to tile it.
enable_reduction_tiling: bool = (
    os.environ.get("SPYRE_INDUCTOR_ENABLE_REDUCTION_TILING", "1") == "1"
)

# For K-split matmuls, permute physical core IDs so the cores collaborating on a
# K reduction land on adjacent ring positions, cutting PSUM chain hops from m*n
# to 1. The split itself is chosen by the cost-model planner; this only reorders
# cores at SDSC emission. Set SPYRE_CORE_ID_K_FAST_EMISSION=0 to disable.
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)

# When True (default), HBM tensor addresses are emitted as runtime symbols
# with !sdscbundle.input_arg<index> parameters and input_arg_extract ops
# in the bundle.mlir.
# When False, HBM tensor addresses are baked as concrete integers
# into the SDSC JSON and bundle.mlir emits sdsc_execute with no operands.
bundle_symbolic_args: bool = os.environ.get("BUNDLE_SYMBOLIC_ARGS", "1") == "1"

# The KTIR path requires literal addresses, so it forces the symbolic path off.
#
# Why here, and why both the flag and the env var:
#
#   * The emitter bakes each buffer's HBM base address into the module as a
#     constant (address assignment in the backend requires constant addresses,
#     dataflow-scheduler#65). It reads that address from ``allocation["hbm"]``,
#     whose *units* depend on this flag: with symbolic args that field holds a
#     sentinel ``arg_index`` (0, 1, 2) rather than an address, so the emitter
#     cannot function on the symbolic path at all.
#   * The C++ side does not consult this module. ``prepare_kernel.cpp`` reads the
#     raw environment variable, with an inverted sense:
#     ``bind_io_addresses_ = (env == nullptr || std::string(env) != "1")`` --
#     only the literal string "1" disables address binding, and *unset* is not
#     equivalent to "0". So the env var must be exported explicitly; flipping
#     only the Python flag leaves the runtime binding addresses over the ones
#     baked into the module.
#
# SIDE EFFECT, deliberate: this mutates the environment of the whole process, so
# it also affects anything else in it that reads BUNDLE_SYMBOLIC_ARGS. That is
# acceptable only because TORCH_SPYRE_KTIR=1 is an experimental, explicit opt-in
# that replaces the SDSC emission path outright; nothing sets it incidentally.
# The emitter keeps its own guard on ``bundle_symbolic_args`` as the safety net
# for any route that reaches it without passing through here.
#
# An assignment, not ``setdefault``: ``torch_spyre/__init__.py`` seeds the var to
# "1" for the SDSC default, and that must lose to an explicit KTIR opt-in
# regardless of which of the two runs first.
if ktir_emitter:
    bundle_symbolic_args = False
    os.environ["BUNDLE_SYMBOLIC_ARGS"] = "0"

# Layout solver class used by default in scratchpad.allocator.ScratchpadAllocator.
# Options:
#  "greedy":       GreedyLayoutSolver (default),
#  "bestfit":      BestFitLayoutSolver,
#  "firstfit":     FirstFitLayoutSolver,
#  "simulated_annealing":  SimulatedAnnealingLayoutSolver,
#  "cpsat":    CpSatLayoutSolver (OR-Tools CP-SAT joint core-division +
#              LX placement, minimizing HBM transfer traffic).

# TODO(isuruf): Change to firstfit when deeptools PR4298 lands
layout_solver: Literal[
    "greedy", "bestfit", "firstfit", "cpsat", "simulated_annealing"
] = os.environ.get("LAYOUT_SOLVER", "greedy")  # type: ignore[assignment]

install_config_module(sys.modules[__name__])
