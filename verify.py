"""Verify the OpSpec->KTIR path end to end on device, against PyTorch.

Runs each case below through OpSpec -> KTIR -> dbo-opt -> spyreCodeDir -> the
jobplan runner, and compares against CPU eager.  Exit status is 0 only if every
case that compiled also matched numerically.

    python verify.py            # every case
    python verify.py add sum    # named cases only

--------------------------------------------------------------------------------
1. Install the KTIR dialect bindings (mlir_ktdp)
--------------------------------------------------------------------------------
The emitter imports `mlir_ktdp.dialects`, supplied by ktir-mlir-frontend. It is
declared as the optional `ktir` extra, but `pip install .[ktir]` is NOT
self-contained: it is a scikit-build-core/CMake project that must configure
against a matching LLVM/MLIR. Use the project's own helper:

    git clone https://github.com/torch-spyre/ktir-mlir-frontend
    cd ktir-mlir-frontend

    # Downloads the LLVM artifact pinned in cmake/llvm-hash.txt, cached under
    # ~/.cache/ktir-mlir/.  Needs GIT_PAT or GITHUB_TOKEN on the FIRST run only.
    # Add --wheel to install mlir_wheel instead of downloading an artifact, or
    # skip this entirely and point MLIR_DIR at an existing LLVM build.
    MLIR_DIR=$(python scripts/setup_mlir.py)

    CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" pip install .

Check it took, including the dialects the emitter needs:

    python -c "from mlir_ktdp.dialects import arith, func, ktdp, linalg, tensor"

--------------------------------------------------------------------------------
2. Point at the backend compiler
--------------------------------------------------------------------------------
    export PATH="$PATH:/path/to/backend/bin"     # the dir holding dbo-opt
    export KTIR_DEVICE_MLIR=/path/to/device.mlir

    # Only if `dbo-opt` cannot already resolve its own libraries. A build with an
    # RPATH covering its tree needs nothing here; one without it needs its lib
    # dir named, and it must be the SAME tree as whichever dbo-opt PATH resolves
    # (check with `which dbo-opt`). dbo-opt is spawned with this process's
    # environment, so anything exported here applies to both.
    export LD_LIBRARY_PATH=/path/to/backend/lib:$LD_LIBRARY_PATH

    python verify.py

`sum` needs a dbo-opt whose embedded scheduler carries the reduction work: an
older one rejects `linalg.reduce` at pass 00 with "V1 only supports add/mul/sub
compute ops". Nothing here can detect that from the outside, so it reports the
case as a backend refusal and keeps going.

Three things that will waste your afternoon if you get them wrong:

  * If you do set LD_LIBRARY_PATH, it must belong to the SAME tree as whichever
    dbo-opt PATH resolves. With more than one backend build installed, a
    mismatched pairing dies on an undefined symbol from a shared library the two
    trees both ship -- which reads like a broken build but is only a pairing
    mistake.

  * Do not reuse one shell across both backends. `TORCH_SPYRE_KTIR` is read once
    at import, so a process compiles through dbo-opt (KTIR) or the SDSC compiler
    for its whole life, and the library path you exported for one is still there
    when you flip the flag and run the other. That surfaces as the other backend
    loading the wrong library, not as anything mentioning the flag. Start a
    fresh shell when you switch.

  * KTIR_DEVICE_MLIR must declare enough HBM to cover the baked addresses (one
    16 GiB segment slot per argument). dbo-opt has no default device and rejects
    a module carrying no ktdf_arch.device op.
"""

import os
import sys

# Fail on a missing name here rather than as an empty --device= in a subprocess.
if not os.environ.get("KTIR_DEVICE_MLIR"):
    sys.exit("unset: KTIR_DEVICE_MLIR -- see this file's header")

# Both are read when torch_spyre.config is imported, so they must be set before
# importing it.  SENCORES is deliberately NOT set: the core count is whatever
# this build is configured for, and the emitter reads the work division the
# frontend derived from it rather than a count of its own.
os.environ["TORCH_SPYRE_KTIR"] = "1"  # select the KTIR emitter over SDSC
os.environ["BUNDLE_SYMBOLIC_ARGS"] = "0"  # bake addresses; dbo-opt requires constants

import torch

import torch_spyre
from torch_spyre._inductor import config as spyre_config

torch_spyre._autoload()

# One entry per capability this path claims, so a claim that stops holding shows
# up here rather than in a paragraph nobody re-checked.  A case is
# (name, what it exercises, fn, input shapes, why it is expected to be refused).
# Shapes are f16 with a stick-aligned last dim (a multiple of 64), and are chosen
# so that the frontend's work division lands on an axis the kernel can split:
# ``sum``'s output is 32 sticks wide, so the division goes to the stick axis
# rather than to the axis being reduced (which would need a cross-core combine).
CASES = [
    (
        "add",
        "one pointwise op, one core",
        lambda x, y: x + y,
        [(64,), (64,)],
        None,
    ),
    (
        "add-divided",
        f"work division across {spyre_config.sencores} cores (grid > 1)",
        lambda x, y: x + y,
        [(256, 64), (256, 64)],
        None,
    ),
    (
        "sum",
        "a reduction: linalg.reduce over the axis that does not survive",
        lambda x: torch.sum(x, dim=0),
        [(256, 2048)],
        None,
    ),
    (
        "chain",
        "two ops in one kernel, the intermediate threaded as a value",
        lambda x, y, z: (x + y) * z,
        [(64,), (64,), (64,)],
        # Not an emitter gap: the emitter threads an intermediate and a golden
        # test pins the shape.  The frontend puts the two ops in *two* kernels
        # (`ktir_fused_add_0` then `ktir_fused_mul_1`) with buf0 allocated in LX
        # between them, so no kernel here contains both ends of it.  Reaching
        # this needs the ops fused into one kernel (torch-spyre#3579); until then
        # the refusal is the honest answer, and this case is what tells us when
        # that changes.
        "the frontend emits two kernels, so the intermediate crosses a boundary",
    ),
]


def run(fn, shapes) -> tuple[str, str]:
    """``(verdict, detail)`` for one case.  Never raises.

    Compared with a relative tolerance: a reduction accumulates in fp16 in a
    different order from eager, so its error scales with the sum rather than with
    one ulp.
    """
    torch.manual_seed(0)
    args = [torch.rand(shape, dtype=torch.float16) for shape in shapes]
    expected = fn(*args).float()
    try:
        actual = torch.compile(fn)(*[a.to("spyre") for a in args]).cpu().float()
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, not a crash
        # Either the emitter refused to emit this, or dbo-opt refused what it
        # emitted; the message says which, and one case failing must not stop the
        # others from reporting.
        return "REFUSED", f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    finally:
        torch._dynamo.reset()  # each case is its own compile
    delta = (actual - expected).abs().max().item()
    ok = torch.allclose(actual, expected, atol=1e-2, rtol=2e-2)
    scale = expected.abs().max().item()
    return ("PASS" if ok else "FAIL"), f"max abs diff {delta} on values up to {scale}"


selected = sys.argv[1:]
unexpected = []
for name, what, fn, shapes, refused_because in CASES:
    if selected and name not in selected:
        continue
    verdict, detail = run(fn, shapes)
    expected_verdict = "REFUSED" if refused_because else "PASS"
    mark = "ok" if verdict == expected_verdict else "UNEXPECTED"
    if mark == "UNEXPECTED":
        unexpected.append(f"{name}: expected {expected_verdict}, got {verdict}")
    print(f"{verdict:8} {mark:11} {name:14} {what}\n{'':21} {detail}")
    if refused_because:
        print(f"{'':21} expected, because {refused_because}")

if unexpected:
    print("\n" + "\n".join(unexpected))
    # An unexpected PASS is as much a result as an unexpected refusal: a case
    # that starts working means a claim in this file is out of date.
    sys.exit(1)
print("\nevery case as expected")
