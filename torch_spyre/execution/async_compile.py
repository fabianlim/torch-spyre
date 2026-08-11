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

import tempfile
from collections.abc import Sequence
import os
import shutil
import subprocess
import torch
import uuid

from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre.profiler._ffdc import CATEGORY_COMPILE, try_collect
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")

DBO_OPT = "dbo-opt"


def _check_ktir_device_prerequisites() -> None:
    """Raise unless the environment can compile emitted KTIR for the device.

    Reports *every* unmet prerequisite at once: they are typically all missing
    together on a first run, and surfacing them one per attempt turns a single
    misconfiguration into a sequence of unrelated-looking failures.

    ``RuntimeError``, not ``NotImplementedError``: nothing here is an
    unimplemented capability -- each item is a setting the caller can fix.
    """
    problems = []

    if _spyre_config.bundle_symbolic_args:
        problems.append(
            "BUNDLE_SYMBOLIC_ARGS=0 is required (config.bundle_symbolic_args is "
            "True). The device path needs literal baked addresses; the symbolic "
            "form -- runtime address arguments -- is emit-only, and the backend "
            "compiler cannot compile it. Set the environment variable, not just "
            "the Python config: prepare_kernel.cpp reads the raw env var "
            'independently, and treats unset as != "0".'
        )

    if not _spyre_config.ktir_device_mlir:
        problems.append(
            "KTIR_DEVICE_MLIR (config.ktir_device_mlir) must name a .mlir "
            "declaring the target device; the emitted KTIR carries no "
            "ktdf_arch.device op, so dbo-opt rejects the module without it."
        )

    if shutil.which(DBO_OPT) is None:
        problems.append(
            f"{DBO_OPT} was not found on PATH; append the deeptools bin dir to "
            "PATH (append, so the installed dxp_standalone is not shadowed)."
        )

    if not problems:
        return

    raise RuntimeError(
        "OpSpec->KTIR: the KTIR path cannot compile for the device in this "
        f"environment ({len(problems)} unmet prerequisite(s)):\n"
        + "\n".join(f"  {n}. {p}" for n, p in enumerate(problems, 1))
        + "\n\nDBO_LIB_PATHS (config.dbo_lib_paths) is commonly required too -- "
        "dbo-opt ships without an RPATH, so its deeptools libraries have to be "
        "named explicitly -- but it is not enforced here: it is unnecessary "
        "where those libraries are already on the default search path."
    )


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    digest = uuid.uuid4().hex[:8]
    kernel_output_dir = tempfile.mkdtemp(
        dir=spyre_dir, prefix=f"{digest}_{kernel_name}_"
    )
    return kernel_output_dir


class SpyreAsyncCompile(AsyncCompile):
    """Spyre kernel compilation (`sdsc`), plus the upstream AsyncCompile.

    A graph mixing Spyre and CPU work emits `async_compile.cpp_pybinding(...)`
    against this same object, so we inherit AsyncCompile for `cpp_pybinding`/
    `wait` rather than stubbing them -- a no-op `wait()` alone can't compile a
    CPU kernel it was never given.

    """

    def triton(self, *args, **kwargs):
        raise NotImplementedError(
            "SpyreAsyncCompile does not support Triton kernels; only "
            "cpp_pybinding (CPU) and sdsc (Spyre) are validated."
        )

    def cpp(self, *args, **kwargs):
        raise NotImplementedError(
            "SpyreAsyncCompile does not support the cpp() path; CPU kernels "
            "go through cpp_pybinding (cpu_backend='cpp')."
        )

    def sdsc(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        # Generate SDSC Bundle from OpSpecs
        output_dir = get_output_dir(kernel_name)
        generate_bundle(kernel_name, output_dir, specs)

        # Invoke backend compiler of SDSC Bundle
        with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
            try:
                subprocess.run(["dxp_standalone", "-d", output_dir], check=True)
            except Exception as exc:
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def ktir(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        """Emit KTDP-dialect MLIR for ``specs`` (OpSpec->KTIR path).

        Mirrors ``sdsc`` but emits KTIR directly instead of an SDSC bundle: the
        emitted KTIR is persisted to disk for inspection and then compiled by
        ``dbo-opt``, which writes a ``spyreCodeDir`` in the same layout
        ``dxp_standalone`` produces, so the result is loaded and launched by the
        same ``SpyreSDSCKernelRunner``.

        Required for device execution (all enforced upfront, before anything is
        emitted -- see ``_check_ktir_device_prerequisites``):

        * ``BUNDLE_SYMBOLIC_ARGS=0`` -- baked literal addresses; the symbolic
          form is emit-only and the backend compiler cannot compile it.
        * ``KTIR_DEVICE_MLIR=<file>`` -- a .mlir declaring the target device.
        * ``dbo-opt`` on ``PATH`` -- the backend compiler.
        * ``DBO_LIB_PATHS=<dirs>`` -- usually needed (dbo-opt has no RPATH), but
          not required where its libraries are already findable.
        """
        # Upfront: an unmet prerequisite is a misconfiguration, not something the
        # emitted KTIR can tell us, so there is no reason to emit first.
        _check_ktir_device_prerequisites()

        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        from torch_spyre._inductor.codegen.ktir import generate_ktir

        # Emit before opening the file: if generate_ktir raises we must not
        # leave a truncated/empty .ktir behind.
        ktir_text = generate_ktir(kernel_name, specs)

        # Persist the emitted KTIR as a text file in the same per-kernel output
        # dir as sdsc's bundle.
        output_dir = get_output_dir(kernel_name)
        ktir_path = os.path.join(output_dir, f"{kernel_name}.ktir")
        with open(ktir_path, "w") as fh:
            fh.write(ktir_text)
        logger.debug("OpSpec->KTIR: wrote %s", ktir_path)

        return self._compile_ktir_with_dbo(kernel_name, ktir_path, output_dir)

    def _compile_ktir_with_dbo(self, kernel_name: str, ktir_path: str, output_dir: str):
        """Compile ``ktir_path`` with ``dbo-opt`` and return a runner for it.

        ``--export-dir`` receives the per-kernel output dir, under which dbo-opt
        writes ``spyreCodeDir/{spyrecode.json, init_binary.bin}`` -- exactly the
        layout ``prepare_kernel`` loads, so no new runner is needed.
        """
        # Re-checked here, not only in ``ktir``: this is also reached directly
        # (tests, callers compiling a .ktir off disk), and the check is a cheap
        # idempotent read of config plus one PATH lookup.
        _check_ktir_device_prerequisites()

        cmd = [
            DBO_OPT,
            "--from-ktir",
            f"--device={_spyre_config.ktir_device_mlir}",
            f"--export-dir={output_dir}",
            "--kEmitSpyreCode",
            ktir_path,
        ]

        # dbo-opt ships without an RPATH, so its deeptools libraries have to be
        # named explicitly -- but only in the CHILD environment: shadowing the
        # runtime's own libraries here would silently corrupt results.
        env = dict(os.environ)
        if _spyre_config.dbo_lib_paths:
            existing = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = (
                f"{_spyre_config.dbo_lib_paths}:{existing}"
                if existing
                else _spyre_config.dbo_lib_paths
            )

        # 0 disables the ceiling; subprocess.run treats timeout=None as "wait".
        timeout = _spyre_config.dbo_timeout or None

        with torch.profiler.record_function(f"dbo-opt:{kernel_name}"):
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )
                # dbo-opt can exit 0 having written nothing, so the artifact
                # itself -- not the return code -- is the success condition.
                spyrecode = os.path.join(output_dir, "spyreCodeDir", "spyrecode.json")
                if not os.path.exists(spyrecode):
                    raise RuntimeError(
                        "OpSpec->KTIR: dbo-opt exited 0 but wrote no "
                        f"{spyrecode}.\ncommand: {' '.join(cmd)}\n"
                        f"stderr:\n{proc.stderr}"
                    )
            except subprocess.TimeoutExpired as exc:
                # Would otherwise land in the broad handler below, which collects
                # correctly but re-raises a TimeoutExpired whose message says
                # nothing about which knob relaxes it.
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise RuntimeError(
                    f"OpSpec->KTIR: dbo-opt timed out after {timeout}s "
                    "(config.dbo_timeout / DBO_TIMEOUT; 0 disables the "
                    f"ceiling).\ncommand: {' '.join(cmd)}"
                ) from exc
            except subprocess.CalledProcessError as exc:
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise RuntimeError(
                    f"OpSpec->KTIR: dbo-opt failed with exit code "
                    f"{exc.returncode}.\ncommand: {' '.join(cmd)}\n"
                    f"stderr:\n{exc.stderr}"
                ) from exc
            except Exception as exc:
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise

        return SpyreSDSCKernelRunner(kernel_name, output_dir)
