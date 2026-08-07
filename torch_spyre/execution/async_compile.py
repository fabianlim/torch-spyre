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

import tempfile
from collections.abc import Sequence
from typing import Any
import os
import subprocess
import torch

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
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

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
            subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir], check=True)

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def wait(self, scope: dict[str, Any]) -> None:
        pass

    def ktir(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        """Emit KTDP-dialect MLIR for ``specs`` and compile it (OpSpec->KTIR path).

        Mirrors ``sdsc`` but emits KTIR instead of an SDSC bundle and compiles
        it with ``dbo-opt`` instead of ``dxp_standalone``.  ``dbo-opt`` writes
        ``<output_dir>/spyreCodeDir/{init_binary.bin, spyrecode.json}``, which
        is exactly what ``SpyreSDSCKernelRunner``'s jobplan path consumes.
        """
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        from torch_spyre._inductor.codegen.ktir import generate_ktir

        # Persist the emitted KTIR as a text file in the same per-kernel output
        # dir as sdsc's bundle.
        output_dir = get_output_dir(kernel_name)
        ktir_path = os.path.join(output_dir, f"{kernel_name}.ktir")
        with open(ktir_path, "w") as fh:
            fh.write(generate_ktir(kernel_name, specs))
        logger.debug("OpSpec->KTIR: wrote %s", ktir_path)

        # Invoke the KTIR backend compiler.  --device is required: without it
        # dbo-opt rejects a module that carries no ktdf_arch.device.
        cmd = [
            _spyre_config.dbo_opt,
            "--from-ktir",
            f"--device={_spyre_config.ktir_device_mlir}",
            "--kEmitSpyreCode",
            "--mlir-disable-threading",
            f"--export-dir={output_dir}",
            ktir_path,
        ]
        # dbo-opt's vendored MLIR/LLVM shared libs, prepended so they win over
        # anything already on the inherited path.
        env = dict(os.environ)
        lib_paths = _spyre_config.dbo_lib_paths
        inherited = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_paths}:{inherited}" if inherited else lib_paths

        with torch.profiler.record_function(f"dbo_opt:{kernel_name}"):
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

        # dbo-opt can exit 0 having honoured neither --export-dir nor
        # --kEmitSpyreCode, so check for the artifact explicitly.
        spyre_code_dir = os.path.join(output_dir, "spyreCodeDir")
        if not os.path.exists(os.path.join(spyre_code_dir, "spyrecode.json")):
            raise RuntimeError(
                f"OpSpec->KTIR: dbo-opt wrote no SpyreCode to {spyre_code_dir} "
                f"(no spyrecode.json) for {ktir_path}.\n"
                f"command: {' '.join(cmd)}\n"
                f"returncode: {proc.returncode}\n"
                f"stderr:\n{proc.stderr}"
            )

        # The KTIR path always produces a spyreCodeDir and never an init.txt,
        # so it always launches via the jobplan.
        return SpyreSDSCKernelRunner(kernel_name, output_dir, use_jobplan=True)
