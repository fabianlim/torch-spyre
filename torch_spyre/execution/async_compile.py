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
        """Emit KTDP-dialect MLIR for ``specs`` (OpSpec->KTIR path).

        Mirrors ``sdsc`` but emits KTIR directly instead of an SDSC bundle.
        Device execution is not wired yet: the emitted KTIR is persisted to
        disk for inspection and this raises ``NotImplementedError``.
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

        raise NotImplementedError(
            "OpSpec->KTIR: device execution is not wired yet; the emitted KTIR "
            f"was written to {ktir_path} for inspection. Execution lands in a "
            "later PR."
        )
