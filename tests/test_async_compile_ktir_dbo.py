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

"""CI-safe wiring test for the ``dbo`` backend on the OpSpec->KTIR path.

Exercises ``SpyreAsyncCompile.ktir`` without a real ``dbo-opt`` binary or a
device: ``generate_ktir`` is stubbed to a fixed string, ``subprocess.run`` is
recorded (and fabricates the ``spyreCodeDir`` dbo would produce), and
``prepare_kernel`` is stubbed so the shared ``SpyreSDSCKernelRunner`` constructs
without hardware.  Asserts the ``TORCH_SPYRE_DBO`` gate, the exact ``dbo-opt``
command line, and that the compiled path returns the SDSC runner.
"""

import os
import unittest
from unittest import mock

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre.execution import async_compile as ac
from torch_spyre.execution import kernel_runner
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

_D0, _D1, _D2 = sympy.symbols("d0 d1 d2")
_SIZE = [16, 512, 64]


def _arg(is_input: bool, index: int, name: str) -> TensorArg:
    return TensorArg(
        is_input=is_input,
        arg_index=index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=list(_SIZE),
        device_coordinates=[_D0, _D1, _D2],
        allocation={"hbm": None},
        name=name,
    )


def _add_specs() -> list:
    """OpSpec list for ``a + b`` (two HBM inputs, one HBM output)."""
    return [
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                _D0: (_SIZE[0], 1),
                _D1: (_SIZE[1], 1),
                _D2: (_SIZE[2], 1),
            },
            args=[_arg(True, 0, "arg0"), _arg(True, 1, "arg1"), _arg(False, 2, "buf0")],
            op_info={},
        )
    ]


class KtirDboWiringTest(unittest.TestCase):
    def setUp(self):
        # Isolate the two switches this path reads.
        self._saved = {
            k: os.environ.get(k) for k in ("TORCH_SPYRE_KTIR", "TORCH_SPYRE_DBO")
        }
        os.environ.pop("TORCH_SPYRE_DBO", None)
        # generate_ktir is imported lazily inside ktir(); stub it so the test
        # needs neither mlir_ktdp nor a live graph.
        self._gen = mock.patch(
            "torch_spyre._inductor.codegen.ktir.generate_ktir",
            return_value="module {}\n",
        )
        self._gen.start()

    def tearDown(self):
        self._gen.stop()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_dbo_unset_emits_then_raises(self):
        """Without TORCH_SPYRE_DBO the KTIR is written but execution is deferred."""
        os.environ.pop("TORCH_SPYRE_DBO", None)
        with mock.patch.object(ac.subprocess, "run") as run:
            with self.assertRaises(NotImplementedError):
                ac.SpyreAsyncCompile().ktir("ktir_add_0", _add_specs())
            run.assert_not_called()

    def test_dbo_set_invokes_dbo_opt_and_returns_runner(self):
        """With TORCH_SPYRE_DBO=1, dbo-opt is invoked and the SDSC runner returned."""
        os.environ["TORCH_SPYRE_DBO"] = "1"

        recorded = {}

        def fake_run(cmd, check):
            recorded["cmd"] = cmd
            recorded["check"] = check
            # Fabricate the spyreCodeDir dbo-opt would export so the runner's
            # prepare_kernel target exists (its body is stubbed below anyway).
            export = next(a for a in cmd if a.startswith("--export-dir="))
            export_dir = export.split("=", 1)[1]
            os.makedirs(os.path.join(export_dir, "spyreCodeDir"), exist_ok=True)

        with (
            mock.patch.object(ac.subprocess, "run", side_effect=fake_run),
            mock.patch.object(kernel_runner, "prepare_kernel", return_value=object()),
        ):
            runner = ac.SpyreAsyncCompile().ktir("ktir_add_0", _add_specs())

        self.assertIsInstance(runner, SpyreSDSCKernelRunner)
        cmd = recorded["cmd"]
        self.assertTrue(recorded["check"])
        self.assertEqual(cmd[0], "dbo-opt")
        self.assertIn("--from-ktir", cmd)
        self.assertIn("--kEmitSpyreCode", cmd)
        # --export-dir points at the per-kernel output dir; the input .ktir and
        # the runner's code_dir both live there.
        export_arg = next(a for a in cmd if a.startswith("--export-dir="))
        output_dir = export_arg.split("=", 1)[1]
        ktir_path = cmd[-1]
        self.assertEqual(runner.code_dir, output_dir)
        self.assertEqual(ktir_path, os.path.join(output_dir, "ktir_add_0.ktir"))
        self.assertTrue(os.path.isfile(ktir_path))


if __name__ == "__main__":
    unittest.main()
