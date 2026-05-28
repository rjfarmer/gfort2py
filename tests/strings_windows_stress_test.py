# SPDX-License-Identifier: GPL-2.0+

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_windows_worker(loop_count: int) -> subprocess.CompletedProcess[str]:
    script = f"""
import os
from pathlib import Path

import numpy as np

import gfort2py as gf

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

root = Path(r"{ROOT}")
so = root / "tests" / "build" / f"strings.{{gf.lib_ext()}}"
mod = root / "tests" / "build" / "strings.mod"

x = gf.fFort(str(so), str(mod))

for _ in range({loop_count}):
    y = np.array(["a/b/c/d/e/f/g"], dtype="S13")
    assert x.check_assumed_shape_str_value(y).result

    assert x.check_str_opt(None, 0).result == 3
    assert x.check_str_opt("123456", 6).result == 1
    assert x.check_str_opt("abcedfg", 7).result == 2
    assert x.check_str_opt("abcd", 4).result == 4

print("OK")
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows stress test")
def test_strings_windows_subprocess_stress():
    # Run in a subprocess to surface native heap corruption as a hard failure.
    result = _run_windows_worker(loop_count=2000)

    if result.returncode != 0:
        pytest.fail(
            "Windows subprocess stress worker failed.\n"
            f"returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert "OK" in result.stdout
