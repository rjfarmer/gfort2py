# SPDX-License-Identifier: GPL-2.0+

import os
import platform
import subprocess
from pathlib import Path

import _pytest.skipping
import pytest

import gfort2py as gf


def build_paths(
    lib_name: str, mod_name: str | None = None, *, as_path: bool = False
) -> tuple[str, str] | tuple[Path, Path]:
    mod_stem = lib_name if mod_name is None else mod_name
    so = f"./tests/build/{lib_name}.{gf.lib_ext()}"
    mod = f"./tests/build/{mod_stem}.mod"
    if as_path:
        return Path(so), Path(mod)
    return so, mod


def pytest_configure(config):
    subprocess.call(["make"], cwd="tests")


def pytest_runtest_setup(item):
    is_windows = platform.system() == "Windows"
    is_github = "GITHUB_ACTIONS" in os.environ

    for mark in item.iter_markers(name="skipIfWindows"):
        if is_github and is_windows:
            pytest.skip()
