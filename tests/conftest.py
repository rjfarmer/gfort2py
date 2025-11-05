# SPDX-License-Identifier: GPL-2.0+

import subprocess
import pytest
import os
import platform
import _pytest.skipping


def pytest_configure(config):
    subprocess.call(["make", "clean"], shell=True, cwd="tests")
    subprocess.call(["make"], shell=True, cwd="tests")


def pytest_runtest_setup(item):
    is_windows = platform.system() == "Windows"
    is_github = "GITHUB_ACTIONS" in os.environ

    for mark in item.iter_markers(name="skipIfWindows"):
        if is_github and is_windows:
            pytest.skip()
