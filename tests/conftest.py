# SPDX-License-Identifier: GPL-2.0+

import subprocess
import pytest
import _pytest.skipping


def pytest_configure(config):
    subprocess.call(["make", "clean"], shell=True, cwd="tests")
    subprocess.call(["make"], shell=True, cwd="tests")
