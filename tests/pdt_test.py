# SPDX-License-Identifier: GPL-2.0+

import ctypes
import os
import sys
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

from .conftest import build_paths

SO, MOD = build_paths("pdt", "pdt")


@pytest.mark.skip
class TestOOMethods:
    # Move loading here sometthing is wierd on Windows and we cant looad the module
    # Looks like an issue with captialisation of PDT names.
    x = gf.fFort(SO, MOD)

    def test_p_proc_call(self):
        pass
