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

x = gf.fFort(SO, MOD)


@pytest.mark.skip
class TestOOMethods:
    def test_p_proc_call(self):
        pass
