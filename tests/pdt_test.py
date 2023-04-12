# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = "./tests/pdt.so"
MOD = "./tests/pdt.mod"

x = gf.fFort(SO, MOD)


@pytest.mark.skip
class TestOOMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_p_proc_call(self):
        pass
