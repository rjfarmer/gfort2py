# SPDX-License-Identifier: GPL-2.0+

import os, sys

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

import subprocess
import numpy.testing as np_test

from contextlib import contextmanager
from io import StringIO
from io import BytesIO

# Decreases recursion depth to make debugging easier
# sys.setrecursionlimit(10)

SO = "./tests/oo.so"
MOD = "./tests/oo.mod"

x = gf.fFort(SO, MOD)


@pytest.mark.skip
class TestOOMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_p_proc_call(self):
        x.p_proc.proc_no_pass = x.func_dt_no_pass
        y = x.p_proc.proc_no_pass(1)
        y2 = x.func_dt_no_pass(1)

        self.assertEqual(y.result, y2.result)
