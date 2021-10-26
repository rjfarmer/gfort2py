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

SO = "./tests/complex.so"
MOD = "./tests/comp.mod"

x = gf.fFort(SO, MOD, rerun=True)


@contextmanager
def captured_output():
    """
    For use when we need to grab the stdout/stderr from fortran (but only in testing)
    Use as:
    with captured_output() as (out,err):
        func()
    output=out.getvalue().strip()
    error=err.getvalue().strip()
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestComplexMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_a_const_cmplx(self):
        self.assertEqual(x.const_cmplx, complex(1.0, 1.0))

    def test_a_const_cmplx_dp(self):
        self.assertEqual(x.const_cmplx_dp, complex(1.0, 1.0))

    def test_a_const_cmplx_qp(self):
        self.assertEqual(x.const_cmplx_qp, complex(1.0, 1.0))

    def test_a_cmplx(self):
        v = complex(1.0, 1.0)
        x.a_cmplx = v
        self.assertEqual(x.a_cmplx, v)

    def test_a_cmplx_dp(self):
        v = complex(1.0, 1.0)
        x.a_cmplx_dp = v
        self.assertEqual(x.a_cmplx_dp, v)

    def test_a_cmplx_qp(self):
        v = complex(1.0, 1.0)
        x.a_cmplx_qp = v
        self.assertEqual(x.a_cmplx_qp, v)

    def test_sub_cmplx_inout(self):
        v = complex(1.0, 1.0)
        y = x.sub_cmplx_inout(v)
        self.assertEqual(y.args["c"], v * 5)

    def test_func_cmplx_value(self):
        v = complex(1.0, 1.0)
        y = x.sub_cmplx_value(v, v)
        self.assertEqual(y.args["cc"], v * 5)

    def test_func_ret_cmplx(self):
        v = complex(1.0, 1.0)
        y = x.func_ret_cmplx(v)
        self.assertEqual(y.result, v * 5)
