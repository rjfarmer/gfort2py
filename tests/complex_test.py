# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/complex.{gf.lib_ext()}"
MOD = "./tests/comp.mod"

x = gf.fFort(SO, MOD)


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
        with pytest.raises(TypeError) as cm:
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
