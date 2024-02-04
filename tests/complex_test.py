# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

try:
    import pyquadp as pyq

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False

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

    def test_a_cmplx(self):
        v = complex(1.0, 1.0)
        x.a_cmplx = v
        self.assertEqual(x.a_cmplx, v)

    def test_a_cmplx_dp(self):
        v = complex(1.0, 1.0)
        x.a_cmplx_dp = v
        self.assertEqual(x.a_cmplx_dp, v)

    @pytest.mark.skipif(gf.utils.is_big_endian(), reason="Skip on big endian systems")
    def test_sub_cmplx_inout(self):
        v = complex(1.0, 1.0)
        y = x.sub_cmplx_inout(v)
        self.assertEqual(y.args["c"], v * 5)

    @pytest.mark.skipif(gf.utils.is_ppc64le(), reason="Skip on ppc64le systems")
    @pytest.mark.skipif(gf.utils.is_big_endian(), reason="Skip on big endian systems")
    def test_func_cmplx_value(self):
        v = complex(1.0, 1.0)
        y = x.sub_cmplx_value(v, v)
        self.assertEqual(y.args["cc"], v * 5)

    def test_func_ret_cmplx(self):
        v = complex(1.0, 1.0)
        y = x.func_ret_cmplx(v)
        self.assertEqual(y.result, v * 5)
