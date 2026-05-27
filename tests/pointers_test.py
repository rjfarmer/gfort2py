# SPDX-License-Identifier: GPL-2.0+

import ctypes
import os
import sys
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

try:
    import pyquadp as pyq

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False


SO = f"./tests/build/pointers.{gf.lib_ext()}"
MOD = "./tests/build/ptrs.mod"

x = gf.fFort(SO, MOD)


class TestPtrsMethods:

    def test_a_int_point(self):
        v = 1
        x.a_int_point = v
        assert x.a_int_point == v

    def test_a_int_lp_point(self):
        v = 1
        x.a_int_lp_point = v
        assert x.a_int_lp_point == v

    def test_a_real_point(self):
        v = 1.0
        x.a_real_point = v
        assert x.a_real_point == v

    def test_a_real_dp_point(self):
        v = 1.0
        x.a_real_dp_point = v
        assert x.a_real_dp_point == v

    def test_a_str_point(self):
        v = "abcdefghij"
        x.a_str_point = v
        assert x.a_str_point == v

    def test_a_int_target(self):
        v = 1
        x.a_int_target = v
        assert x.a_int_target == v

    def test_a_int_lp_target(self):
        v = 1
        x.a_int_lp_target = v
        assert x.a_int_lp_target == v

    def test_a_real_target(self):
        v = 1.0
        x.a_real_target = v
        assert x.a_real_target == v

    def test_a_real_dp_target(self):
        v = 1.0
        x.a_real_dp_target = v
        assert x.a_real_dp_target == v

    def test_a_str_target(self):
        v = "abcdefghij"
        x.a_str_target = v
        assert x.a_str_target == v

    @pytest.mark.skip
    def test_sub_set_scalar_pointer_and_get(self):
        y = x.sub_set_scalar_pointer_and_get(7)

        assert y.args["value_out"] == 14
        assert x.a_int_point == 14
        assert x.a_int_target == 14

    def test_sub_return_scalar_pointer_arg(self):
        x.sub_set_scalar_pointer_and_get(11)

        y = x.sub_return_scalar_pointer_arg()
        assert y.args["value_out"] == 22

    def test_sub_set_array_pointer_and_get(self):
        v = np.arange(1, 6, dtype=np.int32)
        expected = (v + 10) * 2

        y = x.sub_set_array_pointer_and_get(v)
        assert np.array_equal(y.args["values_out"], expected)
        assert np.array_equal(x.d_int_point_1d, expected)

    def test_sub_return_array_pointer_arg(self):
        v = np.arange(3, 8, dtype=np.int32)
        expected = (v + 10) * 2
        x.sub_set_array_pointer_and_get(v)

        y = x.sub_return_array_pointer_arg()
        assert np.array_equal(y.args["values_out"], expected)
