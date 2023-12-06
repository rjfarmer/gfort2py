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


SO = f"./tests/pointers.{gf.lib_ext()}"
MOD = "./tests/ptrs.mod"

x = gf.fFort(SO, MOD)


class TestPtrsMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_a_int_point(self):
        v = 1
        x.a_int_point = v
        self.assertEqual(x.a_int_point, v)

    def test_a_int_lp_point(self):
        v = 1
        x.a_int_lp_point = v
        self.assertEqual(x.a_int_lp_point, v)

    def test_a_real_point(self):
        v = 1.0
        x.a_real_point = v
        self.assertEqual(x.a_real_point, v)

    def test_a_real_dp_point(self):
        v = 1.0
        x.a_real_dp_point = v
        self.assertEqual(x.a_real_dp_point, v)

    def test_a_str_point(self):
        v = "abcdefghij"
        x.a_str_point = v
        self.assertEqual(x.a_str_point, v)

    def test_a_int_target(self):
        v = 1
        x.a_int_target = v
        self.assertEqual(x.a_int_target, v)

    def test_a_int_lp_target(self):
        v = 1
        x.a_int_lp_target = v
        self.assertEqual(x.a_int_lp_target, v)

    def test_a_real_target(self):
        v = 1.0
        x.a_real_target = v
        self.assertEqual(x.a_real_target, v)

    def test_a_real_dp_target(self):
        v = 1.0
        x.a_real_dp_target = v
        self.assertEqual(x.a_real_dp_target, v)

    def test_a_str_target(self):
        v = "abcdefghij"
        x.a_str_target = v
        self.assertEqual(x.a_str_target, v)
