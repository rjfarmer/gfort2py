# SPDX-License-Identifier: GPL-2.0+

import ctypes
import os
import sys
from pathlib import Path
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

from .conftest import build_paths

gf_version = gf.utils.gfortran_version(gf.utils.fc_path())

if gf_version < gf.utils.Version("15.0.0"):
    pytest.skip("Requires gfortran 15 or later", allow_module_level=True)

SO, MOD = build_paths("unsign", "unsign", as_path=True)

x = gf.fFort(SO, MOD)


class TestUnsigned:
    # --- scalar module variable ---

    def test_a_u_set_get(self):
        x.a_u = 42
        assert x.a_u == 42

    def test_a_u_set_zero(self):
        x.a_u = 0
        assert x.a_u == 0

    def test_a_u_set_large(self):
        v = 2**32 - 1  # max uint32
        x.a_u = v
        assert x.a_u == v

    # --- scalar parameter (read-only) ---

    def test_a_u_param(self):
        assert x.a_u_param == 32

    def test_a_u_param_readonly(self):
        with pytest.raises(AttributeError):
            x.a_u_param = 1

    # --- 1-d array parameter ---

    def test_a_u_array_param_1d(self):
        expected = np.array([1, 2, 3, 4, 5], dtype=np.uintc)
        assert np.array_equal(x.a_u_array_param_1d, expected)

    # --- 1-d array module variable ---

    def test_a_u_array_1d_set_get(self):
        v = np.array([10, 20, 30, 40, 50], dtype=np.uintc)
        x.a_u_array_1d = v
        assert np.array_equal(x.a_u_array_1d, v)

    # --- subroutine: set_unsigned_scalar ---

    def test_set_unsigned_scalar(self):
        y = x.set_unsigned_scalar(None, 7)
        assert y.args["x"] == 7

    # --- function: add_unsigned ---

    def test_add_unsigned(self):
        y = x.add_unsigned(3, 4)
        assert y.result == 7

    def test_add_unsigned_large(self):
        a = 2**31
        b = 2**31 - 1
        y = x.add_unsigned(a, b)
        assert y.result == 2**32 - 1

    # --- function: max_unsigned ---

    def test_max_unsigned_first_larger(self):
        y = x.max_unsigned(10, 5)
        assert y.result == 10

    def test_max_unsigned_second_larger(self):
        y = x.max_unsigned(3, 99)
        assert y.result == 99

    def test_max_unsigned_equal(self):
        y = x.max_unsigned(7, 7)
        assert y.result == 7

    # --- subroutine: copy_unsigned_array ---
    @pytest.mark.skipIfWindows(
        reason="Sometimes causes heap crashes on Windows, needs investigation"
    )
    def test_copy_unsigned_array(self):
        src = np.array([1, 2, 3, 4, 5], dtype=np.uintc)
        dst = np.zeros(5, dtype=np.uintc)
        y = x.copy_unsigned_array(src, dst)
        assert np.array_equal(y.args["dst"], src)

    # --- subroutine: scale_unsigned_array ---

    def test_scale_unsigned_array(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uintc)
        y = x.scale_unsigned_array(arr, 3)
        expected = np.array([3, 6, 9, 12, 15], dtype=np.uintc)
        assert np.array_equal(y.args["arr"], expected)

    # --- function: sum_unsigned_array ---

    def test_sum_unsigned_array(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uintc)
        y = x.sum_unsigned_array(arr)
        assert y.result == 15

    def test_sum_unsigned_array_zeros(self):
        arr = np.zeros(5, dtype=np.uintc)
        y = x.sum_unsigned_array(arr)
        assert y.result == 0

    # --- function: add_unsigned_arrays ---
    def test_add_unsigned_arrays(self):
        lhs = np.array([1, 2, 3, 4, 5], dtype=np.uintc)
        rhs = np.array([5, 4, 3, 2, 1], dtype=np.uintc)
        y = x.add_unsigned_arrays(lhs, rhs)
        expected = np.array([6, 6, 6, 6, 6], dtype=np.uintc)
        assert np.array_equal(y.result, expected)

    # --- function: shift_unsigned_array ---
    def test_shift_unsigned_array(self):
        arr = np.array([10, 20, 30, 40, 50], dtype=np.uintc)
        y = x.shift_unsigned_array(arr, 5)
        expected = np.array([15, 25, 35, 45, 55], dtype=np.uintc)
        assert np.array_equal(y.result, expected)
