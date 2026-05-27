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

SO, MOD = build_paths("complex", "comp")

x = gf.fFort(SO, MOD)


class TestComplexMethods:

    def test_a_const_cmplx(self):
        assert x.const_cmplx == complex(1.0, 1.0)

    def test_a_const_cmplx_dp(self):
        assert x.const_cmplx_dp == complex(1.0, 1.0)

    def test_a_cmplx(self):
        v = complex(1.0, 1.0)
        x.a_cmplx = v
        assert x.a_cmplx == v

    def test_a_cmplx_dp(self):
        v = complex(1.0, 1.0)
        x.a_cmplx_dp = v
        assert x.a_cmplx_dp == v

    @pytest.mark.skipif(gf.utils.is_big_endian(), reason="Skip on big endian systems")
    def test_sub_cmplx_inout(self):
        v = complex(1.0, 1.0)
        y = x.sub_cmplx_inout(v)
        assert y.args["c"] == v * 5

    @pytest.mark.skipif(gf.utils.is_ppc64le(), reason="Skip on ppc64le systems")
    @pytest.mark.skipif(gf.utils.is_big_endian(), reason="Skip on big endian systems")
    def test_func_cmplx_value(self):
        v = complex(1.0, 1.0)
        y = x.sub_cmplx_value(v)
        assert y.args["cc"] == v * 5

    def test_func_ret_cmplx(self):
        v = complex(1.0, 1.0)
        y = x.func_ret_cmplx(v)
        assert y.result == v * 5

    @pytest.mark.xfail(
        reason="Complex module array variable roundtrip currently fails dtype conversion from structured real/imag buffers",
    )
    def test_a_cmplx_arr(self):
        v = np.array(
            [
                complex(1.0, 1.0),
                complex(2.0, -1.0),
                complex(3.0, 2.0),
                complex(4.0, 3.0),
                complex(5.0, -4.0),
            ],
            dtype=np.complex64,
        )
        x.a_cmplx_arr = v
        assert np.array_equal(x.a_cmplx_arr, v)

    @pytest.mark.xfail(
        reason="Complex(dp) module array variable roundtrip currently fails dtype conversion from structured real/imag buffers",
    )
    def test_a_cmplx_dp_arr(self):
        v = np.array(
            [
                [complex(1.0, 1.0), complex(2.0, -1.0), complex(3.0, 2.0)],
                [complex(4.0, 3.0), complex(5.0, -4.0), complex(6.0, 5.0)],
            ],
            dtype=np.complex128,
            order="F",
        )
        x.a_cmplx_dp_arr = v
        assert np.array_equal(x.a_cmplx_dp_arr, v)

    @pytest.mark.xfail(
        reason="Complex explicit-shape array argument output conversion currently fails dtype conversion",
    )
    def test_func_cmplx_explicit_arr_2d(self):
        v = np.zeros((2, 3), dtype=np.complex64, order="F")
        v[1, 0] = complex(2.0, 1.0)
        y = x.func_cmplx_explicit_arr_2d(v)
        assert y.result
        assert y.args["x"][0, 0] == complex(9.0, -1.0)

    def test_func_cmplx_assumed_shape_arr_1d(self):
        v = np.zeros([5], dtype=np.complex64)
        v[0] = complex(2.0, 1.0)
        y = x.func_cmplx_assumed_shape_arr_1d(v)
        assert y.result
        assert np.array_equal(
            y.args["x"], np.array([complex(5.0, 2.0)] * 5, dtype=np.complex64)
        )

    @pytest.mark.xfail(
        reason="Complex assumed-size array argument output conversion currently fails dtype conversion",
    )
    def test_func_cmplx_assumed_size_arr_1d(self):
        v = np.zeros([5], dtype=np.complex64)
        v[1] = complex(3.0, -2.0)
        y = x.func_cmplx_assumed_size_arr_1d(v)
        assert y.result
        assert y.args["x"][0] == complex(11.0, -7.0)

    def test_func_cmplx_assumed_rank_arr_1d(self):
        v = np.zeros([5], dtype=np.complex64)
        v[0] = complex(2.0, 1.0)
        y = x.func_cmplx_assumed_rank_arr(v)
        assert y.result

    def test_func_cmplx_assumed_rank_arr_2d(self):
        v = np.zeros((2, 3), dtype=np.complex64, order="F")
        v[1, 0] = complex(2.0, 1.0)
        y = x.func_cmplx_assumed_rank_arr(v)
        assert y.result

    def test_func_ret_cmplx_arr_1d(self):
        y = x.func_ret_cmplx_arr_1d()
        assert np.array_equal(
            y.result,
            np.array(
                [complex(1.0, -1.0), complex(2.0, -2.0), complex(3.0, -3.0)],
                dtype=np.complex64,
            ),
        )

    def test_func_ret_cmplx_arr_n(self):
        y = x.func_ret_cmplx_arr_n(4)
        assert np.array_equal(
            y.result,
            np.array(
                [
                    complex(1.0, 2.0),
                    complex(2.0, 3.0),
                    complex(3.0, 4.0),
                    complex(4.0, 5.0),
                ],
                dtype=np.complex64,
            ),
        )
