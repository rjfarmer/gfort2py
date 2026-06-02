# SPDX-License-Identifier: GPL-2.0+

import os
from pathlib import Path

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

from .conftest import build_paths

pyq = pytest.importorskip("pyquadp", reason="Requires pyquadp to be installed")

SO, MOD = build_paths("quad", "quad", as_path=True)

x = gf.fFort(SO, MOD)

if not hasattr(x, "const_real_qp"):
    pytest.skip("Requires gfortran REAL128 support", allow_module_level=True)


LP_INT128_TEST_VALUE = pyq.qint((1 << 100) + 12345)


def _q(v):
    return pyq.qfloat(v)


def _qc(v):
    return pyq.qcmplx(v)


def _qi(v):
    return pyq.qint(v)


class TestQuadMethods:
    def test_lp_kind_parameter_access(self):
        assert int(x.lp) >= 8

    def test_const_int_lp(self):
        assert x.const_int_lp == 1

    def test_lp_parameters_are_read_only(self):
        with pytest.raises(AttributeError):
            x.lp = 8

        with pytest.raises(AttributeError):
            x.const_int_lp = 2

    def test_a_int_lp_set_get(self):
        x.a_int_lp = 5
        assert x.a_int_lp == 5
        assert x.a_int_lp_set == 6

    def test_a_int_lp_default_array_get(self):
        expected = np.array([1, 2, 3, 4], dtype=np.int64)
        assert np.array_equal(x.a_int_lp_arr, expected)

    def test_sub_int_lp_scalar_inout(self):
        y = x.sub_int_lp_scalar_inout(3)
        assert y.args["x"] == 6

    def test_module_explicit_int_lp_array_set_get(self):
        arr = np.array([4, 3, 2, 1], dtype=np.int64)
        x.a_int_lp_arr = arr
        assert np.array_equal(x.a_int_lp_arr, arr)

    def test_module_explicit_int_lp_array_set_get_int128(self):
        arr = pyq.qiarray.from_list(
            [
                _qi((1 << 96) + 1),
                _qi((1 << 96) + 2),
                _qi((1 << 96) + 3),
                _qi((1 << 96) + 4),
            ]
        )
        x.a_int_lp_arr = arr
        actual = np.asarray(x.a_int_lp_arr, dtype=object)
        assert [v for v in actual.tolist()] == [v for v in arr.tolist()]

    def test_module_alloc_int_lp_array_get(self):
        x.sub_alloc_int_lp_module_arr(4, 8)
        expected = np.full(4, 8, dtype=np.int64)
        assert np.array_equal(x.a_int_lp_alloc_arr, expected)

    def test_module_alloc_int_lp_array_get_int128(self):
        val = _qi((1 << 100) + 99)
        x.sub_alloc_int_lp_module_arr(4, val)
        expected = pyq.qiarray.full(4, val)
        actual = np.asarray(x.a_int_lp_alloc_arr, dtype=object)
        assert [v for v in actual.tolist()] == [v for v in expected.tolist()]

    def test_int_lp_explicit_array_argument(self):
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        y = x.func_int_lp_explicit_arr_1d(arr)
        assert y.result
        assert np.array_equal(y.args["x"], np.array([2, 3, 4, 5], dtype=np.int64))

    @pytest.mark.skip(reason="Needs quad return support in ctypes")
    def test_func_int_lp_ret(self):
        y = x.func_int_lp_ret()
        assert y.result == 42

    def test_func_int_lp_return_array(self):
        y = x.func_int_lp_return_array()
        expected = np.array([1, 2, 3, 4], dtype=object)
        assert np.array_equal(np.asarray(y.result, dtype=object), expected)

    def test_func_int_lp_return_alloc_array(self):
        y = x.func_int_lp_return_alloc_array(4)
        expected = np.array([10, 20, 30, 40], dtype=object)
        assert np.array_equal(np.asarray(y.result, dtype=object), expected)

    def test_func_int_lp_return_from_assumed_shape(self):
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        y = x.func_int_lp_return_from_assumed_shape(arr)
        expected = np.array([6, 7, 8, 9], dtype=object)
        assert np.array_equal(np.asarray(y.result, dtype=object), expected)

    def test_func_int_lp_return_from_assumed_size(self):
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        y = x.func_int_lp_return_from_assumed_size(arr, arr.size)
        expected = np.array([8, 9, 10, 11], dtype=object)
        assert np.array_equal(np.asarray(y.result, dtype=object), expected)

    def test_a_int_lp_set_get_int128(self):
        x.a_int_lp = LP_INT128_TEST_VALUE
        assert x.a_int_lp == pyq.qint(LP_INT128_TEST_VALUE)

    def test_sub_int_lp_scalar_inout_int128(self):
        y = x.sub_int_lp_scalar_inout(LP_INT128_TEST_VALUE)
        assert y.args["x"] == LP_INT128_TEST_VALUE * 2

    def test_const_real_qp(self):
        assert x.const_real_qp == _q(1.0)

    def test_a_real_qp_set_get(self):
        x.a_real_qp = _q("2.5")
        assert x.a_real_qp == _q("2.5")

    def test_sub_alter_mod_scalar(self):
        x.a_real_qp = _q(1.0)
        x.sub_alter_mod()
        assert x.a_real_qp == _q(99.0)

    def test_sub_test_quad_scalar_arg(self):
        y = x.sub_test_quad(_q(4.0), None)
        assert y.args["x"] == _q(12.0)

    def test_sub_qp_scalar_inout(self):
        y = x.sub_qp_scalar_inout(_q(3.0))
        assert y.args["x"] == _q(6.0)

    def test_module_explicit_qp_array_set_get(self):
        arr = pyq.qarray.from_list([4.0, 3.0, 2.0, 1.0])
        x.a_real_qp_arr = arr
        assert np.array_equal(x.a_real_qp_arr, arr)

    def test_module_alloc_qp_array_get(self):
        x.sub_alloc_qp_module_arr(4, _q(8.0))
        expected = pyq.qarray.full(4, 8.0)
        assert np.array_equal(x.a_real_qp_alloc_arr, expected)

    def test_qp_explicit_array_argument(self):
        arr = pyq.qarray.from_list([1.0, 2.0, 3.0, 4.0])
        y = x.func_qp_explicit_arr_1d(arr)
        assert y.result
        assert np.array_equal(y.args["x"], pyq.qarray.from_list([2.0, 3.0, 4.0, 5.0]))

    def test_qp_assumed_shape_array_argument(self):
        arr = pyq.qarray.from_list([1.0, 2.0, 3.0, 4.0])
        y = x.func_qp_assumed_shape_arr_1d(arr)
        assert y.result
        assert np.array_equal(y.args["x"], pyq.qarray.from_list([3.0, 4.0, 5.0, 6.0]))

    def test_qp_assumed_size_array_argument(self):
        arr = pyq.qarray.from_list([1.0, 2.0, 3.0, 4.0])
        y = x.func_qp_assumed_size_arr_1d(arr, arr.size)
        assert y.result
        assert np.array_equal(y.args["x"], pyq.qarray.from_list([4.0, 5.0, 6.0, 7.0]))

    def test_qp_assumed_rank_array_argument_1d(self):
        arr = pyq.qarray.from_list([1.0, 2.0, 3.0, 4.0])
        y = x.func_qp_assumed_rank_arr(arr)
        assert y.result

    def test_qp_assumed_rank_array_argument_2d(self):
        arr = pyq.qarray.asarray([[1.0, 3.0], [2.0, 4.0]])
        y = x.func_qp_assumed_rank_arr(arr)
        assert y.result

    def test_const_cmplx_qp(self):
        assert x.const_cmplx_qp == _qc(complex(1.0, 1.0))

    def test_a_cmplx_qp_set_get(self):
        x.a_cmplx_qp = _qc(complex(2.0, -3.0))
        assert x.a_cmplx_qp == _qc(complex(2.0, -3.0))

    def test_sub_qcmplx_qp_scalar_inout(self):
        y = x.sub_qcmplx_qp_scalar_inout(_qc(complex(2.0, -3.0)))
        assert y.args["x"] == _qc(complex(3.0, -4.0))

    def test_module_explicit_qcmplx_qp_array_set_get(self):
        arr = pyq.qcarray.from_list(
            [
                complex(4.0, -4.0),
                complex(3.0, -3.0),
                complex(2.0, -2.0),
                complex(1.0, -1.0),
            ]
        )
        x.a_cmplx_qp_arr = arr
        assert np.array_equal(x.a_cmplx_qp_arr, arr)

    def test_module_alloc_qcmplx_qp_array_get(self):
        x.sub_alloc_qcmplx_qp_module_arr(4, _qc(complex(8.0, -8.0)))
        expected = pyq.qcarray.full(4, complex(8.0, -8.0))
        assert np.array_equal(x.a_cmplx_qp_alloc_arr, expected)

    def test_qcmplx_qp_explicit_array_argument(self):
        arr = pyq.qcarray.from_list(
            [
                complex(1.0, -1.0),
                complex(2.0, -2.0),
                complex(3.0, -3.0),
                complex(4.0, -4.0),
            ]
        )
        y = x.func_qcmplx_qp_explicit_arr_1d(arr)
        assert y.result
        assert np.array_equal(
            y.args["x"],
            pyq.qcarray.linspace(complex(2.0, 0.0), complex(5.0, -3.0), 4),
        )

    def test_qcmplx_qp_assumed_shape_array_argument(self):
        arr = pyq.qcarray.from_list(
            [
                complex(1.0, -1.0),
                complex(2.0, -2.0),
                complex(3.0, -3.0),
                complex(4.0, -4.0),
            ]
        )
        y = x.func_qcmplx_qp_assumed_shape_arr_1d(arr)
        assert y.result
        assert np.array_equal(
            y.args["x"],
            pyq.qcarray.linspace(complex(3.0, -1.0), complex(6.0, -4.0), 4),
        )

    def test_qcmplx_qp_assumed_size_array_argument(self):
        arr = pyq.qcarray.from_list(
            [
                complex(1.0, -1.0),
                complex(2.0, -2.0),
                complex(3.0, -3.0),
                complex(4.0, -4.0),
            ]
        )
        y = x.func_qcmplx_qp_assumed_size_arr_1d(arr, arr.size)
        assert y.result
        assert np.array_equal(
            y.args["x"],
            pyq.qcarray.linspace(complex(4.0, -4.0), complex(7.0, -7.0), 4),
        )

    def test_qcmplx_qp_assumed_rank_array_argument_1d(self):
        arr = pyq.qcarray.from_list(
            [
                complex(1.0, -1.0),
                complex(2.0, -2.0),
                complex(3.0, -3.0),
                complex(4.0, -4.0),
            ]
        )
        y = x.func_qcmplx_qp_assumed_rank_arr(arr)
        assert y.result

    def test_qcmplx_qp_assumed_rank_array_argument_2d(self):
        arr = pyq.qcarray.asarray(
            [
                [complex(1.0, -1.0), complex(3.0, -3.0)],
                [complex(2.0, -2.0), complex(4.0, -4.0)],
            ]
        )
        y = x.func_qcmplx_qp_assumed_rank_arr(arr)
        assert y.result

    def test_func_qp_return_array(self):
        y = x.func_qp_return_array()
        expected = pyq.qarray.from_list([1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(y.result, expected)

    def test_func_qp_return_alloc_array(self):
        y = x.func_qp_return_alloc_array(4)
        expected = pyq.qarray.from_list([10.0, 20.0, 30.0, 40.0])
        assert np.array_equal(y.result, expected)

    def test_func_qp_return_from_assumed_shape(self):
        arr = pyq.qarray.from_list([1.0, 2.0, 3.0, 4.0])
        y = x.func_qp_return_from_assumed_shape(arr)
        expected = pyq.qarray.from_list([6.0, 7.0, 8.0, 9.0])
        assert np.array_equal(y.result, expected)

    def test_func_qp_return_from_assumed_size(self):
        arr = pyq.qarray.from_list([1.0, 2.0, 3.0, 4.0])
        y = x.func_qp_return_from_assumed_size(arr, arr.size)
        expected = pyq.qarray.from_list([8.0, 9.0, 10.0, 11.0])
        assert np.array_equal(y.result, expected)

    def test_func_qcmplx_qp_return_array(self):
        y = x.func_qcmplx_qp_return_array()
        expected = pyq.qcarray.from_list(
            [
                complex(1.0, -1.0),
                complex(2.0, -2.0),
                complex(3.0, -3.0),
                complex(4.0, -4.0),
            ]
        )
        assert np.array_equal(y.result, expected)

    def test_func_qcmplx_qp_return_alloc_array(self):
        y = x.func_qcmplx_qp_return_alloc_array(4)
        expected = pyq.qcarray.from_list(
            [
                complex(10.0, -10.0),
                complex(20.0, -20.0),
                complex(30.0, -30.0),
                complex(40.0, -40.0),
            ]
        )
        assert np.array_equal(y.result, expected)

    def test_func_qcmplx_qp_return_from_assumed_shape(self):
        arr = pyq.qcarray.from_list(
            [
                complex(1.0, -1.0),
                complex(2.0, -2.0),
                complex(3.0, -3.0),
                complex(4.0, -4.0),
            ]
        )
        y = x.func_qcmplx_qp_return_from_assumed_shape(arr)
        expected = pyq.qcarray.from_list(
            [
                complex(6.0, -6.0),
                complex(7.0, -7.0),
                complex(8.0, -8.0),
                complex(9.0, -9.0),
            ]
        )
        assert np.array_equal(y.result, expected)

    def test_func_qcmplx_qp_return_from_assumed_size(self):
        arr = pyq.qcarray.from_list(
            [
                complex(1.0, -1.0),
                complex(2.0, -2.0),
                complex(3.0, -3.0),
                complex(4.0, -4.0),
            ]
        )
        y = x.func_qcmplx_qp_return_from_assumed_size(arr, arr.size)
        expected = pyq.qcarray.from_list(
            [
                complex(8.0, -8.0),
                complex(9.0, -9.0),
                complex(10.0, -10.0),
                complex(11.0, -11.0),
            ]
        )
        assert np.array_equal(y.result, expected)
