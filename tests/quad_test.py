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


def _q(v):
    return pyq.qfloat(v)


def _qc(v):
    return pyq.qcmplx(v)


class TestQuadMethods:
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
