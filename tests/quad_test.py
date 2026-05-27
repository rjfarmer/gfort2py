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
    @pytest.mark.xfail(
        reason="Quad precision scalar constant decoding is currently incorrect"
    )
    def test_const_real_qp(self):
        assert x.const_real_qp == _q(1.0)

    @pytest.mark.xfail(
        reason="Quad precision scalar module variable set/get currently fails type marshaling"
    )
    def test_a_real_qp_set_get(self):
        x.a_real_qp = _q("2.5")
        assert x.a_real_qp == _q("2.5")

    @pytest.mark.xfail(
        reason="Quad precision scalar module variable set currently fails type marshaling"
    )
    def test_sub_alter_mod_scalar(self):
        x.a_real_qp = _q(1.0)
        x.sub_alter_mod()
        assert x.a_real_qp == _q(99.0)

    @pytest.mark.xfail(
        reason="Quad precision scalar argument passing currently fails type marshaling"
    )
    def test_sub_test_quad_scalar_arg(self):
        y = x.sub_test_quad(_q(4.0), None)
        assert y.args["x"] == _q(12.0)

    @pytest.mark.xfail(
        reason="Quad precision scalar inout argument passing currently fails type marshaling"
    )
    def test_sub_qp_scalar_inout(self):
        y = x.sub_qp_scalar_inout(_q(3.0))
        assert y.args["x"] == _q(6.0)

    @pytest.mark.xfail(
        reason="Quad precision module array set/get is not fully supported yet"
    )
    def test_module_explicit_qp_array_set_get(self):
        arr = np.array([_q(4.0), _q(3.0), _q(2.0), _q(1.0)], dtype=object)
        x.a_real_qp_arr = arr
        assert np.array_equal(x.a_real_qp_arr, arr)

    @pytest.mark.xfail(
        reason="Quad precision allocatable module arrays are not fully supported yet"
    )
    def test_module_alloc_qp_array_get(self):
        x.sub_alloc_qp_module_arr(4, _q(8.0))
        expected = np.array([_q(8.0), _q(8.0), _q(8.0), _q(8.0)], dtype=object)
        assert np.array_equal(x.a_real_qp_alloc_arr, expected)

    @pytest.mark.xfail(
        reason="Quad precision explicit-shape array arguments are not fully supported yet"
    )
    def test_qp_explicit_array_argument(self):
        arr = np.array([_q(1.0), _q(2.0), _q(3.0), _q(4.0)], dtype=object)
        y = x.func_qp_explicit_arr_1d(arr)
        assert y.result
        assert np.array_equal(
            y.args["x"], np.array([_q(2.0), _q(3.0), _q(4.0), _q(5.0)], dtype=object)
        )

    @pytest.mark.xfail(
        reason="Quad precision assumed-shape array arguments are not fully supported yet"
    )
    def test_qp_assumed_shape_array_argument(self):
        arr = np.array([_q(1.0), _q(2.0), _q(3.0), _q(4.0)], dtype=object)
        y = x.func_qp_assumed_shape_arr_1d(arr)
        assert y.result
        assert np.array_equal(
            y.args["x"], np.array([_q(3.0), _q(4.0), _q(5.0), _q(6.0)], dtype=object)
        )

    @pytest.mark.xfail(
        reason="Quad precision assumed-size array arguments are not fully supported yet"
    )
    def test_qp_assumed_size_array_argument(self):
        arr = np.array([_q(1.0), _q(2.0), _q(3.0), _q(4.0)], dtype=object)
        y = x.func_qp_assumed_size_arr_1d(arr, arr.size)
        assert y.result
        assert np.array_equal(
            y.args["x"], np.array([_q(4.0), _q(5.0), _q(6.0), _q(7.0)], dtype=object)
        )

    @pytest.mark.xfail(
        reason="Quad precision assumed-rank array arguments are not fully supported yet"
    )
    def test_qp_assumed_rank_array_argument_1d(self):
        arr = np.array([_q(1.0), _q(2.0), _q(3.0), _q(4.0)], dtype=object)
        y = x.func_qp_assumed_rank_arr(arr)
        assert y.result

    @pytest.mark.xfail(
        reason="Quad precision assumed-rank array arguments are not fully supported yet"
    )
    def test_qp_assumed_rank_array_argument_2d(self):
        arr = np.array(
            [[_q(1.0), _q(3.0)], [_q(2.0), _q(4.0)]], dtype=object, order="F"
        )
        y = x.func_qp_assumed_rank_arr(arr)
        assert y.result

    @pytest.mark.xfail(
        reason="Quad precision complex scalar constant decoding is currently incorrect"
    )
    def test_const_cmplx_qp(self):
        assert x.const_cmplx_qp == _qc(complex(1.0, 1.0))

    @pytest.mark.xfail(
        reason="Quad precision complex scalar module variable set/get currently fails type marshaling"
    )
    def test_a_cmplx_qp_set_get(self):
        x.a_cmplx_qp = _qc(complex(2.0, -3.0))
        assert x.a_cmplx_qp == _qc(complex(2.0, -3.0))

    @pytest.mark.xfail(
        reason="Quad precision complex scalar argument passing currently fails type marshaling"
    )
    def test_sub_qcmplx_qp_scalar_inout(self):
        y = x.sub_qcmplx_qp_scalar_inout(_qc(complex(2.0, -3.0)))
        assert y.args["x"] == _qc(complex(3.0, -4.0))

    @pytest.mark.xfail(
        reason="Quad precision complex module array set/get is not fully supported yet"
    )
    def test_module_explicit_qcmplx_qp_array_set_get(self):
        arr = np.array(
            [
                _qc(complex(4.0, -4.0)),
                _qc(complex(3.0, -3.0)),
                _qc(complex(2.0, -2.0)),
                _qc(complex(1.0, -1.0)),
            ],
            dtype=object,
        )
        x.a_cmplx_qp_arr = arr
        assert np.array_equal(x.a_cmplx_qp_arr, arr)

    @pytest.mark.xfail(
        reason="Quad precision complex allocatable module arrays are not fully supported yet"
    )
    def test_module_alloc_qcmplx_qp_array_get(self):
        x.sub_alloc_qcmplx_qp_module_arr(4, _qc(complex(8.0, -8.0)))
        expected = np.array(
            [
                _qc(complex(8.0, -8.0)),
                _qc(complex(8.0, -8.0)),
                _qc(complex(8.0, -8.0)),
                _qc(complex(8.0, -8.0)),
            ],
            dtype=object,
        )
        assert np.array_equal(x.a_cmplx_qp_alloc_arr, expected)

    @pytest.mark.xfail(
        reason="Quad precision complex explicit-shape array arguments are not fully supported yet"
    )
    def test_qcmplx_qp_explicit_array_argument(self):
        arr = np.array(
            [
                _qc(complex(1.0, -1.0)),
                _qc(complex(2.0, -2.0)),
                _qc(complex(3.0, -3.0)),
                _qc(complex(4.0, -4.0)),
            ],
            dtype=object,
        )
        y = x.func_qcmplx_qp_explicit_arr_1d(arr)
        assert y.result
        assert np.array_equal(
            y.args["x"],
            np.array(
                [
                    _qc(complex(2.0, 0.0)),
                    _qc(complex(3.0, -1.0)),
                    _qc(complex(4.0, -2.0)),
                    _qc(complex(5.0, -3.0)),
                ],
                dtype=object,
            ),
        )

    @pytest.mark.xfail(
        reason="Quad precision complex assumed-shape array arguments are not fully supported yet"
    )
    def test_qcmplx_qp_assumed_shape_array_argument(self):
        arr = np.array(
            [
                _qc(complex(1.0, -1.0)),
                _qc(complex(2.0, -2.0)),
                _qc(complex(3.0, -3.0)),
                _qc(complex(4.0, -4.0)),
            ],
            dtype=object,
        )
        y = x.func_qcmplx_qp_assumed_shape_arr_1d(arr)
        assert y.result
        assert np.array_equal(
            y.args["x"],
            np.array(
                [
                    _qc(complex(3.0, -1.0)),
                    _qc(complex(4.0, -2.0)),
                    _qc(complex(5.0, -3.0)),
                    _qc(complex(6.0, -4.0)),
                ],
                dtype=object,
            ),
        )

    @pytest.mark.xfail(
        reason="Quad precision complex assumed-size array arguments are not fully supported yet"
    )
    def test_qcmplx_qp_assumed_size_array_argument(self):
        arr = np.array(
            [
                _qc(complex(1.0, -1.0)),
                _qc(complex(2.0, -2.0)),
                _qc(complex(3.0, -3.0)),
                _qc(complex(4.0, -4.0)),
            ],
            dtype=object,
        )
        y = x.func_qcmplx_qp_assumed_size_arr_1d(arr, arr.size)
        assert y.result
        assert np.array_equal(
            y.args["x"],
            np.array(
                [
                    _qc(complex(4.0, -4.0)),
                    _qc(complex(5.0, -5.0)),
                    _qc(complex(6.0, -6.0)),
                    _qc(complex(7.0, -7.0)),
                ],
                dtype=object,
            ),
        )

    @pytest.mark.xfail(
        reason="Quad precision complex assumed-rank array arguments are not fully supported yet"
    )
    def test_qcmplx_qp_assumed_rank_array_argument_1d(self):
        arr = np.array(
            [
                _qc(complex(1.0, -1.0)),
                _qc(complex(2.0, -2.0)),
                _qc(complex(3.0, -3.0)),
                _qc(complex(4.0, -4.0)),
            ],
            dtype=object,
        )
        y = x.func_qcmplx_qp_assumed_rank_arr(arr)
        assert y.result

    @pytest.mark.xfail(
        reason="Quad precision complex assumed-rank array arguments are not fully supported yet"
    )
    def test_qcmplx_qp_assumed_rank_array_argument_2d(self):
        arr = np.array(
            [
                [_qc(complex(1.0, -1.0)), _qc(complex(3.0, -3.0))],
                [_qc(complex(2.0, -2.0)), _qc(complex(4.0, -4.0))],
            ],
            dtype=object,
            order="F",
        )
        y = x.func_qcmplx_qp_assumed_rank_arr(arr)
        assert y.result
