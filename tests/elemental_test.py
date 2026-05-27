# SPDX-License-Identifier: GPL-2.0+

import os

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

from .conftest import build_paths

SO, MOD = build_paths("elemental", "elements")

x = gf.fFort(SO, MOD)


@pytest.mark.xfail(reason="Elemental procedure behavior under test development")
class TestElementalMethods:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (3, 6),
            (np.array([1, 2, 3], dtype=np.int32), np.array([2, 4, 6], dtype=np.int32)),
            (
                np.array([[1, 2], [3, 4]], dtype=np.int32),
                np.array([[2, 4], [6, 8]], dtype=np.int32),
            ),
        ],
    )
    def test_ele_func_1(self, value, expected):
        out = x.ele_func_1(value).result
        if isinstance(expected, np.ndarray):
            assert np.array_equal(out, expected)
        else:
            assert out == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (7, 14),
            (
                np.array([0, 5, 10], dtype=np.int32),
                np.array([0, 10, 20], dtype=np.int32),
            ),
            (
                np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int32),
                np.array([[2, 6, 10], [4, 8, 12]], dtype=np.int32),
            ),
        ],
    )
    def test_ele_func_res(self, value, expected):
        out = x.ele_func_res(value).result
        if isinstance(expected, np.ndarray):
            assert np.array_equal(out, expected)
        else:
            assert out == expected

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (2, 5, 14),
            (
                np.array([1, 2, 3], dtype=np.int32),
                np.array([4, 5, 6], dtype=np.int32),
                np.array([10, 14, 18], dtype=np.int32),
            ),
            (
                np.array([[1, 2], [3, 4]], dtype=np.int32),
                np.array([[5, 6], [7, 8]], dtype=np.int32),
                np.array([[12, 16], [20, 24]], dtype=np.int32),
            ),
        ],
    )
    def test_ele_func_2(self, left, right, expected):
        out = x.ele_func_2(left, right).result
        if isinstance(expected, np.ndarray):
            assert np.array_equal(out, expected)
        else:
            assert out == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (2.0, 6.0),
            (
                np.array([1.0, 2.0, 3.5], dtype=np.float64),
                np.array([3.0, 6.0, 10.5], dtype=np.float64),
            ),
            (
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                np.array([[3.0, 6.0], [9.0, 12.0]], dtype=np.float64),
            ),
        ],
    )
    def test_ele_sub_2(self, value, expected):
        out = x.ele_sub_2(value, expected).args["y"]
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(out, expected)
        else:
            assert out == expected

    def test_call_ele(self):
        x.call_ele()
