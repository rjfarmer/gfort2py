# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/dummy_arrays.{gf.lib_ext()}"
MOD = "./tests/dummy_arrays.mod"

x = gf.fFort(SO, MOD)


# @pytest.mark.skip
class TestDummyArrayMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_sub_alloc_1d_arrs(self):
        y = x.sub_alloc_int_1d_arrs()

    def test_c_int_alloc_1d_non_alloc(self):
        y = x.sub_alloc_int_1d_cleanup()
        self.assertEqual(x.c_int_alloc_1d, None)

    def test_ndarray(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()

        self.assertEqual(np.sum(x.c_int_alloc_1d), 5)

    def test_c_int_alloc_1d(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5])
        v[:] = 1
        assert np.array_equal(x.c_int_alloc_1d, v)

    def test_c_int_alloc_2d(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5])
        v[:] = 1
        assert np.array_equal(x.c_int_alloc_2d, v)

    def test_c_int_alloc_3d(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_int_alloc_3d, v)

    def test_c_int_alloc_4d(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_int_alloc_4d, v)

    def test_c_int_alloc_5d(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5, 5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_int_alloc_5d, v)

    def test_c_int_alloc_1d_set(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5])
        v[:] = 5
        x.c_int_alloc_1d = v
        assert np.array_equal(x.c_int_alloc_1d, v)
        self.assertEqual(np.sum(x.c_int_alloc_1d), 25)

    def test_c_int_alloc_2d_set(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5])
        v[:] = 5
        x.c_int_alloc_2d = v
        assert np.array_equal(x.c_int_alloc_2d, v)

    def test_c_int_alloc_3d_set(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5, 5])
        v[:] = 5
        x.c_int_alloc_3d = v
        assert np.array_equal(x.c_int_alloc_3d, v)

    def test_c_int_alloc_4d_set(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5, 5, 5])
        v[:] = 5
        x.c_int_alloc_4d = v
        assert np.array_equal(x.c_int_alloc_4d, v)

    @pytest.mark.skip
    def test_c_int_alloc_5d_set(self):
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([5, 5, 5, 5, 5])
        v[:] = 5
        x.c_int_alloc_5d = v
        assert np.array_equal(x.c_int_alloc_5d, v)

    @pytest.mark.skip
    def test_c_int_alloc_1d_large(self):
        # Can have issues exiting when using large (>255) arrays
        y = x.sub_alloc_int_1d_cleanup()
        y = x.sub_alloc_int_1d_arrs()
        v = np.zeros([256], dtype="int32")
        v[:] = 5
        x.c_int_alloc_1d = v
        assert np.array_equal(x.c_int_alloc_1d, v)

    def test_c_real_alloc_1d(self):
        y = x.sub_alloc_real_1d_cleanup()
        y = x.sub_alloc_real_1d_arrs()
        v = np.zeros([5])
        v[:] = 1
        assert np.array_equal(x.c_real_alloc_1d, v)

    def test_c_real_alloc_2d(self):
        y = x.sub_alloc_real_1d_cleanup()
        y = x.sub_alloc_real_1d_arrs()
        v = np.zeros([5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_alloc_2d, v)

    def test_c_real_alloc_3d(self):
        y = x.sub_alloc_real_1d_cleanup()
        y = x.sub_alloc_real_1d_arrs()
        v = np.zeros([5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_alloc_3d, v)

    def test_c_real_alloc_4d(self):
        y = x.sub_alloc_real_1d_cleanup()
        y = x.sub_alloc_real_1d_arrs()
        v = np.zeros([5, 5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_alloc_4d, v)

    def test_c_real_alloc_5d(self):
        y = x.sub_alloc_real_1d_cleanup()
        y = x.sub_alloc_real_1d_arrs()
        v = np.zeros([5, 5, 5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_alloc_5d, v)

    def test_c_real_dp_alloc_1d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5])
        v[:] = 2.0
        x.c_real_dp_alloc_1d = v
        assert np.array_equal(x.c_real_dp_alloc_1d, v)

    def test_c_real_dp_alloc_2d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_2d = v
        assert np.array_equal(x.c_real_dp_alloc_2d, v)

    def test_c_real_dp_alloc_3d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_3d = v
        assert np.array_equal(x.c_real_dp_alloc_3d, v)

    def test_c_real_dp_alloc_4d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_4d = v
        assert np.array_equal(x.c_real_dp_alloc_4d, v)

    @pytest.mark.skip
    def test_c_real_dp_alloc_5d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5, 5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_5d = v
        assert np.array_equal(x.c_real_dp_alloc_5d, v)

    def test_c_real_dp_alloc_1d(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5])
        v[:] = 1
        assert np.array_equal(x.c_real_dp_alloc_1d, v)

    def test_c_real_dp_alloc_2d(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_dp_alloc_2d, v)

    def test_c_real_dp_alloc_3d(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_dp_alloc_3d, v)

    def test_c_real_dp_alloc_4d(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_dp_alloc_4d, v)

    def test_c_real_dp_alloc_5d(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5, 5, 5])
        v[:] = 1
        assert np.array_equal(x.c_real_dp_alloc_5d, v)

    def test_c_real_dp_alloc_1d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5])
        v[:] = 2.0
        x.c_real_dp_alloc_1d = v
        assert np.array_equal(x.c_real_dp_alloc_1d, v)

    def test_c_real_dp_alloc_2d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_2d = v
        assert np.array_equal(x.c_real_dp_alloc_2d, v)

    def test_c_real_dp_alloc_3d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_3d = v
        assert np.array_equal(x.c_real_dp_alloc_3d, v)

    @pytest.mark.skip
    def test_c_real_dp_alloc_4d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_4d = v
        assert np.array_equal(x.c_real_dp_alloc_4d, v)

    @pytest.mark.skip
    def test_c_real_dp_alloc_5d_set(self):
        y = x.sub_alloc_real_dp_1d_cleanup()
        y = x.sub_alloc_real_dp_1d_arrs()
        v = np.zeros([5, 5, 5, 5, 5])
        v[:] = 2.0
        x.c_real_dp_alloc_5d = v
        assert np.array_equal(x.c_real_dp_alloc_5d, v)

    def test_func_assumed_shape_arr_1d(self):
        v = np.zeros([5], dtype="int32")
        v[0] = 2.0
        y = x.func_assumed_shape_arr_1d(v)
        self.assertEqual(y.result, True)
        assert np.array_equal(y.args["x"], np.array([9, 9, 9, 9, 9]))

    def test_func_assumed_shape_arr_2d(self):
        v = np.zeros([5, 5], dtype="int32")
        v[1, 0] = 2.0
        y = x.func_assumed_shape_arr_2d(v)
        self.assertEqual(y.result, True)

    def test_func_assumed_shape_arr_3d(self):
        v = np.zeros([5, 5, 5], dtype="int32")
        v[2, 1, 0] = 2.0
        y = x.func_assumed_shape_arr_3d(v)
        self.assertEqual(y.result, True)

    def test_func_assumed_shape_arr_4d(self):
        v = np.zeros([5, 5, 5, 5], dtype="int32")
        v[3, 2, 1, 0] = 2.0
        y = x.func_assumed_shape_arr_4d(v)
        self.assertEqual(y.result, True)

    def test_func_assumed_shape_arr_5d(self):
        v = np.zeros([5, 5, 5, 5, 5], dtype="int32")
        v[4, 3, 2, 1, 0] = 2.0
        y = x.func_assumed_shape_arr_5d(v)
        self.assertEqual(y.result, True)

    def test_func_assumed_size_arr_1d(self):
        v = np.zeros([5], dtype="int32")
        v[1] = 2
        y = x.func_assumed_size_arr_1d(v)
        self.assertEqual(y.result, True)

    def test_func_assumed_size_arr_real_1d(self):
        v = np.zeros([5], dtype="float32")
        v[1] = 2.0
        y = x.func_assumed_size_arr_real_1d(v)
        self.assertEqual(y.result, True)

    def test_func_assumed_size_arr_real_dp_1d(self):
        v = np.zeros([5], dtype="float64")
        v[1] = 2.0
        y = x.func_assumed_size_arr_real_dp_1d(v)
        self.assertEqual(y.result, True)

    def test_sub_alloc_arr_1d(self):
        y = x.sub_alloc_arr_1d(None)
        vTest = np.zeros(10)
        vTest[:] = 10
        assert np.array_equal(y.args["x"], vTest)

    def test_logical_arr(self):
        xarr = np.zeros(10)
        x2arr = np.zeros(10)
        x2arr[:] = False
        xarr[:] = True

        y = x.func_alltrue_arr_1d(xarr)
        y2 = x.func_allfalse_arr_1d(x2arr)
        self.assertEqual(y.result, True)
        self.assertEqual(y2.result, True)

    def test_sub_arr_assumed_rank_int_1d(self, capfd):
        v = np.arange(10, 15)
        o = " ".join([str(i) for i in v.flatten(order="F")])

        y = x.sub_arr_assumed_rank_int_1d(v)
        out, err = capfd.readouterr()
        assert np.array_equal(y.args["zzz"], np.array([100] * 5))

    def test_sub_arr_assumed_rank_real_1d(self, capfd):
        v = np.arange(10.0, 15.0)
        o = " ".join([str(i) for i in v.flatten(order="F")])

        y = x.sub_arr_assumed_rank_real_1d(v)
        out, err = capfd.readouterr()
        assert np.array_equal(y.args["zzz"], np.array([100.0] * 5))

    def test_sub_arr_assumed_rank_dp_1d(self, capfd):
        v = np.arange(10.0, 15.0)
        o = " ".join([str(i) for i in v.flatten(order="F")])

        y = x.sub_arr_assumed_rank_dp_1d(v)
        out, err = capfd.readouterr()
        assert np.array_equal(y.args["zzz"], np.array([100.0] * 5))

    @pytest.mark.skip
    def test_sub_check_alloc_int_2d(self):
        arr_test = np.zeros((3, 4), dtype=np.int32, order="F")
        y = x.sub_check_alloc_int_2d(None)
        y2 = x.sub_check_alloc_int_2d(arr_test)
        assert np.array_equal(y.args["x"], y2.args["x"])

        z = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        assert np.array_equal(y.args["x"], z)

    @pytest.mark.skip
    def test_sub_check_alloc_int_3d(self):
        arr_test = np.zeros((3, 4, 5), dtype=np.int32, order="F")
        y = x.sub_check_alloc_int_3d(None)
        y2 = x.sub_check_alloc_int_3d(arr_test)
        assert np.array_equal(y.args["x"], y2.args["x"])

        z = np.array(
            [
                [
                    [1, 2, 3, 4, 5],
                    [5, 6, 7, 8, 9],
                    [9, 10, 11, 12, 13],
                    [13, 14, 15, 16, 17],
                ],
                [
                    [5, 6, 7, 8, 9],
                    [9, 10, 11, 12, 13],
                    [13, 14, 15, 16, 17],
                    [17, 18, 19, 20, 21],
                ],
                [
                    [9, 10, 11, 12, 13],
                    [13, 14, 15, 16, 17],
                    [17, 18, 19, 20, 21],
                    [21, 22, 23, 24, 25],
                ],
            ]
        )

        assert np.array_equal(y.args["x"], z)

    # GH:39
    def test_sub_multi_array_pass(self):
        y = 0.0
        xp = np.array([-0.6, 1.25])
        yp = np.array([0.0, 0.0])

        res = x.multi_array_pass(y, xp, yp)

        assert res.args["y"] == -1000.0
        assert np.array_equal(res.args["xp"], np.array([13, -2]))
        assert np.array_equal(res.args["yp"], np.array([1, -42.014]))
