# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/explicit_arrays.{gf.lib_ext()}"
MOD = "./tests/explicit_arrays.mod"

x = gf.fFort(SO, MOD)


class TestExplicitArrayMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_const_int_arr_error(self):
        with pytest.raises(AttributeError) as cm:
            x.const_int_arr = "abc"

    def test_const_int_arr(self):
        np.testing.assert_array_equal(
            x.const_int_arr, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype="int32")
        )

    def test_const_real_arr(self):
        np.testing.assert_array_equal(
            x.const_real_arr,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0], dtype="float"),
        )

    def test_const_dp_arr(self):
        np.testing.assert_array_equal(
            x.const_real_dp_arr,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0], dtype="float"),
        )

    def test_b_int_exp_1d(self):
        v = np.random.randint(0, 100, size=(5))
        x.b_int_exp_1d = v
        np.testing.assert_array_equal(x.b_int_exp_1d, v)

    def test_b_int_exp_2d(self):
        v = np.asfortranarray(np.random.randint(0, 100, size=(5, 5)), dtype="int32")
        x.b_int_exp_2d = v
        np.testing.assert_array_equal(x.b_int_exp_2d, v)

    def test_b_int_exp_3d(self):
        v = np.random.randint(0, 100, size=(5, 5, 5))
        x.b_int_exp_3d = v
        np.testing.assert_array_equal(x.b_int_exp_3d, v)

    def test_b_int_exp_4d(self):
        v = np.random.randint(0, 100, size=(5, 5, 5, 5))
        x.b_int_exp_4d = v
        np.testing.assert_array_equal(x.b_int_exp_4d, v)

    def test_b_int_exp_5d(self):
        v = np.random.randint(0, 100, size=(5, 5, 5, 5, 5))
        x.b_int_exp_5d = v
        np.testing.assert_array_equal(x.b_int_exp_5d, v)

    def test_b_real_exp_1d(self):
        v = np.random.random(size=(5))
        x.b_real_exp_1d = v
        np.testing.assert_allclose(x.b_real_exp_1d, v)

    def test_b_real_exp_2d(self):
        v = np.random.random(size=(5, 5))
        x.b_real_exp_2d = v
        np.testing.assert_allclose(x.b_real_exp_2d, v)

    def test_b_real_exp_3d(self):
        v = np.random.random(size=(5, 5, 5))
        x.b_real_exp_3d = v
        np.testing.assert_allclose(x.b_real_exp_3d, v)

    def test_b_real_exp_4d(self):
        v = np.random.random(size=(5, 5, 5, 5))
        x.b_real_exp_4d = v
        np.testing.assert_allclose(x.b_real_exp_4d, v)

    def test_b_real_exp_5d(self):
        v = np.random.random(size=(5, 5, 5, 5, 5))
        x.b_real_exp_5d = v
        np.testing.assert_allclose(x.b_real_exp_5d, v)

    def test_b_real_exp_5d_2(self):
        v = np.random.random(size=(2, 3, 4, 5, 6))
        x.b_real_exp_5d_2 = v
        np.testing.assert_allclose(x.b_real_exp_5d_2, v)

    def test_b_real_dp_exp_1d(self):
        v = np.random.random(size=(5))
        x.b_real_dp_exp_1d = v
        np.testing.assert_allclose(x.b_real_dp_exp_1d, v)

    def test_b_real_dp_exp_2d(self):
        v = np.random.random(size=(5, 5))
        x.b_real_dp_exp_2d = v
        np.testing.assert_allclose(x.b_real_dp_exp_2d, v)

    def test_b_real_dp_exp_3d(self):
        v = np.random.random(size=(5, 5, 5))
        x.b_real_dp_exp_3d = v
        np.testing.assert_allclose(x.b_real_dp_exp_3d, v)

    def test_b_real_dp_exp_4d(self):
        v = np.random.random(size=(5, 5, 5, 5))
        x.b_real_dp_exp_4d = v
        np.testing.assert_allclose(x.b_real_dp_exp_4d, v)

    def test_b_real_dp_exp_5d(self):
        v = np.random.random(size=(5, 5, 5, 5, 5))
        x.b_real_dp_exp_5d = v
        np.testing.assert_allclose(x.b_real_dp_exp_5d, v)

    @pytest.mark.skip
    def test_sub_array_n_int_1d(self, capfd):
        v = np.arange(0, 5)
        o = " ".join([str(i) for i in v.flatten(order="F")])

        y = x.sub_array_n_int_1d(np.size(v), v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    @pytest.mark.skip
    def test_sub_array_n_int_2d(self, capfd):
        v = [0, 1, 2, 3, 4] * 5
        v = np.array(v).reshape(5, 5)
        o = " ".join([str(i) for i in np.asfortranarray(v).flatten(order="F")])

        y = x.sub_array_n_int_2d(5, 5, v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_int_1d(self, capfd):
        v = np.arange(0, 5)
        o = " ".join([str(i) for i in v.flatten(order="F")])

        y = x.sub_exp_array_int_1d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_int_2d(self, capfd):
        v = np.arange(0, 5 * 5).reshape((5, 5))
        o = "".join(
            [str(i).zfill(2).ljust(3) for i in np.asfortranarray(v).flatten(order="F")]
        )
        y = x.sub_exp_array_int_2d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_int_3d(self, capfd):
        v = np.arange(0, 5 * 5 * 5).reshape((5, 5, 5))
        o = "".join(
            [str(i).zfill(3).ljust(4) for i in np.asfortranarray(v).flatten(order="F")]
        )
        y = x.sub_exp_array_int_3d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_real_1d(self, capfd):
        v = np.arange(0, 5.0).reshape((5))
        o = "  ".join(
            ["{:>4.1f}".format(i) for i in np.asfortranarray(v).flatten(order="F")]
        )
        y = x.sub_exp_array_real_1d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_real_2d(self, capfd):
        v = np.arange(0, 5.0 * 5.0).reshape((5, 5))
        o = "  ".join(
            ["{:>4.1f}".format(i) for i in np.asfortranarray(v).flatten(order="F")]
        )
        y = x.sub_exp_array_real_2d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_real_3d(self, capfd):
        v = np.arange(0, 5.0 * 5.0 * 5.0).reshape((5, 5, 5))
        o = " ".join(
            ["{:>5.1f}".format(i) for i in np.asfortranarray(v).flatten(order="F")]
        )
        y = x.sub_exp_array_real_3d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_int_1d_multi(self, capfd):
        u = 19
        w = 20
        v = np.arange(0, 5)
        o = " ".join([str(i) for i in np.asfortranarray(v).flatten(order="F")])
        y = x.sub_exp_array_int_1d_multi(u, v, w)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), str(u) + " " + o.strip() + " " + str(w))

    def test_sub_exp_array_real_dp_1d(self, capfd):
        v = np.arange(0, 5.0).reshape((5))
        o = "  ".join(
            ["{:>4.1f}".format(i) for i in np.asfortranarray(v).flatten(order="F")]
        )

        y = x.sub_exp_array_real_dp_1d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_real_dp_2d(self, capfd):
        v = np.arange(0, 5.0 * 5.0).reshape((5, 5))
        o = "  ".join(
            ["{:>4.1f}".format(i) for i in np.asfortranarray(v).flatten(order="F")]
        )

        y = x.sub_exp_array_real_dp_2d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_array_real_dp_3d(self, capfd):
        v = np.arange(0, 5.0 * 5.0 * 5.0).reshape((5, 5, 5))
        o = " ".join(
            ["{:>5.1f}".format(i) for i in np.asfortranarray(v).flatten(order="F")]
        )

        y = x.sub_exp_array_real_dp_3d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_sub_exp_inout(self, capfd):
        v = np.array([1, 2, 3, 4, 5])

        y = x.sub_exp_inout(v)
        out, err = capfd.readouterr()

        np.testing.assert_array_equal(y.args["x"], 2 * v)

    def test_sub_arr_exp_p(self, capfd):
        v = np.arange(0, 5)
        o = " ".join([str(i) for i in v.flatten(order="F")])

        y = x.sub_exp_array_int_1d(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), o.strip())

    def test_logical_arr_multi(self):
        xarr = np.zeros(5)
        xarr[:] = True

        y = x.func_logical_multi(1.0, 2.0, xarr, 3.0, 4.0)
        self.assertEqual(y.result, True)

    @pytest.mark.skip("Skipping as we seg fault")
    def test_mesh_exp(self):
        # Github issue #13
        i = 5
        y = x.func_mesh_exp(i)
        self.assertEqual(y.result, np.arrange(0, i))

    @pytest.mark.skip
    def test_check_exp_2d_2m3(self):
        # Github issue #19
        arr_test = np.zeros((3, 4), dtype=int, order="F")

        arr_test[0, 1] = 1
        arr_test[1, 0] = 2
        arr_test[1, 2] = 3
        arr_test[-2, -1] = 4

        y = x.check_exp_2d_2m3_nt(arr_test, 4, 0)

        self.assertEqual(y.args["success"], True)

        arr_test[0, 3] = 5

        np.testing.assert_array_equal(y.args["arr"], arr_test)
