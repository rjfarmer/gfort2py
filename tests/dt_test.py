# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/dt.{gf.lib_ext()}"
MOD = "./tests/dt.mod"

x = gf.fFort(SO, MOD)


# @pytest.mark.skip
class TestDTMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_dt_set_value(self):
        x.f_struct_simple["x"] = 1
        x.f_struct_simple["y"] = 0
        y = x.f_struct_simple
        self.assertEqual(y["x"], 1)
        self.assertEqual(y["y"], 0)

    def test_dt_set_dict(self):
        x.f_struct_simple = {"x": 5, "y": 5}
        y = x.f_struct_simple
        self.assertEqual(y["x"], 5)
        self.assertEqual(y["y"], 5)

    def test_dt_bad_dict(self):
        with pytest.raises(KeyError) as cm:
            x.f_struct_simple = {"asw": 2, "y": 0}

    def test_dt_bad_value(self):
        with pytest.raises(TypeError) as cm:
            x.f_struct_simple["x"] = "asde"

    def test_sub_dt_in_s_simple(self, capfd):
        y = x.sub_f_simple_in({"x": 1, "y": 10})
        out, err = capfd.readouterr()
        o = " ".join([str(i) for i in [1, 10]])
        self.assertEqual(out.strip(), o)

    def test_sub_dt_out_s_simple(self, capfd):
        y = x.sub_f_simple_out({})
        out, err = capfd.readouterr()
        self.assertEqual(y.args["x"]["x"], 1)
        self.assertEqual(y.args["x"]["y"], 10)

    def test_sub_dt_inout_s_simple(self, capfd):
        y = x.sub_f_simple_inout({"x": 5, "y": 3})
        out, err = capfd.readouterr()
        o = "  ".join([str(i) for i in [5, 3]])
        self.assertEqual(out.strip(), o)
        self.assertEqual(y.args["zzz"]["x"], 1)
        self.assertEqual(y.args["zzz"]["y"], 10)

    def test_sub_dt_inoutp_s_simple(self, capfd):
        y = x.sub_f_simple_inoutp({"x": 5, "y": 3})
        out, err = capfd.readouterr()
        o = "  ".join([str(i) for i in [5, 3]])
        self.assertEqual(out.strip(), o)
        self.assertEqual(y.args["zzz"]["x"], 1)
        self.assertEqual(y.args["zzz"]["y"], 10)

    def test_nested_dts(self):
        x.g_struct["a_int"] = 10
        self.assertEqual(x.g_struct["a_int"], 10)
        x.g_struct = {"a_int": 10, "f_struct": {"a_int": 3}}
        self.assertEqual(x.g_struct["f_struct"]["a_int"], 3)
        x.g_struct["f_struct"]["a_int"] = 8
        self.assertEqual(x.g_struct["f_struct"]["a_int"], 8)
        y = x.func_check_nested_dt()
        self.assertEqual(y.result, True)

    def test_func_set_f_struct(self):
        y = x.func_set_f_struct()
        self.assertEqual(y.result, True)

        self.assertEqual(x.f_struct["a_int"], 5)
        self.assertEqual(x.f_struct["a_int_lp"], 6)
        self.assertEqual(x.f_struct["a_real"], 7.0)
        self.assertEqual(x.f_struct["a_real_dp"], 8.0)
        self.assertEqual(x.f_struct["a_str"], "9999999999")

        v = np.array([9, 10, 11, 12, 13], dtype="int32")
        assert np.array_equal(x.f_struct["b_int_exp_1d"], v)

        v = np.array([9, 10, 11, 12, 13], dtype="int32")
        assert np.array_equal(x.e_int_target_1d, v)

    def test_func_set_f_struct_array_alloc(self):
        y = x.func_set_f_struct()

        v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int32")
        assert np.array_equal(x.f_struct["c_int_alloc_1d"], v)

    def test_func_set_f_struct_array_ptr(self):
        y = x.func_set_f_struct()

        v = np.array([9, 10, 11, 12, 13], dtype="int32")
        assert np.array_equal(x.f_struct["d_int_point_1d"], v)

    # @pytest.mark.skip
    def test_recur_dt(self):  # Skip for now
        with pytest.raises(NotImplementedError) as cm:
            x.r_recur["a_int"] = 9
            self.assertEqual(x.r_recur["a_int"], 9)
            x.r_recur["s_recur"]["a_int"] = 9
            self.assertEqual(x.r_recur["s_recur"]["a_int"], 9)
            x.r_recur["s_recur"]["s_recur"]["a_int"] = 9
            self.assertEqual(x.r_recur["s_recur"]["s_recur"]["a_int"], 9)

    def test_arr_dt_exp_1d_set(self):
        x.g_struct_exp_1d[0]["a_int"] = 5
        self.assertEqual(x.g_struct_exp_1d[0]["a_int"], 5)
        x.g_struct_exp_1d[1]["a_int"] = 9
        self.assertEqual(x.g_struct_exp_1d[1]["a_int"], 9)
        self.assertEqual(
            x.g_struct_exp_1d[0]["a_int"], 5
        )  # recheck we didnt corrupt things

        with pytest.raises(IndexError) as cm:
            y = x.g_struct_exp_1d[10]

    def test_dt_not_array(self):
        with pytest.raises(KeyError) as cm:
            y = x.f_struct[9]["a_int"]

    def test_arr_dt_exp_2d_set(self):
        x.g_struct_exp_2d[0, 0]["a_int"] = 1
        x.g_struct_exp_2d[1, 0]["a_int"] = 2
        x.g_struct_exp_2d[0, 1]["a_int"] = 3
        x.g_struct_exp_2d[1, 1]["a_int"] = 4

        self.assertEqual(x.g_struct_exp_2d[0, 0]["a_int"], 1)
        self.assertEqual(x.g_struct_exp_2d[1, 0]["a_int"], 2)
        self.assertEqual(x.g_struct_exp_2d[0, 1]["a_int"], 3)
        self.assertEqual(x.g_struct_exp_2d[1, 1]["a_int"], 4)

        y = x.check_g_struct_exp_2d()
        self.assertEqual(y.result, True)

    def test_sub_struct_exp_1d(self):
        y = x.sub_struct_exp_1d({})

        s = y.args["x"]

        self.assertEqual(s[0]["a_int"], 5)
        self.assertEqual(s[1]["a_int"], 9)

        assert np.array_equal(s[0]["b_int_exp_1d"], np.array([66, 66, 66, 66, 66]))
        assert np.array_equal(s[1]["b_int_exp_1d"], np.array([77, 77, 77, 77, 77]))

    def test_fvar_as_arg(self, capfd):
        y = x.f_struct_simple
        y["x"] = 99
        y["y"] = 98

        z = x.sub_f_simple_in(y)
        out, err = capfd.readouterr()
        o = "99 98"
        self.assertEqual(out.strip(), o)

        z2 = x.sub_f_simple_in(z.args["x"])
        out, err = capfd.readouterr()
        o = "99 98"
        self.assertEqual(out.strip(), o)

    def test_func_return_s_struct_nested_2(self):
        y = x.func_return_s_struct_nested_2()
        self.assertEqual(y.result["a_int"], 123)
        self.assertEqual(y.result["f_nested"]["a_int"], 234)
        self.assertEqual(y.result["f_nested"]["f_struct"]["a_int"], 345)

    def test_derived_type_intent_out(self, capfd):
        # GH: #32
        p = {}
        y = x.derived_structure(p)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "10 20 30 40")

        assert np.all(y.args["p"]["iq"] == np.array([10, 20, 30, 40]))
