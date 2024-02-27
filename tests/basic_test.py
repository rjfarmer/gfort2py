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


SO = f"./tests/basic.{gf.lib_ext()}"
MOD = "./tests/basic.mod"

x = gf.fFort(SO, MOD)


class TestBasicMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_version(self):
        assert gf.__version__

    def test_mod_info(self):
        assert len(gf.mod_info(MOD).keys())

    def test_doc(self):
        assert x.__doc__ == f"MODULE={MOD}"

    def test_str(self):
        assert str(x) == MOD

    def test_mising_var(self):
        with pytest.raises(AttributeError) as cm:
            a = x.invalid_var

    def test_lookups(self):
        assert len(x.keys())
        assert "a_int" in x
        assert len(dir(x))

    def test_a_int(self):
        v = 1
        x.a_int = v
        self.assertEqual(x.a_int, v)

    def test_a_int_str(self):
        with pytest.raises(TypeError) as cm:
            x.a_int = "abc"

    def test_a_real(self):
        v = 1.0
        x.a_real = v
        self.assertEqual(x.a_real, v)

    def test_a_real_str(self):
        with pytest.raises(TypeError) as cm:
            x.a_real = "abc"

    def test_const_int_set(self):
        with pytest.raises(AttributeError) as cm:
            x.const_int = 2

    def test_const_int(self):
        self.assertEqual(x.const_int, 1)

    def test_const_int_p1(self):
        self.assertEqual(x.const_int_p1, 2)

    def test_const_int_long(self):
        self.assertEqual(x.const_int_lp, 1)

    def test_const_real_dp(self):
        self.assertEqual(x.const_real_dp, 1.0)

    def test_const_real_pi_dp(self):
        self.assertEqual(x.const_real_pi_dp, 3.14)

    def test_sub_no_args(self, capfd):
        x.sub_no_args()
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "1")

    def test_sub_alter_mod(self):
        y = x.sub_alter_mod()
        self.assertEqual(x.a_int, 99)
        self.assertEqual(x.a_int_lp, 99)
        self.assertEqual(x.a_real, 99.0)
        self.assertEqual(x.a_real_dp, 99.0)

    def test_int_lp(self):
        x.a_int_lp = 5
        assert x.a_int_lp == 5
        assert x.a_int_lp_set == 6

    def test_func_int_in(self):
        v = 5
        y = x.func_int_in(v)
        self.assertEqual(int(y.result), 2 * v)

    def test_func_int_in_multi(self):
        v = 5
        w = 3
        u = 4
        y = x.func_int_in_multi(v, w, u)
        self.assertEqual(y.result, v + w + u)

    def test_sub_int_in(self, capfd):
        v = 5

        y = x.sub_int_in(v)
        out, err = capfd.readouterr()
        self.assertEqual(int(out), 2 * v)

    def test_func_int_no_args(self):
        y = x.func_int_no_args()
        self.assertEqual(y.result, 2)

    def test_func_real_no_args(self):
        y = x.func_real_no_args()
        self.assertEqual(y.result, 3.0)

    def test_func_real_dp_no_args(self):
        y = x.func_real_dp_no_args()
        self.assertEqual(y.result, 4.0)

    def test_sub_int_out(self, capfd):
        v = 5

        y = x.sub_int_out(v)
        out, err = capfd.readouterr()
        self.assertEqual(y.args, {"x": 1})

    def test_sub_int_inout(self, capfd):
        v = 5

        y = x.sub_int_inout(v)
        out, err = capfd.readouterr()
        self.assertEqual(y.args, {"x": 2 * v})

    def test_sub_int_no_intent(self, capfd):
        v = 5

        y = x.sub_int_no_intent(v)
        out, err = capfd.readouterr()
        self.assertEqual(y.args, {"x": 2 * v})

    def test_sub_real_inout(self, capfd):
        v = 5.0

        y = x.sub_real_inout(v)
        out, err = capfd.readouterr()
        self.assertEqual(y.args, {"x": 2 * v})

    def test_func_return_res(self):
        y = x.func_return_res(2)
        self.assertEqual(y.result, True)
        y = x.func_return_res(10)
        self.assertEqual(y.result, False)

    def test_sub_int_p(self, capfd):
        y = x.sub_int_p(1)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "1")
        self.assertEqual(y.args["zzz"], 5)

    def test_sub_real_p(self, capfd):
        y = x.sub_real_p(1.0)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "1.00")
        self.assertEqual(y.args["zzz"], 5.0)

    def test_sub_opt(self, capfd):
        y = x.sub_int_opt(1)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "100")

        y = x.sub_int_opt(None)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "200")

    def test_sub_opt_val(self, capfd):
        y = x.sub_int_opt_val(1)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "100")

        y = x.sub_int_opt_val(None)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), "200")

    def test_second_mod(self):
        y = x.sub_use_mod()
        self.assertEqual(x.test2_x, 1)

    def test_func_value(self):
        y = x.func_int_value(5)
        self.assertEqual(y.result, 10)

    def test_func_pass_mod_var(self):
        x.a_int = 5
        z = x.func_int_in(x.a_int)
        self.assertEqual(z.result, 10)

    def test_sub_man_args(self):
        # if this doesn't seg fault we are good
        x.sub_many_args(
            1, 2, 3, 4, True, False, True, "abc", "def", "ghj", "qwerty", "zxcvb"
        )

    def test_func_intent_out(self):
        y = x.func_intent_out(9, 0)
        self.assertEqual(y.result, 9)
        self.assertEqual(y.args["x"], 9)

        y = x.func_intent_out(x=0, y=9)  # Swap order of arguments
        self.assertEqual(y.result, 9)
        self.assertEqual(y.args["x"], 9)

    def test_func_result(self):
        y = x.func_result(9, 0)
        self.assertEqual(y.result, 18)
        self.assertEqual(y.args["y"], 9)

    def test_func_bool(self):
        y = x.func_test_bool(1)
        assert type(y.result) == bool
        self.assertEqual(y.result, True)
        y = x.func_test_bool(0)
        assert type(y.result) == bool
        self.assertEqual(y.result, False)

    def test_negatives(self):
        assert x.const_neg_int == -1
        assert x.const_neg_real == -3.14

    def test_logical_parammeters(self):
        assert x.const_logical_true
        assert not x.const_logical_false

    def test_func_check_mod(self):
        x.a_int = 5
        x.a_int_lp = 5
        x.a_real = 5.0
        x.a_real_dp = 5.0

        assert x.func_check_mod().result

    def test_mixed_case(self):

        assert x.const_int_MIXED == 1
        assert x.const_int_mixed == 1
        assert x.CONST_INT_MIXED == 1

        x.a_int_MIXED = 5
        assert x.a_int_MIXED == 5

        assert x.a_int_mixed == 5

        x.a_int_mixed = 6
        assert x.A_INT_MIXED == 6

        assert x.func_TEST_CASE().result
        assert x.func_test_case().result
        assert x.FUNC_test_case().result
