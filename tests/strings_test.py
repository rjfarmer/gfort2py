# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/strings.{gf.lib_ext()}"
MOD = "./tests/strings.mod"

x = gf.fFort(SO, MOD)


class TestStringMethods:
    def assertEqual(self, x, y):
        assert x == y

    def test_a_str(self):
        v = "123456798 "
        x.a_str = v
        self.assertEqual(x.a_str, v)

    def test_a_str_bad_length(self):
        v = "132456789kjhgjhf"
        x.a_str = v
        self.assertEqual(x.a_str, v[0:10])

    def test_sub_str_in_explicit(self, capfd):
        v = "1324567980"

        y = x.sub_str_in_explicit(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), v)

    def test_sub_str_in_implicit(self, capfd):
        v = "123456789"

        y = x.sub_str_in_implicit(v)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), v)

    def test_sub_str_multi(self, capfd):
        v = 5
        u = "123456789"
        w = 4

        y = x.sub_str_multi(v, u, w)
        out, err = capfd.readouterr()
        self.assertEqual(out.strip(), str(v + w) + " " + u)

    def test_sub_str_p(self, capfd):
        y = x.sub_str_p("abcdef")
        out, err = capfd.readouterr()
        assert err == ""
        self.assertEqual(y.args["zzz"], "xyzxyz")
        self.assertEqual(out.strip(), "abcdef")

    def test_func_ret_str(self):
        y = x.func_ret_str("abcde")
        self.assertEqual(y.result, "Abcde")

    # We need to call a func on the argument before passing it to func_str_int_len
    def test_func_str_int_len(self, capfd):
        out, err = capfd.readouterr()
        y = x.func_str_int_len(10)

        assert y.result == "10"

    def test_str_alloc(self):
        self.assertEqual(x.str_alloc, None)  # Empty at start

        x.str_alloc = "abcdefghijklmnop"
        self.assertEqual(x.str_alloc, "abcdefghijklmnop")
        y = x.check_str_alloc(1)
        self.assertEqual(y.result, True)

        x.str_alloc = "12345678        "  # Need to empty the space afterwards
        self.assertEqual(x.str_alloc, "12345678        ")
        y = x.check_str_alloc(2)
        self.assertEqual(y.result, False)

    def test_str_alloc_sub(self):
        z = None
        y = x.sub_str_alloc(z)
        self.assertEqual(y.args["x_alloc"], "abcdef")

        y2 = x.sub_str_alloc2(None)
        self.assertEqual(y2.args["x"], "zxcvbnm")

    @pytest.mark.skip
    def test_str_alloc_sub_realloc(self):
        y2 = x.sub_str_alloc2("qwerty")
        self.assertEqual(y2.args["x"], "asdfghjkl")

    def test_str_array_type_chceck(self):
        with pytest.raises(TypeError) as cm:
            x.a_str_exp_1d = np.zeros(5, dtype=np.str_)

    def test_str_func_inout_str(self, capfd):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        res = x.str_array_inout(z)
        out, err = capfd.readouterr()

        assert out.strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_func_str_array_dt(self):
        data = {
            "start_guard": 123456789,
            "a_str_exp_1d": np.array(
                ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
                dtype="S10",
            ),
            "end_guard": 123456789,
        }

        assert not x.func_str_array_dt(data).result

        data = {
            "start_guard": 123456789,
            "a_str_exp_1d": np.array(
                ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
                dtype="S10",
            ),
            "end_guard": 123456789,
        }

        assert x.func_str_array_dt(data).result

    def test_func_str_array_dt_alloc(self):
        data = {
            "start_guard": 123456789,
            "b_str_alloc_1d": None,
            "end_guard": 123456789,
        }

        res = x.func_str_array_dt_alloc(data)

        assert res.result

        z = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(z == res.args["x"]["b_str_alloc_1d"])

    def test_str_func_inout_str2(self, capfd):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        res = x.str_array_inout2(z, 5)
        out, err = capfd.readouterr()

        assert out.strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_str_func_inout_str3(self, capfd):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        res = x.str_array_inout3(z)
        out, err = capfd.readouterr()

        assert out.strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_str_func_inout_alloc(self, capfd):
        z = None

        res = x.str_array_allocate(z)

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    @pytest.mark.skip
    def test_str_func_inout_str4(self, capfd):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        res = x.str_array_inout4(z)
        out, err = capfd.readouterr()

        assert out.strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_str_array_param(self):
        assert np.all(x.a_str_p_1d == np.array(["aa", "bb", "cc"], dtype="S2"))

    def test_check_a_str_exp_1d(self, capfd):
        x.a_str_exp_1d = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        res = x.check_a_str_exp_1d()
        out, err = capfd.readouterr()

        assert out.strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(x.a_str_exp_1d == z2)

    def test_b_str_alloc_1d(self):
        x.b_str_alloc_1d = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        res = x.check_b_str_alloc_1d()

        assert res.result

        x.b_str_alloc_1d = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb"],
            dtype="S10",
        )

        res = x.check_b_str_alloc_1d()

        assert not res.result

    def test_alloc_b_str_alloc_1d(self):
        res = x.alloc_b_str_alloc_1d()

        assert np.all(
            x.b_str_alloc_1d
            == np.array(
                ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
                dtype="S10",
            )
        )

    def test_set_str_array_dt_out(self):
        res = x.set_str_array_dt_out({})

        assert res.args["x"]["a_str1"] == "qwertyuiop[]"
        assert res.args["x"]["a_str2"] == "asdfghjkl;zx"
        assert res.args["x"]["a_int"] == 99

    def test_set_chr_star_star(self):
        res = x.set_chr_star_star("            ")
        assert res.args["x"] == "abcdefghijkl"

    def test_check_assumed_shape_str(self, capfd):
        y = np.array(["a/b/c/d/e/f/g"], dtype="S")

        res = x.check_assumed_shape_str(y)
        out, err = capfd.readouterr()
        assert out.strip() == "a/b/c/d/e/f/g"

    def test_check_str_opt(self):
        res = x.check_str_opt(None, 0)
        assert res.result == 3

        res = x.check_str_opt("123456", 6)
        assert res.result == 1

        res = x.check_str_opt("abcedfg", 7)
        assert res.result == 2

        res = x.check_str_opt("abcd", 4)
        assert res.result == 4
