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

SO, MOD = build_paths("strings", "strings")

x = gf.fFort(SO, MOD)


class TestStringMethods:

    def test_a_str(self):
        v = "123456798 "
        x.a_str = v
        assert x.a_str == v

    def test_a_str_bad_length(self):
        v = "132456789kjhgjhf"
        x.a_str = v
        assert x.a_str == v[0:10]

    def test_sub_str_in_explicit(self, fortran_output):
        v = "1324567980"

        with fortran_output() as get_output:
            y = x.sub_str_in_explicit(v)
        assert get_output().strip() == v

    def test_sub_str_in_implicit(self, fortran_output):
        v = "123456789"

        with fortran_output() as get_output:
            y = x.sub_str_in_implicit(v)
        assert get_output().strip() == v

    def test_sub_str_multi(self, fortran_output):
        v = 5
        u = "123456789"
        w = 4

        with fortran_output() as get_output:
            y = x.sub_str_multi(v, u, w)
        assert get_output().strip() == str(v + w) + " " + u

    def test_sub_str_p(self, fortran_output):
        with fortran_output() as get_output:
            y = x.sub_str_p("abcdef")
        assert y.args["zzz"] == "xyzxyz"
        assert get_output().strip() == "abcdef"

    def test_func_ret_str(self):
        y = x.func_ret_str("abcde")
        assert y.result == "Abcde"

    @pytest.mark.skip
    # We need to call a func on the argument before passing it to func_str_int_len
    def test_func_str_int_len(self):
        y = x.func_str_int_len(10)

        assert y.result == "10"

    def test_str_alloc(self):
        assert x.str_alloc is None  # Empty at start

        x.str_alloc = "abcdefghijklmnop"
        assert x.str_alloc == "abcdefghijklmnop"
        y = x.check_str_alloc(1)
        assert y.result

        x.str_alloc = "12345678        "  # Need to empty the space afterwards
        assert x.str_alloc == "12345678        "
        y = x.check_str_alloc(2)
        assert not y.result

    def test_str_alloc_sub(self):
        z = None
        y = x.sub_str_alloc(z)
        assert y.args["x_alloc"] == "abcdef"

        y2 = x.sub_str_alloc2(None)
        assert y2.args["x"] == "zxcvbnm"

    def test_str_alloc_sub_realloc(self):
        y2 = x.sub_str_alloc2("qwerty")
        assert y2.args["x"] == "asdfghjkl"

    def test_str_array_type_chceck(self):
        with pytest.raises(TypeError) as cm:
            x.a_str_exp_1d = np.zeros(5, dtype=np.str_)

    def test_str_func_inout_str(self, fortran_output):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        with fortran_output() as get_output:
            res = x.str_array_inout(z)

        assert (
            get_output().strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"
        )

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

    def test_str_func_inout_str2(self, fortran_output):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        with fortran_output() as get_output:
            res = x.str_array_inout2(z, 5)

        assert (
            get_output().strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"
        )

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_str_func_inout_str3(self, fortran_output):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        with fortran_output() as get_output:
            res = x.str_array_inout3(z)

        assert (
            get_output().strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"
        )

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_str_func_inout_alloc(self):
        z = None

        res = x.str_array_allocate(z)

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_str_func_inout_str4(self, fortran_output):
        z = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        with fortran_output() as get_output:
            res = x.str_array_inout4(z)

        assert (
            get_output().strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"
        )

        z2 = np.array(
            ["zzzzzzzzzz", "yyyyyyyyyy", "qqqqqqqqqq", "wwwwwwwwww", "xxxxxxxxxx"],
            dtype="S10",
        )

        assert np.all(res.args["x"] == z2)

    def test_str_array_param(self):
        assert np.all(x.a_str_p_1d == np.array(["aa", "bb", "cc"], dtype="S2"))

    def test_check_a_str_exp_1d(self, fortran_output):
        x.a_str_exp_1d = np.array(
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd", "eeeeeeeeee"],
            dtype="S10",
        )

        with fortran_output() as get_output:
            res = x.check_a_str_exp_1d()

        assert (
            get_output().strip() == "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"
        )

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

    def test_check_assumed_shape_str(self):
        y = np.array(["a/b/c/d/e/f/g"], dtype="S13")
        res = x.check_assumed_shape_str_value(y)
        assert res.result

    @pytest.mark.skipIfWindows(
        reason="Soemtimes casues heap crashes on Windows, needs investigation"
    )
    def test_check_str_opt(self):
        res = x.check_str_opt(None, 0)
        assert res.result == 3

        res = x.check_str_opt("123456", 6)
        assert res.result == 1

        res = x.check_str_opt("abcedfg", 7)
        assert res.result == 2

        res = x.check_str_opt("abcd", 4)
        assert res.result == 4
