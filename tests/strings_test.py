# SPDX-License-Identifier: GPL-2.0+

import os, sys

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

import subprocess
import numpy.testing as np_test

from contextlib import contextmanager
from io import StringIO
from io import BytesIO

# Decreases recursion depth to make debugging easier
# sys.setrecursionlimit(10)

SO = "./tests/strings.so"
MOD = "./tests/strings.mod"

x = gf.fFort(SO, MOD)

@pytest.mark.skip
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
        self.assertEqual(y.res, "Abcde")

    @pytest.mark.skip("Skipping")
    # We need to call a func on the argument before passing it to func_str_int_len
    def test_func_str_int_len(self):

        y = x.func_str_int_len(10)

        self.assertEqual(out, "10")

    @pytest.mark.skip("Skipping")
    def test_str_alloc(self):
        self.assertEqual(x.str_alloc, "")  # Empty at start

        x.str_alloc = "abcdefghijklmnop"
        self.assertEqual(x.str_alloc, "abcdefghijklmnop")
        y = x.check_str_alloc(1)
        self.assertEqual(y.res, True)

        x.str_alloc = "12345678        "  # Need to empty the space afterwards
        self.assertEqual(x.str_alloc, "12345678        ")
        y = x.check_str_alloc(2)
        self.assertEqual(y.res, True)
