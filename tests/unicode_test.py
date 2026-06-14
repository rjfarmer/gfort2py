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

SO, MOD = build_paths("unicode", "unicode")

x = gf.fFort(SO, MOD)


class TestUnicodeMethods:
    def test_set(self):
        assert x.uni_set.strip() == "😀😎😩"

    def test_set_unicode_string(self):
        x.sub_set_uni_set("漢字Ω")
        assert x.uni_set.strip() == "漢字Ω"

    def test_param(self):
        # TODO: Overload fParam to handle this
        assert x.uni_param == "😀😎😩"

    def test_unicode_argument_roundtrip(self):
        y = x.sub_uni_echo("🌍🚀")
        assert y.args["y"].strip() == "🌍🚀"

    def test_unicode_function_return(self):
        y = x.func_uni_ret()
        assert y.result.strip() == "🌍🚀"

    def test_unicode_array_get(self):
        assert np.all(
            x.uni_arr == np.array(["😀😀😀", "😎😎😎", "😩😩😩"], dtype=np.str_)
        )

    def test_unicode_array_set_and_roundtrip(self):
        values = np.array(["🚀🚀🚀", "🌍🌍🌍", "✨✨✨"], dtype=np.str_)
        x.uni_arr = values
        assert np.all(x.uni_arr == values)

    def test_unicode_array_argument_roundtrip(self):
        values = np.array(["🚀🚀🚀", "🌍🌍🌍", "✨✨✨"], dtype=np.str_)
        y = x.sub_uni_arr_inout(values)
        assert np.all(y.args["x"] == np.array(["😀😎😩", "漢字", "aΩβ"], dtype=np.str_))

    @pytest.mark.xfail(
        reason="Unicode assumed-shape character arrays are not stable yet",
        run=False,
    )
    def test_unicode_assumed_shape_array_argument(self):
        values = np.array(["🚀🚀🚀", "🌍🌍🌍", "✨✨✨"], dtype=np.str_)
        y = x.func_uni_assumed_shape_ok(values)
        assert y.result

    def test_unicode_allocatable_array(self):
        x.alloc_uni_alloc_arr()
        assert np.all(
            x.uni_alloc_arr == np.array(["😀😎😩", "漢字Ω", "aΩβ"], dtype=np.str_)
        )

        values = np.array(["🚀🚀🚀", "🌍🌍🌍", "✨✨✨"], dtype=np.str_)
        x.uni_alloc_arr = values
        assert np.all(x.uni_alloc_arr == values)

    @pytest.mark.xfail(
        reason="Unicode assumed-rank character arrays are not stable yet",
        run=False,
    )
    def test_unicode_assumed_rank_array_argument_1d(self):
        values = np.array(["🚀🚀🚀", "🌍🌍🌍", "✨✨✨"], dtype=np.str_)
        y = x.func_uni_assumed_rank_ok(values)
        assert y.result

    @pytest.mark.xfail(
        reason="Unicode assumed-rank character arrays are not stable yet",
        run=False,
    )
    def test_unicode_assumed_rank_array_argument_2d(self):
        values = np.array([["🚀🚀🚀", "✨✨✨"], ["🌍🌍🌍", "漢字Ω"]], dtype=np.str_)
        y = x.func_uni_assumed_rank_ok(values)
        assert y.result

    def test_unicode_allocatable_return(self):
        y = x.func_return_alloc_unicode()
        assert y.result.strip() == "🚀🚀🚀"
