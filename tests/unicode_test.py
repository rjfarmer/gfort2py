# SPDX-License-Identifier: GPL-2.0+

import ctypes
import os
import sys
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

SO = f"./tests/build/unicode.{gf.lib_ext()}"
MOD = "./tests/build/unicode.mod"

x = gf.fFort(SO, MOD)


class TestUnicodeMethods:
    def test_set(self):
        assert x.uni_set.strip() == "😀😎😩"

    def test_param(self):
        # TODO: Overload fParam to handle this
        assert x.uni_param == "😀😎😩"
