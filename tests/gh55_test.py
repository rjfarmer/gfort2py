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

SO, MOD = build_paths("gh55", "gh55")

x = gf.fFort(SO, MOD)


class Testgh55Methods:
    def test_func_str_return_array(self):

        size = 3
        res = x.return_char(size)

        return_str = res.result

        assert return_str.shape == ((2**size) - 1,)
        assert return_str[-1] == b"abcdefghil"
