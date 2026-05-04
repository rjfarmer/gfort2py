# SPDX-License-Identifier: GPL-2.0+

import ctypes
import os
import sys
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

SO = f"./tests/build/gh55.{gf.lib_ext()}"
MOD = "./tests/build/gh55.mod"

x = gf.fFort(SO, MOD)


@pytest.mark.skip("Currently broken needs array descriptor for strings")
class Testgh55Methods:
    def test_func_str_return_array(self):

        size = 3
        res = x.return_char(size)

        return_str = res.result

        assert np.size(np.shape(return_str)) == ((2**size) - 1)
        assert return_str[-1] == "abcdefghij"
