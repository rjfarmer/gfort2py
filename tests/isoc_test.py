# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/isoc.{gf.lib_ext()}"
MOD = "./tests/isoc.mod"

x = gf.fFort(SO, MOD)


class TestISOC:
    def test_c_name(self):
        # This should fail as we access via the
        # Fortran not c name
        with pytest.raises(KeyError) as e:
            x.a_int_binc_c = 1

    def test_f_name(self):
        x.a_int = 5
        assert x.a_int == 5

    def test_function(self):
        res = x.func_bind_c(1)

        assert res.result == 2
