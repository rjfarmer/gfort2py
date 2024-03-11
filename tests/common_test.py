# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/common.{gf.lib_ext()}"
MOD = "./tests/com.mod"

x = gf.fFort(SO, MOD)


class TestCommonBlocks:
    def test_set_values(self):
        with pytest.raises(AttributeError) as cm:
            x.x_int = 1
            assert x.x_int == 1

    def test_get_comm(self):
        assert len(x._module.common)
