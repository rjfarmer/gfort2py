# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = "./tests/.so"
MOD = "./tests/.mod"

# x=gf.fFort(SO,MOD)


class TestStringMethods:
    def assertEqual(self, x, y):
        assert x == y
