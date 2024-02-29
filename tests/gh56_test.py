# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/gh56.{gf.lib_ext()}"
MOD = "./tests/gh56.mod"

x = gf.fFort(SO, MOD)


class Testgh56Methods:
    def test_gh56(self, capfd):
        y1 = x.get_array(10).result

        y2 = x.get_array(5).result

        assert len(y1["d"]) == 10
        assert len(y2["d"]) == 5

        assert np.array_equal(
            y1["d"],
            np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32
            ),
        )

        assert np.array_equal(
            y2["d"], np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        )
