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

SO, MOD = build_paths("gh56", "gh56")

x = gf.fFort(SO, MOD)


@pytest.mark.skip
class Testgh56Methods:
    @pytest.mark.skipIfWindows
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
