# SPDX-License-Identifier: GPL-2.0+
# This file is auto generated do not edit by hand

import ctypes
import os
import sys

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf

from .conftest import build_paths

SO, MOD = build_paths("t_vars_modules", "vars_modules")

x = gf.fFort(SO, MOD)


class Test_vars_modules:
    def test_check_ints(self):
        assert x.int_i1_0 == -1

        x.int_i1_0 = -2

        assert x.int_i1_0 == -2

        result = x.check_int_i1_0()

        assert result.result

        assert x.int_i1_1 == 0

        x.int_i1_1 = 0

        assert x.int_i1_1 == 0

        result = x.check_int_i1_1()

        assert result.result

        assert x.int_i1_2 == 1

        x.int_i1_2 = 2

        assert x.int_i1_2 == 2

        result = x.check_int_i1_2()

        assert result.result

        assert x.int_i2_0 == -1

        x.int_i2_0 = -2

        assert x.int_i2_0 == -2

        result = x.check_int_i2_0()

        assert result.result

        assert x.int_i2_1 == 0

        x.int_i2_1 = 0

        assert x.int_i2_1 == 0

        result = x.check_int_i2_1()

        assert result.result

        assert x.int_i2_2 == 1

        x.int_i2_2 = 2

        assert x.int_i2_2 == 2

        result = x.check_int_i2_2()

        assert result.result

        assert x.int_i4_0 == -1

        x.int_i4_0 = -2

        assert x.int_i4_0 == -2

        result = x.check_int_i4_0()

        assert result.result

        assert x.int_i4_1 == 0

        x.int_i4_1 = 0

        assert x.int_i4_1 == 0

        result = x.check_int_i4_1()

        assert result.result

        assert x.int_i4_2 == 1

        x.int_i4_2 = 2

        assert x.int_i4_2 == 2

        result = x.check_int_i4_2()

        assert result.result

        assert x.int_i8_0 == -1

        x.int_i8_0 = -2

        assert x.int_i8_0 == -2

        result = x.check_int_i8_0()

        assert result.result

        assert x.int_i8_1 == 0

        x.int_i8_1 = 0

        assert x.int_i8_1 == 0

        result = x.check_int_i8_1()

        assert result.result

        assert x.int_i8_2 == 1

        x.int_i8_2 = 2

        assert x.int_i8_2 == 2

        result = x.check_int_i8_2()

        assert result.result

    def test_check_reals(self):
        assert x.real_r4_0 == -3.140000104904175

        x.real_r4_0 = -6.28000020980835

        assert x.real_r4_0 == -6.28000020980835

        result = x.check_real_r4_0()

        assert result.result

        assert x.real_r4_1 == 0

        x.real_r4_1 = 0

        assert x.real_r4_1 == 0

        result = x.check_real_r4_1()

        assert result.result

        assert x.real_r4_2 == 3.140000104904175

        x.real_r4_2 = 6.28000020980835

        assert x.real_r4_2 == 6.28000020980835

        result = x.check_real_r4_2()

        assert result.result

        assert x.real_r8_0 == -3.140000104904175

        x.real_r8_0 = -6.28000020980835

        assert x.real_r8_0 == -6.28000020980835

        result = x.check_real_r8_0()

        assert result.result

        assert x.real_r8_1 == 0

        x.real_r8_1 = 0

        assert x.real_r8_1 == 0

        result = x.check_real_r8_1()

        assert result.result

        assert x.real_r8_2 == 3.140000104904175

        x.real_r8_2 = 6.28000020980835

        assert x.real_r8_2 == 6.28000020980835

        result = x.check_real_r8_2()

        assert result.result
