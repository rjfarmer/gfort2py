# SPDX-License-Identifier: GPL-2.0+
# This file is auto generated do not edit by hand

import os, sys
import ctypes

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/t_params_modules.{gf.lib_ext()}"
MOD = "./tests/params_modules.mod"

x = gf.fFort(SO, MOD)


class Test_params_modules:
    def test_check_ints(self):
        assert x.int_i1_0 == -1

        with pytest.raises(AttributeError) as cm:
            x.int_i1_0 = -99

        assert x.int_i1_1 == 0

        with pytest.raises(AttributeError) as cm:
            x.int_i1_1 = -99

        assert x.int_i1_2 == 1

        with pytest.raises(AttributeError) as cm:
            x.int_i1_2 = -99

        assert x.int_i2_0 == -1

        with pytest.raises(AttributeError) as cm:
            x.int_i2_0 = -99

        assert x.int_i2_1 == 0

        with pytest.raises(AttributeError) as cm:
            x.int_i2_1 = -99

        assert x.int_i2_2 == 1

        with pytest.raises(AttributeError) as cm:
            x.int_i2_2 = -99

        assert x.int_i4_0 == -1

        with pytest.raises(AttributeError) as cm:
            x.int_i4_0 = -99

        assert x.int_i4_1 == 0

        with pytest.raises(AttributeError) as cm:
            x.int_i4_1 = -99

        assert x.int_i4_2 == 1

        with pytest.raises(AttributeError) as cm:
            x.int_i4_2 = -99

        assert x.int_i8_0 == -1

        with pytest.raises(AttributeError) as cm:
            x.int_i8_0 = -99

        assert x.int_i8_1 == 0

        with pytest.raises(AttributeError) as cm:
            x.int_i8_1 = -99

        assert x.int_i8_2 == 1

        with pytest.raises(AttributeError) as cm:
            x.int_i8_2 = -99

    def test_check_reals(self):
        assert x.real_r4_0 == -3.140000104904175

        with pytest.raises(AttributeError) as cm:
            x.real_r4_0 = -99.9

        assert x.real_r4_1 == 0

        with pytest.raises(AttributeError) as cm:
            x.real_r4_1 = -99.9

        assert x.real_r4_2 == 3.140000104904175

        with pytest.raises(AttributeError) as cm:
            x.real_r4_2 = -99.9

        assert x.real_r8_0 == -3.140000104904175

        with pytest.raises(AttributeError) as cm:
            x.real_r8_0 = -99.9

        assert x.real_r8_1 == 0

        with pytest.raises(AttributeError) as cm:
            x.real_r8_1 = -99.9

        assert x.real_r8_2 == 3.140000104904175

        with pytest.raises(AttributeError) as cm:
            x.real_r8_2 = -99.9

    def test_check_logicals(self):
        assert not x.logicals_0

        with pytest.raises(AttributeError) as cm:
            x.logicals_0 = True

        assert x.logicals_1

        with pytest.raises(AttributeError) as cm:
            x.logicals_1 = False

    def test_check_ints_1d(self):
        assert np.allclose(x.int_i1_1d, np.array([-10, -1, 0, 1, 10]))

        with pytest.raises(AttributeError) as cm:
            x.int_i1_1d = np.array([1, 2, 3])

        assert np.allclose(x.int_i2_1d, np.array([-10, -1, 0, 1, 10]))

        with pytest.raises(AttributeError) as cm:
            x.int_i2_1d = np.array([1, 2, 3])

        assert np.allclose(x.int_i4_1d, np.array([-10, -1, 0, 1, 10]))

        with pytest.raises(AttributeError) as cm:
            x.int_i4_1d = np.array([1, 2, 3])

        assert np.allclose(x.int_i8_1d, np.array([-10, -1, 0, 1, 10]))

        with pytest.raises(AttributeError) as cm:
            x.int_i8_1d = np.array([1, 2, 3])

    def test_check_reals_1d(self):
        assert np.allclose(
            x.real_r4_1d, np.array([-3.140000104904175, 0.0, 3.140000104904175])
        )

        with pytest.raises(AttributeError) as cm:
            x.real_r4_1d = np.array([1, 2, 3])

        assert np.allclose(
            x.real_r8_1d, np.array([-3.140000104904175, 0.0, 3.140000104904175])
        )

        with pytest.raises(AttributeError) as cm:
            x.real_r8_1d = np.array([1, 2, 3])

    def test_check_logicals_1d(self):
        assert np.allclose(x.logicals_0_1d, np.array([True, False, True, False]))

    def test_check_ints_2d(self):
        assert np.allclose(
            x.int_i1_2d, np.array([-10, -1, 0, 1, 10, 50]).reshape(2, 3, order="F")
        )

        with pytest.raises(AttributeError) as cm:
            x.int_i1_2d = np.array([1, 2, 3])

        assert np.allclose(
            x.int_i2_2d, np.array([-10, -1, 0, 1, 10, 50]).reshape(2, 3, order="F")
        )

        with pytest.raises(AttributeError) as cm:
            x.int_i2_2d = np.array([1, 2, 3])

        assert np.allclose(
            x.int_i4_2d, np.array([-10, -1, 0, 1, 10, 50]).reshape(2, 3, order="F")
        )

        with pytest.raises(AttributeError) as cm:
            x.int_i4_2d = np.array([1, 2, 3])

        assert np.allclose(
            x.int_i8_2d, np.array([-10, -1, 0, 1, 10, 50]).reshape(2, 3, order="F")
        )

        with pytest.raises(AttributeError) as cm:
            x.int_i8_2d = np.array([1, 2, 3])

    def test_check_reals_2d(self):
        assert np.allclose(
            x.real_r4_2d,
            np.array(
                [
                    -6.28000020980835,
                    -3.140000104904175,
                    0.0,
                    1.1111,
                    3.140000104904175,
                    6.28000020980835,
                ]
            ).reshape(2, 3, order="F"),
        )

        with pytest.raises(AttributeError) as cm:
            x.real_r4_2d = np.array([1, 2, 3])

        assert np.allclose(
            x.real_r8_2d,
            np.array(
                [
                    -6.28000020980835,
                    -3.140000104904175,
                    0.0,
                    1.1111,
                    3.140000104904175,
                    6.28000020980835,
                ]
            ).reshape(2, 3, order="F"),
        )

        with pytest.raises(AttributeError) as cm:
            x.real_r8_2d = np.array([1, 2, 3])

    def test_check_logicals_2d(self):
        assert np.allclose(
            x.logicals_0_2d,
            np.array([True, False, True, False, True, False]).reshape(2, 3, order="F"),
        )

    def test_check_cmplx(self):
        assert x.complex_r4_0 == complex(-3.140000104904175, -3.140000104904175)

        with pytest.raises(AttributeError) as cm:
            x.complex_r4_0 = -99.9

        assert x.complex_r4_1 == complex(0.0, 0.0)

        with pytest.raises(AttributeError) as cm:
            x.complex_r4_1 = -99.9

        assert x.complex_r4_2 == complex(3.140000104904175, 3.140000104904175)

        with pytest.raises(AttributeError) as cm:
            x.complex_r4_2 = -99.9

        assert x.complex_r8_0 == complex(-3.140000104904175, -3.140000104904175)

        with pytest.raises(AttributeError) as cm:
            x.complex_r8_0 = -99.9

        assert x.complex_r8_1 == complex(0.0, 0.0)

        with pytest.raises(AttributeError) as cm:
            x.complex_r8_1 = -99.9

        assert x.complex_r8_2 == complex(3.140000104904175, 3.140000104904175)

        with pytest.raises(AttributeError) as cm:
            x.complex_r8_2 = -99.9

    def test_check_cmplx_1d(self):
        assert np.allclose(
            x.complex_r4_1d,
            np.array(
                [
                    complex(-3.140000104904175, -3.140000104904175),
                    complex(0.0, 0.0),
                    complex(3.140000104904175, 3.140000104904175),
                ]
            ),
        )

        with pytest.raises(AttributeError) as cm:
            x.complex_r4_1d = np.array([1, 2, 3])

        assert np.allclose(
            x.complex_r8_1d,
            np.array(
                [
                    complex(-3.140000104904175, -3.140000104904175),
                    complex(0.0, 0.0),
                    complex(3.140000104904175, 3.140000104904175),
                ]
            ),
        )

        with pytest.raises(AttributeError) as cm:
            x.complex_r8_1d = np.array([1, 2, 3])

    def test_check_cmplx_2d(self):
        assert np.allclose(
            x.complex_r4_2d,
            np.array(
                [
                    complex(-6.28000020980835, -6.28000020980835),
                    complex(-3.140000104904175, -3.140000104904175),
                    complex(0.0, 0.0),
                    complex(0.0, -1.0),
                    complex(3.140000104904175, 3.140000104904175),
                    complex(-6.28000020980835, -6.28000020980835),
                ]
            ).reshape(2, 3, order="F"),
        )

        with pytest.raises(AttributeError) as cm:
            x.complex_r4_2d = np.array([1, 2, 3])

        assert np.allclose(
            x.complex_r8_2d,
            np.array(
                [
                    complex(-6.28000020980835, -6.28000020980835),
                    complex(-3.140000104904175, -3.140000104904175),
                    complex(0.0, 0.0),
                    complex(0.0, -1.0),
                    complex(3.140000104904175, 3.140000104904175),
                    complex(-6.28000020980835, -6.28000020980835),
                ]
            ).reshape(2, 3, order="F"),
        )

        with pytest.raises(AttributeError) as cm:
            x.complex_r8_2d = np.array([1, 2, 3])
