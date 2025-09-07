# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/params2.{gf.lib_ext()}"
MOD = "./tests/params2.mod"

x = gf.fFort(SO, MOD)


class TestSParams2:
    def test_positive(self):
        assert a2rad == 0.017453292519943295
        assert amu == 1.6605390671738466e-24
        assert au == 14959787070000.0
        assert avo == 6.02214076e23
        assert boltz_sigma == 5.670374419184426e-05
        assert boltzm == 1.380649e-16
        assert cgas == 83144626.1815324
        assert clight == 29979245800.0
        assert crad == 7.56573325028e-15
        assert dayyer == 365.25
        assert dp == 8
        assert eulercon == 0.5772156649015329
        assert eulernum == 2.718281828459045
        assert ev2erg == 1.602176634e-12
        assert fine == 0.0072973525693
        assert five_thirds == 1.6666666666666667
        assert four_13 == 1.5874010519681994
        assert four_thirds == 1.3333333333333333
        assert four_thirds_pi == 4.1887902047863905
        assert hbar == 1.0545718176461565e-27
        assert hion == 13.605693122994
        assert iln10 == 0.43429448190325187
        assert kerg == 1.380649e-16
        assert kev == 8.617333262145177e-05
        assert ln10 == 2.3025850929940455
        assert ln2 == 0.6931471805599453
        assert ln3 == 1.0986122886681096
        assert ln4pi3 == 1.432411958301181
        assert lnpi == 1.1447298858494002
        assert ly == 9.4607304725808e17
        assert ma2rad == -0.017453292519943295
        assert mamu == -1.6605390671738466e-24
        assert mau == -14959787070000.0
        assert mavo == -6.02214076e23
        assert mboltz_sigma == -5.670374419184426e-05
        assert mboltzm == -1.380649e-16
        assert mcgas == -83144626.1815324
        assert mclight == -29979245800.0
        assert mcrad == -7.56573325028e-15
        assert mdayyer == -365.25
        assert me == 9.1093837015e-28
        assert meulercon == -0.5772156649015329
        assert meulernum == -2.718281828459045
        assert mev2erg == -1.602176634e-12
        assert mev2gr == 1.7826619216278976e-27
        assert mev_amu == 9.648533212331003e17
        assert mev_to_ergs == 1.602176634e-06

    def test_negative(self):
        assert mfine == -0.0072973525693
        assert mfive_thirds == -1.6666666666666667
        assert mfour_13 == -1.5874010519681994
        assert mfour_thirds == -1.3333333333333333
        assert mfour_thirds_pi == -4.1887902047863905
        assert mhbar == -1.0545718176461565e-27
        assert mhion == -13.605693122994
        assert miln10 == -0.43429448190325187
        assert mkerg == -1.380649e-16
        assert mkev == -8.617333262145177e-05
        assert mln10 == -2.3025850929940455
        assert mln2 == -0.6931471805599453
        assert mln3 == -1.0986122886681096
        assert mln4pi3 == -1.432411958301181
        assert mlnpi == -1.1447298858494002
        assert mly == -9.4607304725808e17
        assert mme == -9.1093837015e-28
        assert mmev2gr == -1.7826619216278976e-27
        assert mmev_amu == -9.648533212331003e17
        assert mmev_to_ergs == -1.602176634e-06
        assert mmn == -1.67492749804e-24
        assert mmp == -1.67262192369e-24
        assert mn == 1.67492749804e-24
        assert mnum_neu_fam == -3.0
        assert mone_sixth == -0.16666666666666666
        assert mone_third == -0.3333333333333333
        assert mp == 1.67262192369e-24
        assert mpc == -3.0856775814913674e18
        assert mpi == -3.141592653589793
        assert mpi2 == -9.869604401089358
        assert mpi4 == -12.566370614359172
        assert mplanck_h == -6.62607015e-27
        assert mqconv == -9.648533212331002e17
        assert mqe == -4.803204712570263e-10
        assert mrad2a == -57.29577951308232
        assert mrbohr == -5.29177210903e-09
        assert msecday == -86400.0
        assert msecyer == -31557600.0
        assert msige == -6.6524587321e-25
        assert msqrt2 == -1.414213562373095
        assert msqrt_2_div_3 == -0.816496580927726
        assert mtwo_13 == -1.259921049894873
        assert mtwo_thirds == -0.6666666666666666
        assert mweinberg_theta == -0.2229
        assert num_neu_fam == 3.0
        assert one_sixth == 0.16666666666666666
        assert one_third == 0.3333333333333333
