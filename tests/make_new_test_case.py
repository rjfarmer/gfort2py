#!/usr/bin/env python
#  SPDX-License-Identifier: GPL-2.0+
import sys


def make_test_case(name):

    with open(f"{name}_test.py", "w") as f:
        str = f"""
# SPDX-License-Identifier: GPL-2.0+

import os, sys
import ctypes
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/{name}.{{gf.lib_ext()}}"
MOD = "./tests/{name}.mod"

x = gf.fFort(SO,MOD)


class Test{name}Methods:
    pass

        """

        print(str, file=f)

    with open(f"{name}.f90", "w") as f:
        str = f"""   

! SPDX-License-Identifier: GPL-2.0+

module {name}

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	contains



end module {name}

        """
        print(str, file=f)


if __name__ == "__main__":
    for i in sys.argv[1:]:
        make_test_case(i)
