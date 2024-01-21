import os, sys
import ctypes
from pprint import pprint
import platform

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest


class TestCompileMethods:
    def test_compile_nomod_str(self):
        fstr = """
            integer function myfunc(x,y)
                integer :: x,y
                myfunc = x+y
            end function myfunc
            """

        x = gf.compile(string=fstr)

        result = x.myfunc(1, 2)

        assert result.result == 3

    def test_compile_mod_str(self):
        fstr = """
            module abc
            contains
            integer function myfunc(x,y)
                integer :: x,y
                myfunc = x+y
            end function myfunc
            end module abc
            """

        x = gf.compile(string=fstr)

        result = x.myfunc(1, 2)

        assert result.result == 3

    def test_compile_nomod_file(self):
        x = gf.compile(file="tests/compile_nomod_test.f90")

        result = x.myfunc(1, 2)

        assert result.result == 3

    def test_compile_mod_file(self):
        x = gf.compile(file="tests/compile_mod_test.f90")

        result = x.myfunc(1, 2)

        assert result.result == 3
