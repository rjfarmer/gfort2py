import ctypes
import os
import platform
import sys
from pprint import pprint

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import pytest

import gfort2py as gf
from gfort2py.compilation.compile import Compile


class TestCompileMethods:
    def test_compile_library_filename_unique_per_instance(self):
        c1 = Compile("subroutine s\nend subroutine s", "same_name")
        c2 = Compile("subroutine s\nend subroutine s", "same_name")

        assert c1.library_filename != c2.library_filename

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

    def test_compile_nomod_file(self):
        x = gf.compile(file="tests/src/compile_nomod_test.f90")

        result = x.myfunc(1, 2)

        assert result.result == 3
