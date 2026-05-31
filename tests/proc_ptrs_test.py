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

SO, MOD = build_paths("proc_ptrs", "proc_ptrs")

x = gf.fFort(SO, MOD)


class TestProcPtrsMethods:

    def test_proc_ptr_ffunc(self):
        x.sub_null_proc_ptr()
        with pytest.raises(AttributeError) as cm:
            y = x.p_func_func_run_ptr(1)

        x.p_func_func_run_ptr = x.func_func_run
        y = x.p_func_func_run_ptr(1)
        assert y.result == 10
        y = x.p_func_func_run_ptr(2)
        assert y.result == 20

        y = x.func_proc_ptr(5)
        y2 = x.p_func_func_run_ptr(5)
        assert y.result == y2.result

    def test_proc_ptr_ffunc2(self):
        x.sub_null_proc_ptr()
        y = x.p_func_func_run_ptr2(1)  # Already set

        x.p_func_func_run_ptr2 = x.func_func_run
        y = x.p_func_func_run_ptr2(10)
        assert y.result == 100

    def test_proc_pointer_bind_from_proc_pointer(self):
        x.p_func_func_run_ptr2 = x.func_func_run2
        x.p_func_func_run_ptr = x.p_func_func_run_ptr2

        y = x.p_func_func_run_ptr(7)
        assert y.result == 14

    def test_proc_update(self):
        x.sub_null_proc_ptr()
        x.p_func_func_run_ptr = x.func_func_run
        y = x.p_func_func_run_ptr(1)
        assert y.result == 10

        x.p_func_func_run_ptr = x.func_func_run2
        y = x.p_func_func_run_ptr(1)
        assert y.result == 2

    @pytest.mark.skipif(gf.utils.is_big_endian(), reason="Skip on big endian systems")
    def test_proc_func_arg(self):
        y = x.func_func_arg_dp(5, x.func_real)
        assert y.result == 500

        y = x.func_func_arg(x.func_func_run)
        assert y.result == 10

        y = x.func_func_arg(func=x.func_func_run)
        assert y.result == 10

    def test_proc_func_arg_compile(self):
        fstr = """
                integer function test(x)
                    integer :: x

                    test = 3*x
                end function test

                """

        f = gf.compile(fstr)

        y = x.func_func_arg(f.test)
        assert y.result == 3

    def test_proc_proc_func_arg(self):
        x.sub_null_proc_ptr()
        x.p_func_func_run_ptr = x.func_func_run

        y = x.proc_proc_func_arg(x.p_func_func_run_ptr)
        assert y.result == 90

    def test_proc_pointer_proc_pointer_dummy(self):
        x.sub_null_proc_ptr()
        x.p_func_func_run_ptr = x.func_func_run

        y = x.p_proc_proc_func_arg_ptr(x.p_func_func_run_ptr)
        assert y.result == 90
