# SPDX-License-Identifier: GPL-2.0+

import os, sys

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

import subprocess
import numpy.testing as np_test

from contextlib import contextmanager
from io import StringIO
from io import BytesIO

# Decreases recursion depth to make debugging easier
# sys.setrecursionlimit(10)

SO = "./tests/.so"
MOD = "./tests/.mod"

# x=gf.fFort(SO,MOD,rerun=True)


@contextmanager
def captured_output():
    """
    For use when we need to grab the stdout/stderr from fortran (but only in testing)
    Use as:
    with captured_output() as (out,err):
        func()
    output=out.getvalue().strip()
    error=err.getvalue().strip()
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestStringMethods:
    pass
