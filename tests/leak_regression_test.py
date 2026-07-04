# SPDX-License-Identifier: GPL-2.0+
"""Regression test for the assumed-shape array-descriptor leak.

``ftype_assumed_shape.ctype`` used to build a fresh ``ctypes.Structure``
subclass on every call. CPython never frees ``Structure`` subclasses, so each
call to a procedure with an assumed-shape array argument leaked memory
(~16 KB/call, measured) and created a new ``Structure`` subclass. The
descriptor class is now cached per ``(ndims, is_64bit())``; this test asserts
that repeated calls do not create an unbounded number of ``Structure``
subclasses.
"""

import ctypes
import gc
import os

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np

import gfort2py as gf

from .conftest import build_paths

SO, MOD = build_paths("leak_regression", "leak_regression")

x = gf.fFort(SO, MOD)


def _count_ctypes_structure_subclasses():
    return sum(
        1
        for obj in gc.get_objects()
        if isinstance(obj, type) and issubclass(obj, ctypes.Structure)
    )


def test_assumed_shape_ctype_is_cached():
    a = np.ones((4, 4), dtype=np.float64, order="F")

    x.bump_assumed_shape_2d(a)  # warm up: first call builds + caches the descriptor
    gc.collect()
    before = _count_ctypes_structure_subclasses()

    for _ in range(200):
        x.bump_assumed_shape_2d(a)

    gc.collect()
    after = _count_ctypes_structure_subclasses()

    # Pre-fix this grew by ~200 (a new descriptor Structure subclass per call).
    assert after - before <= 2, (
        f"{after - before} new ctypes.Structure subclasses created over 200 "
        "calls; the assumed-shape array descriptor is not being cached"
    )
