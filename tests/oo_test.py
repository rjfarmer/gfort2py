# SPDX-License-Identifier: GPL-2.0+

import os

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import pytest

import gfort2py as gf

from .conftest import build_paths

SO, MOD = build_paths("oo", "oo")

x = gf.fFort(SO, MOD)


class TestOOMethods:
    def test_proc_no_pass_matches_module_function(self):
        y = x.p_proc.proc_no_pass(3)
        y2 = x.func_dt_no_pass(3)

        assert y.result == y2.result

    def test_proc_pass_polymorphic_not_supported_yet(self):
        with pytest.raises(NotImplementedError):
            x.p_proc.proc_pass(9)

    def test_proc_pass_rejects_explicit_self(self):
        with pytest.raises((ValueError, NotImplementedError)):
            x.p_proc.proc_pass(x.p_proc, 9)

    def test_typebound_methods_available_on_extended_type(self):
        x.sub_set_p_proc_extend(4, 2.5)

        y = x.p_proc_extend.proc_no_pass(4)

        assert y.result == 20
        with pytest.raises(NotImplementedError):
            x.p_proc_extend.proc_pass(6)
