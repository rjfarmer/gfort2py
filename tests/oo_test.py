# SPDX-License-Identifier: GPL-2.0+

import os

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
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

    def test_proc_pass_alternate_location(self):
        y = x.p_proc.proc_pass(3)
        y2 = x.p_proc.proc_pass2(3)

        assert y.result == y2.result

    def test_proc_pass_updates_underlying_object(self):
        x.sub_set_p_proc(2)

        out = x.p_proc.proc_pass(9)

        assert out.result is None
        assert x.func_get_p_proc().result == 45

    def test_proc_pass_rejects_explicit_self(self):
        with pytest.raises(ValueError):
            x.p_proc.proc_pass(x.p_proc, 9)

    def test_typebound_methods_available_on_extended_type(self):
        x.sub_set_p_proc_extend(4, 2.5)

        y = x.p_proc_extend.proc_no_pass(4)

        assert y.result == 20
        x.p_proc_extend.proc_pass(6)
        vals = x.sub_get_p_proc_extend(0, 0.0).args
        assert vals["ai"] == 30

    def test_sub_set_and_get_p_proc_roundtrip(self):
        x.sub_set_p_proc(23)
        y = x.func_get_p_proc()

        assert y.result == 23

    def test_p_proc_arr_module_var_and_getter_roundtrip(self):
        vals = np.array([7, 8, 9], dtype=np.int32)
        x.sub_set_p_proc_arr(vals)

        out = x.sub_get_p_proc_arr(np.zeros(3, dtype=np.int32))
        assert np.array_equal(out.args["vals"], vals)

        # Also exercise direct module variable access for p_proc_arr.
        assert x.p_proc_arr[0]["a_int"] == 7
        assert x.p_proc_arr[1]["a_int"] == 8
        assert x.p_proc_arr[2]["a_int"] == 9

    def test_func_return_obj(self):
        y = x.func_return_obj(41)

        assert y.result["a_int"] == 41

    def test_func_return_obj_array(self):
        y = x.func_return_obj_array()

        assert y.result[0]["a_int"] == 11
        assert y.result[1]["a_int"] == 22

    def test_func_return_obj_array_alloc(self):
        y = x.func_return_obj_array_alloc(4)

        assert y.result[0]["a_int"] == 101
        assert y.result[3]["a_int"] == 104

    def test_sub_class_set_get_on_module_var(self):
        y = x.sub_class_set_get(x.p_proc, 17, 0)

        assert y.args["y"] == 17
        assert x.func_get_p_proc().result == 17

    def test_func_return_class_base_and_extended(self):
        base = x.func_return_class(5, False).result
        ext = x.func_return_class(7, True).result

        # Returned polymorphic class descriptors should be usable as CLASS dummies.
        out_base = x.sub_class_set_get(base, 33, 0)
        out_ext = x.sub_class_set_get(ext, 44, 0)

        assert out_base.args["y"] == 33
        assert out_ext.args["y"] == 44

    def test_class_array_procedures_reject_plain_python_placeholders(self):
        # These procedures require a class wrapper input, not a plain list.
        with pytest.raises(TypeError):
            x.sub_fill_class_array([], 5)

        with pytest.raises(TypeError):
            x.func_check_class_array([], 5)

        with pytest.raises(TypeError):
            x.sub_make_class_array([], 3, False)
