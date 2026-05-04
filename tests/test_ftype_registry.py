# SPDX-License-Identifier: GPL-2.0+

import os

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import pytest

from gfort2py.types import (
    FortranSymbol,
    _ftype_registry,
    find_ftype,
    register_ftype,
)
from gfort2py.types.character import ftype_character
from gfort2py.types.complex import ftype_complex_4, ftype_complex_8, ftype_complex_16
from gfort2py.types.integer import (
    ftype_integer_1,
    ftype_integer_2,
    ftype_integer_4,
    ftype_integer_8,
)
from gfort2py.types.logical import (
    ftype_logical_1,
    ftype_logical_2,
    ftype_logical_4,
    ftype_logical_8,
)
from gfort2py.types.real import ftype_real_4, ftype_real_8, ftype_real_16
from gfort2py.types.unsigned import (
    ftype_unsigned_1,
    ftype_unsigned_2,
    ftype_unsigned_4,
    ftype_unsigned_8,
)


class TestRegistryContents:
    """All built-in (ftype, kind) pairs are present at import time."""

    @pytest.mark.parametrize(
        "ftype,kind,expected",
        [
            ("integer", 1, ftype_integer_1),
            ("integer", 2, ftype_integer_2),
            ("integer", 4, ftype_integer_4),
            ("integer", 8, ftype_integer_8),
            ("real", 4, ftype_real_4),
            ("real", 8, ftype_real_8),
            ("real", 16, ftype_real_16),
            ("complex", 4, ftype_complex_4),
            ("complex", 8, ftype_complex_8),
            ("complex", 16, ftype_complex_16),
            ("logical", 1, ftype_logical_1),
            ("logical", 2, ftype_logical_2),
            ("logical", 4, ftype_logical_4),
            ("logical", 8, ftype_logical_8),
            ("unsigned", 1, ftype_unsigned_1),
            ("unsigned", 2, ftype_unsigned_2),
            ("unsigned", 4, ftype_unsigned_4),
            ("unsigned", 8, ftype_unsigned_8),
            ("character", None, ftype_character),
        ],
    )
    def test_builtin_entry(self, ftype, kind, expected):
        assert _ftype_registry[(ftype, kind)] is expected


class TestFindFtype:
    """find_ftype resolves types correctly and raises clearly on misses."""

    @pytest.mark.parametrize(
        "ftype,kind,expected",
        [
            ("integer", 4, ftype_integer_4),
            ("integer", 8, ftype_integer_8),
            ("real", 4, ftype_real_4),
            ("real", 8, ftype_real_8),
            ("complex", 8, ftype_complex_8),
            ("logical", 4, ftype_logical_4),
            ("unsigned", 4, ftype_unsigned_4),
        ],
    )
    def test_lookup_returns_correct_class(self, ftype, kind, expected):
        assert find_ftype(ftype, kind) is expected

    @pytest.mark.parametrize("kind", [1, 1000])
    def test_character_any_kind_returns_ftype_character(self, kind):
        # character is always matched by type alone regardless of kind value
        assert find_ftype("character", kind) is ftype_character

    def test_unknown_type_raises_type_error(self):
        with pytest.raises(TypeError, match="ftype='bogus'"):
            find_ftype("bogus", 4)

    def test_unknown_kind_raises_type_error(self):
        with pytest.raises(TypeError, match="kind=99"):
            find_ftype("integer", 99)


class TestRegisterFtype:
    """register_ftype adds and overrides entries in the registry."""

    def test_register_new_kind(self):
        register_ftype("integer", 999, ftype_integer_4)
        try:
            assert find_ftype("integer", 999) is ftype_integer_4
        finally:
            del _ftype_registry[("integer", 999)]

    def test_register_overrides_existing(self):
        original = _ftype_registry[("integer", 4)]
        register_ftype("integer", 4, ftype_integer_8)
        try:
            assert find_ftype("integer", 4) is ftype_integer_8
        finally:
            register_ftype("integer", 4, original)

    def test_register_new_ftype(self):
        register_ftype("custom", 4, ftype_integer_4)
        try:
            assert find_ftype("custom", 4) is ftype_integer_4
        finally:
            del _ftype_registry[("custom", 4)]


class TestFortranSymbolProtocol:
    """fParam satisfies FortranSymbol; plain objects without the interface do not."""

    def test_fparam_satisfies_protocol(self):
        import gfModParser as gf

        import gfort2py as gft
        from gfort2py.types.parameters import fParam

        m = gf.Module("./tests/build/basic.mod")
        params = gf.Parameters(m)
        key = list(params.keys())[0]
        param = fParam(m[key])
        assert isinstance(param, FortranSymbol)

    def test_object_missing_ftype_does_not_satisfy_protocol(self):
        class Incomplete:
            @property
            def value(self):
                return 0

            @property
            def kind(self):
                return 4

            @property
            def module(self):
                return "m"

        assert not isinstance(Incomplete(), FortranSymbol)

    def test_fully_implemented_object_satisfies_protocol(self):
        class Full:
            @property
            def value(self):
                return 0

            @property
            def ftype(self):
                return "integer"

            @property
            def kind(self):
                return 4

            @property
            def module(self):
                return "m"

        assert isinstance(Full(), FortranSymbol)
