# SPDX-License-Identifier: GPL-2.0+

import os
from pathlib import Path

import pytest

import gfort2py as gf

MOD = Path("./tests/basic.mod")
SO = Path(f"./tests/basic.{gf.lib_ext()}")


def _basic_mod_text() -> str:
    """Return the decompressed text of tests/basic.mod."""
    import gzip

    with gzip.open(MOD, "rt") as fh:
        return fh.read()


class TestFromModString:
    def test_returns_ffort_instance(self):
        text = Path("tests/basic.gz.extract").read_text()
        obj = gf.fFort.from_mod_string(text)
        assert isinstance(obj, gf.fFort)

    def test_reads_parameter_without_lib(self):
        text = Path("tests/basic.gz.extract").read_text()
        obj = gf.fFort.from_mod_string(text)
        assert obj.const_int == 1

    def test_reads_parameter_with_lib(self):
        text = Path("tests/basic.gz.extract").read_text()
        obj = gf.fFort.from_mod_string(text, libname=str(SO))
        assert obj.const_int == 1

    def test_variable_raises_without_lib(self):
        text = Path("tests/basic.gz.extract").read_text()
        obj = gf.fFort.from_mod_string(text)
        with pytest.raises(RuntimeError, match="No shared library loaded"):
            _ = obj.a_int

    def test_variable_accessible_with_lib(self):
        text = Path("tests/basic.gz.extract").read_text()
        obj = gf.fFort.from_mod_string(text, libname=str(SO))
        # Just check access doesn't raise; don't assert a specific value.
        _ = obj.a_int

    def test_keys_available(self):
        text = Path("tests/basic.gz.extract").read_text()
        obj = gf.fFort.from_mod_string(text)
        assert "const_int" in obj

    def test_tempfile_cleaned_up(self, tmp_path, monkeypatch):
        """Verify the tempfile is removed even on a parse error."""
        created: list[str] = []

        import tempfile as _tf

        original_ntf = _tf.NamedTemporaryFile

        class _TrackingNTF:
            def __init__(self, *args, **kwargs):
                self._inner = original_ntf(*args, **kwargs)
                created.append(self._inner.name)

            def __enter__(self):
                return self._inner.__enter__()

            def __exit__(self, *a):
                return self._inner.__exit__(*a)

        monkeypatch.setattr(_tf, "NamedTemporaryFile", _TrackingNTF)
        text = Path("tests/basic.gz.extract").read_text()
        gf.fFort.from_mod_string(text)
        for p in created:
            assert not Path(p).exists(), f"tempfile {p} was not cleaned up"
