# SPDX-License-Identifier: GPL-2.0+

import contextlib
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import gfort2py as gf


def build_paths(
    lib_name: str, mod_name: str | None = None, *, as_path: bool = False
) -> tuple[str, str] | tuple[Path, Path]:
    mod_stem = lib_name if mod_name is None else mod_name
    so = f"./tests/build/{lib_name}.{gf.lib_ext()}"
    mod = f"./tests/build/{mod_stem}.mod"
    if as_path:
        return Path(so), Path(mod)
    return so, mod


def pytest_configure(config):
    _ = config
    subprocess.call(["make"], cwd="tests")


@contextlib.contextmanager
def _redirect_fd1():
    """Redirect C-level fd 1 to a temp file and yield a callable to read it back."""
    stdout_fd = 1
    saved_fd = os.dup(stdout_fd)
    result: list[str] = []
    try:
        with tempfile.TemporaryFile() as tmp:
            os.dup2(tmp.fileno(), stdout_fd)
            try:

                def read() -> str:
                    return result[0] if result else ""

                yield read
            finally:
                os.dup2(saved_fd, stdout_fd)
                sys.stdout.flush()
                tmp.seek(0)
                result.append(tmp.read().decode())
    finally:
        os.close(saved_fd)


@pytest.fixture
def fortran_output():
    """Cross-platform fixture that captures Fortran stdout output at the fd level.

    Works where pytest's ``capfd`` is unreliable (e.g. Windows) because it
    redirects file descriptor 1 directly using ``os.dup2``.

    Usage::

        def test_something(fortran_output):
            with fortran_output() as get_output:
                x.some_sub()
            assert get_output().strip() == "expected"
    """
    return _redirect_fd1
