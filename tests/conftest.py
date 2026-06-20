# SPDX-License-Identifier: GPL-2.0+

import contextlib
import gzip
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import gfort2py as gf
from gfort2py.compilation.platform import factory_platform

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"


def _compiler_version_major(fc: str) -> int:
    result = subprocess.run([fc, "-dumpversion"], capture_output=True, check=False)
    if result.returncode != 0:
        return 0

    version_text = result.stdout.decode().strip()
    major_text = version_text.split(".", maxsplit=1)[0]
    return int(major_text) if major_text.isdigit() else 0


def _ensure_toolchain_on_path(fc: str) -> None:
    fc_bin = str(Path(fc).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if fc_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([fc_bin, *path_parts])


def _compile_sources(
    fc: str, source_dir: Path, extra_flags: list[str] | None = None
) -> None:
    flags = [
        "-ggdb",
        "-fdump-tree-original",
        "-D_FORTIFY_SOURCE=2",
        "-ffree-line-length-none",
        "-ffree-form",
        "-fstack-clash-protection",
        "-fstack-protector-all",
        "-fstack-protector",
    ]
    if extra_flags:
        flags.extend(extra_flags)

    platform_flags = factory_platform().library_flags
    for src in sorted(source_dir.glob("*.f90")):
        output_lib = BUILD_DIR / f"{src.stem}.{gf.lib_ext()}"
        cmd = [
            fc,
            *flags,
            *platform_flags,
            "-cpp",
            f"-J{BUILD_DIR}",
            "-o",
            str(output_lib),
            str(src),
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise RuntimeError(f"Failed to compile {src.name} with {fc}:\n{stderr}")


def _extract_mod_files() -> None:
    for mod_file in BUILD_DIR.glob("*.mod"):
        extract_target = mod_file.with_suffix(".gz.extract")
        with gzip.open(mod_file, "rb") as gz_file:
            extract_target.write_bytes(gz_file.read())


def _build_test_artifacts() -> None:
    fc = gf.utils.fc_path()
    _ensure_toolchain_on_path(fc)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    _compile_sources(fc, ROOT / "src")

    if _compiler_version_major(fc) >= 15:
        _compile_sources(fc, ROOT / "src15", extra_flags=["-funsigned"])

    _extract_mod_files()


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

    if not is_windows():
        subprocess.call(["make"], cwd="tests")

    try:
        _build_test_artifacts()
    except ValueError as exc:
        raise pytest.UsageError(
            "Could not find gfortran in this environment. Set FC to a full compiler path "
            "or ensure gfortran is on PATH."
        ) from exc
    except RuntimeError as exc:
        raise pytest.UsageError(str(exc)) from exc


def is_windows() -> bool:
    return platform.system() == "Windows"


def pytest_runtest_setup(item):

    if is_windows():
        for _ in item.iter_markers(name="skipIfWindows"):
            pytest.skip()


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
