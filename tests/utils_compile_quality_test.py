# SPDX-License-Identifier: GPL-2.0+

import ctypes
import subprocess
from pathlib import Path

import pytest

import gfort2py.utils as gf_utils
from gfort2py.compilation.compile import Compile, CompileArgs


def test_fc_path_raises_when_lookup_is_empty(monkeypatch):
    monkeypatch.setattr(gf_utils.os, "environ", {})
    monkeypatch.setattr(gf_utils.platform, "system", lambda: "Linux")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b""
        )

    monkeypatch.setattr(gf_utils.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="Did not find a gfortran compiler"):
        gf_utils.fc_path()


def test_compile_args_preserve_quoted_flags(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    observed_cmd = []

    def fake_output_folder():
        return out_dir

    def fake_run(cmd, input, capture_output, check):
        observed_cmd.extend(cmd)
        out_index = cmd.index("-o") + 1
        Path(cmd[out_index]).touch()
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b""
        )

    monkeypatch.setattr(
        "gfort2py.compilation.compile.output_folder", fake_output_folder
    )
    monkeypatch.setattr("gfort2py.compilation.compile.subprocess.run", fake_run)

    c = Compile("end", "quoted_flags", fc="/usr/bin/gfortran")
    ok = c.compile(
        args=CompileArgs(
            INCLUDE_FLAGS='-I"/tmp/include path"',
            LDFLAGS='-Wl,-rpath,"/tmp/rpath dir"',
        )
    )

    assert ok
    assert "-I/tmp/include path" in observed_cmd
    assert "-Wl,-rpath,/tmp/rpath dir" in observed_cmd


def test_compile_uses_fresh_default_args_each_call(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    call_commands = []

    def fake_output_folder():
        return out_dir

    def fake_run(cmd, input, capture_output, check):
        call_commands.append(cmd)
        out_index = cmd.index("-o") + 1
        Path(cmd[out_index]).touch()
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b""
        )

    monkeypatch.setattr(
        "gfort2py.compilation.compile.output_folder", fake_output_folder
    )
    monkeypatch.setattr("gfort2py.compilation.compile.subprocess.run", fake_run)

    c = Compile("end", "defaults", fc="/usr/bin/gfortran")
    assert c.compile()
    assert c.compile()
    assert len(call_commands) == 2


def test_strlen_ctype_matches_pointer_size():
    expected = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32
    assert gf_utils.strlen_ctype() is expected


def test_compile_args_preserve_windows_backslashes(monkeypatch):
    monkeypatch.setattr("gfort2py.compilation.compile.os.name", "nt")
    args = CompileArgs(INCLUDE_FLAGS="-Itests\\build")
    assert "-Itests\\build" in args.argv()
