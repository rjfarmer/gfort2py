# SPDX-License-Identifier: GPL-2.0+

import subprocess
from pathlib import Path

import gfort2py.utils as gf_utils
from gfort2py.compilation.compile import Compile, CompileArgs


def test_fc_path_raises_when_lookup_is_empty(monkeypatch):
    monkeypatch.delenv("FC", raising=False)

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b""
        )

    monkeypatch.setattr(gf_utils.subprocess, "run", fake_run)

    try:
        gf_utils.fc_path()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Did not find a gfortran compilier" in str(exc)


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
