# SPDX-License-Identifier: GPL-2.0+

import ctypes

import gfort2py.utils as utils


def test_get_c_runtime_uses_msvcrt_on_windows(monkeypatch):
    calls = []

    def fake_cdll(name):
        calls.append(name)
        return name

    monkeypatch.setattr(utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    assert utils.get_c_runtime() == "msvcrt.dll"
    assert calls == ["msvcrt.dll"]
