# SPDX-License-Identifier: GPL-2.0+

from pathlib import Path

from .make_new_test_case import make_test_case


def test_make_test_case_uses_test_directory() -> None:
    name = "generated_test_case"
    test_dir = Path(__file__).resolve().parent
    test_file = test_dir / f"{name}_test.py"
    source_file = test_dir / "src" / f"{name}.f90"

    try:
        make_test_case(name)

        assert 'build_paths("generated_test_case")' in test_file.read_text()
        assert source_file.is_file()
    finally:
        test_file.unlink(missing_ok=True)
        source_file.unlink(missing_ok=True)
