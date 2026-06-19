# SPDX-License-Identifier: GPL-2.0+


import os
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .platform import factory_platform, is_windows
from .utils import output_folder


@dataclass
class CompileArgs:
    FFLAGS: str = "-O2 -ffree-line-length-none -fimplicit-none -ffree-form -cpp "
    LDLIBS: str = ""
    LDFLAGS: str = ""
    INCLUDE_FLAGS: str = ""

    def __str__(self):
        return " ".join(asdict(self).values()).strip()

    def argv(self) -> list[str]:
        """Return compiler/linker flags as argv tokens preserving quoting."""
        argv: list[str] = []
        for value in asdict(self).values():
            if value:
                split_value = value
                if is_windows():
                    # In POSIX splitting mode, backslashes are escape characters.
                    # Doubling preserves literal Windows path separators.
                    split_value = value.replace("\\", "\\\\")
                argv.extend(shlex.split(split_value, posix=True))
        return argv


class Compile:
    def __init__(self, text: str, name: str, fc=None):
        self.text = text
        self.name = name

        self.platform = factory_platform()
        self._fc = fc
        self._library_filename = output_folder().joinpath(
            f"lib{self.name}_{uuid.uuid4().hex[:8]}.{self.platform.library_ext}"
        )

    def compile(self, *, args: CompileArgs | None = None) -> bool:
        # Compile code, reading from stdin
        compile_args = CompileArgs() if args is None else args
        print(f"Compiling {self.name} with args: {compile_args}")
        print(f"Output library: {self.library_filename}")
        print(
            [
                str(self.platform.fcpath(self._fc)),
                "-o",
                str(self.library_filename),
                "-x",
                "f95",
                *compile_args.argv(),
                *self.platform.library_flags,
                "-J",
                str(output_folder()),
                "-",
            ],
        )
        p = subprocess.run(
            [
                str(self.platform.fcpath(self._fc)),
                "-o",
                str(self.library_filename),
                "-x",
                "f95",
                *compile_args.argv(),
                *self.platform.library_flags,
                "-J",
                str(output_folder()),
                "-",
            ],
            input=self.text.encode(),
            capture_output=True,
            check=False,
        )
        print(f"Compiler stdout: {p.stdout.decode()}")
        print(f"Compiler stderr: {p.stderr.decode()}")
        print(f"Compiler return code: {p.returncode}")
        if p.returncode != 0:
            raise subprocess.CalledProcessError(
                p.returncode,
                p.args,
                output=p.stdout,
                stderr=p.stderr,
            )

        print(f"Compilation of {self.name} succeeded. {self.library_filename.exists()}")
        return self.library_filename.exists()

    @property
    def library_filename(self) -> Path:
        return self._library_filename

    @property
    def module_filename(self) -> Path:
        output = output_folder()
        return output.joinpath(f"{self.name}.mod")

    @property
    def object_filename(self) -> Path:
        output = output_folder()
        return output.joinpath(f"{self.name}.o")
