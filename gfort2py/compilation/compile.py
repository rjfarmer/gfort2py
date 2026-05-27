# SPDX-License-Identifier: GPL-2.0+


import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .platform import factory_platform
from .utils import output_folder


@dataclass
class CompileArgs:
    FFLAGS: str = "-O2 -ffree-line-length-none -fimplicit-none -ffree-form -cpp"
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
                argv.extend(shlex.split(value))
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

        if p.returncode != 0:
            raise subprocess.CalledProcessError(
                p.returncode,
                p.args,
                output=p.stdout,
                stderr=p.stderr,
            )

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
