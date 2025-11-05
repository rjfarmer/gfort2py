# SPDX-License-Identifier: GPL-2.0+


import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict

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


class Compile:
    def __init__(self, text: str, name: str, fc=None):
        self.text = text
        self.name = name

        self.platform = factory_platform()
        self._fc = fc

    def compile(self, *, args: CompileArgs = CompileArgs) -> bool:
        # Compile code, reading from stdin
        try:
            p = subprocess.run(
                [
                    str(self.platform.fcpath(self._fc)),
                    "-o",
                    str(self.library_filename),
                    "-x",
                    "f95",
                    *str(args).split(),
                    *self.platform.library_flags,
                    "-J",
                    str(output_folder()),
                    "-",
                ],
                input=self.text.encode(),
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            print(p.stdout)
            print(p.stderr.decode())
            raise

        # print(self.text)

        # print(p.stdout)
        # print(p.stderr.decode())
        # print(" ".join(p.args))

        # print(self.library_filename)
        return self.library_filename.exists()

    @property
    def library_filename(self) -> str:
        output = output_folder()
        return output.joinpath(f"lib{self.name}.{self.platform.library_ext}")

    @property
    def module_filename(self) -> str:
        output = output_folder()
        return output.joinpath(f"{self.name}.mod")

    @property
    def object_filename(self) -> str:
        output = output_folder()
        return output.joinpath(f"{self.name}.o")
