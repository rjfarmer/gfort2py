# SPDX-License-Identifier: GPL-2.0+

import abc
import ctypes
from pathlib import Path
import os
import subprocess
import platform


__all__ = ["factory_platform", "PlatformABC"]


class PlatformABC(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load_library(self, libname: Path) -> ctypes.CDLL:
        pass

    @property
    @abc.abstractmethod
    def library_ext(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def library_flags(self) -> list[str]:
        pass

    @property
    @abc.abstractmethod
    def fcpath(self) -> Path:
        pass


class PlatformLinux(PlatformABC):

    def load_library(self, libname: Path) -> ctypes.CDLL:
        libname = Path.resolve(Path(libname))
        if not libname.exists():
            raise FileNotFoundError(f"Can't find {libname}")

        return ctypes.CDLL(libname)

    @property
    def library_ext(self) -> str:
        return "so"

    @property
    def library_flags(self) -> list[str]:
        return ["-fPIC", "-shared"]

    def fcpath(self, path=None) -> Path:
        if path is not None:
            return Path(path)

        if "FC" in os.environ:
            return Path(os.environ["FC"])

        path = (
            Path.resolve(
                subprocess.run(["which", "gfortran"], capture_output=True)
                .stdout.decode()
                .strip()
            )
            .split()[0]
            .strip()
        )

        return path


class PlatformMac(PlatformABC):

    def load_library(self, libname: Path) -> ctypes.CDLL:
        libname = Path.resolve(Path(libname))
        if not libname.exists():
            raise FileNotFoundError(f"Can't find {libname}")

        return ctypes.CDLL(libname)

    @property
    def library_ext(self) -> str:
        return "dylib"

    @property
    def library_flags(self) -> list[str]:
        return ["-dynamiclib"]

    def fcpath(self, path=None) -> Path:
        if path is not None:
            return Path(path)

        if "FC" in os.environ:
            return Path(os.environ["FC"])

        if os.path.exists("/usr/local/bin/gfortran"):
            return Path("/usr/local/bin/gfortran")

        path = (
            Path.resolve(
                subprocess.run(["which", "gfortran"], capture_output=True)
                .stdout.decode()
                .strip()
            )
            .split()[0]
            .strip()
        )

        return path


class PlatformWindows(PlatformABC):
    def load_library(self, libname: Path) -> ctypes.CDLL:
        libname = Path.resolve(Path(libname))
        if not libname.exists():
            raise FileNotFoundError(f"Can't find {libname}")

        kwargs = {}
        os.add_dll_directory(libname.parent)
        kwargs["winmode"] = 0

        return ctypes.CDLL(str(libname), **kwargs)

    @property
    def library_ext(self) -> str:
        return "dll"

    @property
    def library_flags(self) -> list[str]:
        return ["-shared"]

    def fcpath(self, path=None) -> Path:
        if path is not None:
            return Path(path)

        if "FC" in os.environ:
            return Path(os.environ["FC"])

        path = (
            Path.resolve(
                subprocess.run(["where", "gfortran"], capture_output=True)
                .stdout.decode()
                .strip()
            )
            .split()[0]
            .strip()
        )

        return path


def factory_platform() -> PlatformABC:
    os_platform = platform.system()
    if os_platform == "Darwin":
        return PlatformMac()
    elif os_platform == "Windows":
        return PlatformWindows()
    else:
        return PlatformLinux()
