# SPDX-License-Identifier: GPL-2.0+

import abc
import ctypes
from pathlib import Path
import os
import subprocess
import platform


__all__ = ["factory_platform", "PlatformABC"]


class PlatformError(Exception):
    pass


class PlatformABC(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def which(self) -> str:
        """
        Get the platform dependant version of "which" command

        Returns:
            str: _description_
        """
        pass

    @abc.abstractmethod
    def load_library(self, libname: Path) -> ctypes.CDLL:
        """Load a ctype library

        Args:
            libname (Path): Path to library

        Returns:
            ctypes.CDLL: Loaded library
        """
        pass

    @property
    @abc.abstractmethod
    def library_ext(self) -> str:
        """Return the platform dependant shared library file extension

        Returns:
            str: file extension
        """
        pass

    @property
    @abc.abstractmethod
    def library_flags(self) -> list[str]:
        """Return platform dependant shared library flags needed for compilation

        Returns:
            list[str]: Compile time flags
        """
        pass

    def _find(self) -> Path:
        """Find the gfortranc ompiler

        Raises:
            PlatformError: _description_

        Returns:
            Path: _description_
        """
        result = subprocess.run(
            [self.which, "gfortran"], capture_output=True, check=False
        )

        if result.returncode != 0:
            raise PlatformError("Could not find a gfortran compiler")

        return Path(result.stdout.decode().strip()).resolve()

    def fcpath(self, path=None) -> Path:
        """Return the platform dependant path to a gfortran compiler.

        If path is not None return Path(path).
        else look for the FC environment variable else use the platform dependant
        which/where to search for gfortran

        Args:
            path (_type_, optional): Path to compiler. Defaults to None.

        Returns:
            Path: _description_
        """
        if path is not None:
            return Path(path)

        if "FC" in os.environ:
            return Path(os.environ["FC"])

        return self._find()


class PlatformLinux(PlatformABC):
    which = "which"

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


class PlatformMac(PlatformABC):
    which = "which"

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

    # @property
    # def fcpath(self, path=None) -> Path:
    #     if path is not None:
    #         return Path(path)

    #     if "FC" in os.environ:
    #         return Path(os.environ["FC"])

    #     if os.path.exists("/usr/local/bin/gfortran"):
    #         return Path("/usr/local/bin/gfortran")

    #     return self._find()


class PlatformWindows(PlatformABC):
    which = "where"

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


def factory_platform() -> PlatformABC:
    os_platform = platform.system()
    if os_platform == "Darwin":
        return PlatformMac()
    elif os_platform == "Windows":
        return PlatformWindows()
    else:
        return PlatformLinux()
