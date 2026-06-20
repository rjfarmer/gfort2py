# SPDX-License-Identifier: GPL-2.0+

import abc
import ctypes
import os
import platform
import subprocess
from pathlib import Path

__all__ = [
    "factory_platform",
    "PlatformABC",
    "PlatformError",
    "is_windows",
    "is_linux",
    "is_mac",
]


WINDOWS_DEFAULT_GFORTRAN_PATHS = [
    r"C:\msys64\ucrt64\bin\gfortran.exe",
    r"C:\msys64\mingw64\bin\gfortran.exe",
    r"C:\msys64\clang64\bin\gfortran.exe",
    r"C:\msys64\clangarm64\bin\gfortran.exe",
    r"C:\msys64\mingw32\bin\gfortran.exe",
    r"C:\msys64\usr\bin\gfortran.exe",
]


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

    def _load_posix_library(self, libname: Path) -> ctypes.CDLL:
        libname = Path(libname).resolve()
        if not libname.exists():
            raise FileNotFoundError(f"Can't find {libname}")
        return ctypes.CDLL(str(libname))

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
        for candidate in result.stdout.decode().splitlines():
            fc = Path(candidate.strip())
            if fc.exists():
                return fc.resolve()

        raise PlatformError("Could not find a gfortran compiler")

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
    @property
    def which(self) -> str:
        return "which"

    def load_library(self, libname: Path) -> ctypes.CDLL:
        return self._load_posix_library(libname)

    @property
    def library_ext(self) -> str:
        return "so"

    @property
    def library_flags(self) -> list[str]:
        return ["-fPIC", "-shared"]


class PlatformMac(PlatformABC):
    @property
    def which(self) -> str:
        return "which"

    def load_library(self, libname: Path) -> ctypes.CDLL:
        return self._load_posix_library(libname)

    @property
    def library_ext(self) -> str:
        return "dylib"

    @property
    def library_flags(self) -> list[str]:
        return ["-dynamiclib"]


class PlatformWindows(PlatformABC):

    @property
    def which(self) -> str:
        return "where"

    def load_library(self, libname: Path) -> ctypes.CDLL:
        libname = Path(libname).resolve()
        if not libname.exists():
            raise FileNotFoundError(f"Can't find {libname}")

        os.add_dll_directory(str(libname.parent))  # type: ignore[attr-defined]
        return ctypes.CDLL(str(libname), winmode=0)  # type: ignore[call-arg]

    @property
    def library_ext(self) -> str:
        return "dll"

    @property
    def library_flags(self) -> list[str]:
        return ["-shared"]

    def _find(self) -> Path:
        try:
            super()._find()
        except PlatformError:
            pass

        for candidate in WINDOWS_DEFAULT_GFORTRAN_PATHS:
            fc = Path(candidate)
            if fc.exists():
                return fc.resolve()

        raise PlatformError("Could not find a gfortran compiler")


def factory_platform() -> PlatformABC:
    os_platform = platform.system()
    if os_platform == "Darwin":
        return PlatformMac()
    elif os_platform == "Windows":
        return PlatformWindows()
    else:
        return PlatformLinux()


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_mac() -> bool:
    return platform.system() == "Darwin"
