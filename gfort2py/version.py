# SPDX-License-Identifier: GPL-2.0+

from importlib import metadata

try:
    __version__ = metadata.version("gfort2py")
except metadata.PackageNotFoundError:
    # package is not installed
    pass
