# SPDX-License-Identifier: GPL-2.0+

import abc
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Type

import gfModParser as gf

from ...types import factory
from .argument import fArg
from .arguments import fArguments

# Handle setting up those return values that actually need to be passed as argument


class fReturnArguments(fArguments):
    pass


class fReturnCharArguments(fReturnArguments):
    pass


class fReturnArrayArguments(fReturnArguments):
    pass


class fReturnDTArguments(fReturnArguments):
    pass
