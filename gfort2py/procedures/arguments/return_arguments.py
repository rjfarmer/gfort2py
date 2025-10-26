# SPDX-License-Identifier: GPL-2.0+

import sys
from typing import Type, Any, Iterable
import abc
from dataclasses import dataclass

import gfModParser as gf

from ...types import factory

from .arguments import fArguments
from .argument import fArg

# Handle setting up those return values that actually need to be passed as argument


class fReturnArguments(fArguments):
    pass


class fReturnCharArguments(fReturnArguments):
    pass


class fReturnArrayArguments(fReturnArguments):
    pass


class fReturnDTArguments(fReturnArguments):
    pass
