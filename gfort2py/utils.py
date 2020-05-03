# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function

import numpy as np
import ctypes
import numbers

from .errors import *


def find_key_val(list_dicts, key, value):
    v = value.lower()
    for idx, i in enumerate(list_dicts):
        if i[key].lower() == v:
            return idx
