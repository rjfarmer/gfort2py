# SPDX-License-Identifier: GPL-2.0+

from . import parseModCommon as pmc
from .utils import *
from .utils_cpython import *

class parseMod(pmc.parseModBase):
    def __init__(self,*args):
        self.mod_version=14
        super(parseMod,self).__init__(*args)

