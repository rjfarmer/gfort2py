# SPDX-License-Identifier: GPL-2.0+
from functools import cache


class fParam:
    def __init__(self, obj):
        self.obj = obj

    @property
    @cache
    def value(self):
        return self.obj.properties.exp_type.value

    @value.setter
    def value(self, value):
        raise AttributeError("Parameters can't be altered")

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    @property
    def module(self) -> str:
        return self.obj.module
