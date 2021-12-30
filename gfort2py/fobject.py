# SPDX-License-Identifier: GPL-2.0+

class fObject:
    """
    Ojects should inherit this class and have self.value set.

    This class then implements all the normal python dunder methods
    on self.value
    """

    def __eq__(self, other):
        if self.value is None:
            return other is None

        return self.value.__eq__(other)

    def __ne__(self, other):
        return self.value.__neq__(other)

    def __lt__(self, other):
        return self.value.__lt__(other)

    def __le__(self, other):
        return self.value.__le__(other)

    def __gt__(self, other):
        return self.value.__gt__(other)

    def __ge__(self, other):
        return self.value.__ge__(other)

    def __add__(self, other):
        return self.value.__add__(other)

    def __sub__(self, other):
        return self.value.__sub__(other)

    def __matmul__(self, other):
        return self.value.__matmul__(other)

    def __truediv__(self, other):
        return self.value.__truediv__(other)

    def __floordiv__(self, other):
        return self.value.__floordiv__(other)

    def __mod__(self, other):
        return self.value.__mod__(other)

    def __divmod__(self, other):
        return self.value.__divmod__(other)

    def __pow__(self, other, modulo=None):
        return self.value.__pow__(other, modulo)

    def __lshift__(self, other):
        return self.value.__lshift__(other)

    def __rshift__(self, other):
        return self.value.__rshift__(other)

    def __and__(self, other):
        return self.value.__and__(other)

    def __xor__(self, other):
        return self.value.__xor__(other)

    def __or__(self, other):
        return self.value.__or__(other)

    def __radd__(self, other):
        return self.value.__radd__(other)

    def __rsub__(self, other):
        return self.value.__rsub__(other)

    def __rmatmul__(self, other):
        return self.value.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self.value.__rtruediv__(other)

    def __rfloordiv__(self, other):
        return self.value.__rfloordiv__(other)

    def __rmod__(self, other):
        return self.value.__rmod__(other)

    def __rdivmod__(self, other):
        return self.value.__rdivmod__(other)

    def __rlshift__(self, other):
        return self.value.__rlshift__(other)

    def __rrshift__(self, other):
        return self.value.__rrshift__(other)

    def __rand__(self, other):
        return self.value.__rand__(other)

    def __rxor__(self, other):
        return self.value.__rxor__(other)

    def __ror__(self, other):
        return self.value.__ror__(other)

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __complex__(self):
        return self.value.__complex__()

    def __int__(self):
        return self.value.__int__()

    def __float__(self):
        return self.value.__float__()

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)

    def __trunc__(self):
        return self.value.__trunc__()

    def __floor__(self):
        return self.value.__floor__()

    def __ceil__(self):
        return self.value.__ceil__()

    def __len__(self):
        return self.value.__len__()

    def __index__(self):
        return int(self.value)

    def __str__(self):
        return self.value.__str__()

    def __repr__(self):
        return self.value.__repr__()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            return getattr(self.value, attr)

    def __dir__(self):
        return self.value.__dir__()