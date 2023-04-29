import ctypes

from .fVar_t import fVar_t


class fStr(fVar_t):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._len = None

    def ctype(self):
        return self._ctype_base * self.len()

    def from_param(self, value):
        if self.obj.is_deferred_len():
            self._len = len(value)

        if self.cvalue is None:
            self.cvalue = self.ctype()()

        self._value = value

        if hasattr(self._value, "encode"):
            self._value = self._value.encode()

        if len(self._value) > self.len():
            self._value = self._value[: self.len()]
        else:
            self._value = self._value + b" " * (self.len() - len(self._value))

        # self._buf = bytearray(self._value)  # Need to keep hold of the reference
        self.cvalue.value = self._value

        return self.cvalue

    @property
    def value(self):
        try:
            return self.cvalue.value.decode()
        except AttributeError:
            return str(self.cvalue)  # Functions returning str's give us str not bytes

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self._len is None:
            if self.obj.is_deferred_len():
                self._len = len(self.cvalue)
            else:
                self._len = self.obj.strlen.value
        return self._len

    def ctype_len(self):
        return ctypes.c_int64(self.len())

    def __doc__(self):
        try:
            return f"{self.type}(LEN={self.obj.strlen}) :: {self.name}"
        except AttributeError:
            return f"{self.type}(LEN=:) :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)


class fAllocStr(fStr):
    def __init__(self, *args, **kwargs):
        self._len = None
        super().__init__(*args, **kwargs)

    def ctype(self):
        return self._ctype_base

    @property
    def _ctype_base(self):
        return ctypes.c_char_p * self.len()

    @_ctype_base.setter
    def _ctype_base(self, value):
        return ctypes.c_char_p * self.len()

    def from_param(self, value):
        if value is None:
            return ctypes.c_char_p(None)

        self._len = len(value)

        self._value = value

        if hasattr(self._value, "encode"):
            self._value = self._value.encode()

        self.cvalue = self.ctype().from_address(ctypes.addressof(self.cvalue))

        if value is None or not len(value):
            return self.cvalue

        if len(self._value) > self.len():
            self._value = self._value[: self.len()]
        else:
            self._value = self._value + b" " * (self.len() - len(self._value))

        self.cvalue[0] = self._value

        return self.cvalue

    @property
    def value(self):
        try:
            x = self.cvalue[0]
        except:
            return None

        if x is None:
            return None
        else:
            return x.decode()

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self._len is None:
            self._len = 1
        return self._len

    def in_dll(self, lib):
        self.cvalue = self.ctype().in_dll(lib, self.mangled_name)
        return self.cvalue

    def __doc__(self):
        return f"character(LEN=(:)), allocatable :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)
