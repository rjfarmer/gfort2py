import ctypes
from .var import fVar


class fDerivedType(fVar):

    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self._ctype = ctypes.c_void_p
        self._desc = fDerivedTypeDesc(lib, self._dt_def)
        self.TEST_FLAG=TEST_FLAG

    def get(self):
        pass
        # print(self._desc._nameArgs)

    def set_mod(self):
        raise AttributeError("Not implementated yet")

    def __dir__(self):
        lv = []
        if '_desc' in self.__dict__:
            if '_nameArgs' in self._desc:
                lv = [str(i.name) for i in self._desc._nameArgs]
        return lv


class fDerivedTypeDesc(ctypes.Structure):

    def __init__(self, lib, obj=None,TEST_FLAG=False):
        self._lib = lib
        self._args = []
        self._nameArgs = []
        self._typeArgs = []
        self.TEST_FLAG=TEST_FLAG

        if obj is not None:
            self.__dict__.update(obj)
            self.setup_desc()

    def setup_desc(self):
        for i in self.args:
            self._args.append(fVar(self._lib, i))
            self._nameArgs.append(self._args[-1].name.replace("\'", ''))
            self._typeArgs.append(self._args[-1]._ctype)

        self.set_fields(self._nameArgs, self._typeArgs)

    def set_fields(self, nameArgs, typeArgs):
        self._fields_ = [(i, j) for i, j in zip(nameArgs, typeArgs)]

    def add_arg(self, name, ctype):
        self._fields_.append((name, ctype))
