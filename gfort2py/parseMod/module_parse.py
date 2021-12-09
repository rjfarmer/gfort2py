# SPDX-License-Identifier: GPL-2.0+

from pyparsing import OneOrMore, nestedExpr
from dataclasses import dataclass
import numpy as np
import gzip
import sys

import pprint

import typing as t


def string_clean(string):
    if string is None:
        return
    if string.startswith("'") or string.startswith('"'):
        string = string[1:]
    if string.endswith("'") or string.endswith('"'):
        string = string[:-1]

    return string


class VersionError(Exception):
    pass


#################################


@dataclass
class s_item:
    name: str
    ambiguous: bool
    id: int

    def __post_init__(self):
        self.name = string_clean(self.name)
        self.ambiguous = self.ambiguous != "0"
        self.id = int(self.id)


class Summary:
    def __init__(self, summ):
        self._item_id = {}
        self._item_name = {}

        for i in range(0, len(summ), 3):
            d = s_item(*summ[i : i + 3])
            self._item_id[d.id] = d
            self._item_name[d.name] = d

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._item_id[key]
        if isinstance(key, str):
            return self._item_name[key]
        else:
            raise TypeError(f"Dont understand type of {key}")

    def keys(self):
        return list(self._item_id.keys()) + list(self._item_name.keys())

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self._item_id
        if isinstance(key, str):
            return key in self._item_name
        else:
            raise TypeError(f"Dont understand type of {key}")

    def names(self):
        return self._item_name.keys()


#################################


@dataclass
class c_item:
    name: str
    id: int
    saved_flag: bool
    _unknown: int
    _unknown2: str

    def __post_init__(self):
        self.name = string_clean(self.name)
        self.id = int(self.id)
        self.saved_flag = int(self.saved_flag)
        self._unknown = int(self._unknown)
        self._unknown2 = string_clean(self._unknown2)


#################################


@dataclass(init=False)
class generics:
    name: str = ""
    module: str = ""
    id: t.List[int] = -1

    def __init__(self, *args):
        self.name = string_clean(args[0])
        self.module = string_clean(args[1])
        self.id = []
        for i in args[2:]:
            self.id.append(int(i))


#################################


def hextofloat(s):
    # Given hex like parameter '0.12decde@9' returns 5065465344.0
    man, exp = s.split("@")
    exp = int(exp)
    decimal = man.index(".")
    man = man[decimal + 1 :]
    man = man.ljust(exp, "0")
    man = man[:exp] + "." + man[exp:]
    man = man + "P0"
    return float.fromhex(man)


#####################################


def print_args(x):
    print()
    for i in x:
        print(i)
    print()


@dataclass(init=False)
class attribute:
    flavor: str = ""
    intent: str = ""
    proc: str = ""
    if_source: str = ""
    save: str = ""
    ext_attr: int = -1
    extension: int = -1
    attributes: t.Tuple[str] = None

    def __init__(self, *args):
        self.flavor = string_clean(args[0])
        self.intent = string_clean(args[1])
        self.proc = string_clean(args[2])
        self.if_source = string_clean(args[3])
        self.save = string_clean(args[4])
        self.ext_attr = int(args[5])
        self.extension = int(args[6])
        self.attributes = [string_clean(i) for i in args[7:]]


@dataclass
class namespace:
    ref: int = -1

    def __post_init__(self):
        self.ref = symbol_ref(self.ref)


@dataclass
class header:
    id: int
    name: str  # If first letter is captialised then its a dt
    module: str
    bindc: str
    parent_id: int

    def __post_init__(self):
        self.id = int(self.id)
        self.parent_id = int(self.parent_id)
        self.name = string_clean(self.name)
        self.module = string_clean(self.module)
        self.bindc = len(string_clean(self.bindc)) > 0

    @property
    def mn_name(self):
        if self.module:
            return f"__{self.module}_MOD_{self.name}"


@dataclass
class symbol_ref:
    ref: int = -1

    def __post_init__(self):
        self.ref = int(self.ref)


@dataclass(init=False)
class formal_arglist:
    symbol: t.List[symbol_ref] = None

    def __init__(self, *args):
        self.symbol = []
        for i in args:
            self.symbol.append(symbol_ref(i))

    def __len__(self):
        return len(self.symbol)

    def __iter__(self):
        return iter(self.symbol)


@dataclass(init=False)
class derived_ns:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@dataclass(init=False)
class actual_arglist:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class typebound_proc:
    pass


@dataclass(init=False)
class typespec:
    type: str = ""
    kind: int = -1  # If class symbol_ref else kind
    class_ref: symbol_ref = None  # If class/derived type symbol_ref else kind
    interface: symbol_ref = None
    is_c_interop: int = -1
    is_iso_c: int = -1
    type2: str = ""  # Repeat of type
    charlen: int = -1  # If character
    deferred_cl: bool = False  # if character and deferred length

    def __init__(self, *args):
        self.type = args[0]
        if self.type == "CLASS" or self.type == "DERIVED":
            self.class_ref = symbol_ref(args[1])
        else:
            self.kind = int(args[1])

        if len(args[2]):
            self.interface = symbol_ref(args[2])

        self.is_c_interop = bool(int(args[3]))
        self.is_iso_c = bool(int(args[4]))
        self.type2 = args[5]
        try:
            self.charlen = expression(
                *args[6][0]
            )  # TODO: might this need to be iterated for mulit-d strings?
        except IndexError:
            self.charlen = -1
        try:
            self.deferred_cl = args[7] == "DEFERRED_CL"
        except (TypeError, IndexError):
            self.deferred_cl = False


@dataclass(init=False)
class expression:
    exp_type: str = ""
    ts: typespec = None
    rank: int = -1
    value: t.Any = None
    arglist: actual_arglist = None  # PDT's?
    charlen: int = -1

    def __init__(self, *args):
        self.exp_type = args[0]
        self.ts = typespec(*args[1])
        self.rank = int(args[2])

        if self.exp_type == "OP":
            print(args)
        elif self.exp_type == "FUNCTION":
            print(args)
        elif self.exp_type == "CONSTANT":
            if self.ts.type == "REAL":
                self.value = hextofloat(string_clean(args[3]))
            elif self.ts.type == "INTEGER":
                self.value = int(string_clean(args[3]))
            elif self.ts.type == "CHARACTER":
                self.charlen = int(args[3])
                self.value = string_clean(args[4])
            elif self.ts.type == "COMPLEX":
                self.value = complex(
                    hextofloat(string_clean(args[3])), hextofloat(string_clean(args[4]))
                )
            else:
                print(args)
        elif self.exp_type == "VARIABLE":
            self.value = symbol_ref(args[3])
        elif self.exp_type == "SUBSTRING":
            print(args)
        elif self.exp_type == "ARRAY":
            self.value = []
            for i in args[3]:
                self.value.append(
                    expression(*i[0]).value
                )  # Wheres the extra component comming from?
        elif self.exp_type == "NULL":
            print(args)
        elif self.exp_type == "COMPCALL":
            print(args)
        elif self.exp_type == "PPC":
            print(args)
        else:
            raise AttributeError(f"Can't match {self.exp_type}")


@dataclass(init=False)
class arrayspec:
    rank: int = -1
    corank: int = -1
    array_type: str = ""
    lower: t.List[expression] = None
    upper: t.List[expression] = None

    def __init__(self, *args):
        if not len(args):
            return

        self.rank = int(args[0])
        self.corank = int(args[1])
        self.array_type = args[2]
        self.lower = []
        self.upper = []
        for i in range(self.rank + self.corank):
            if len(args[3 + i * 2]):
                self.lower.append(expression(*args[3 + i * 2]))
            if len(args[4 + i * 2]):
                self.upper.append(expression(*args[4 + i * 2]))

    @property
    def fshape(self):
        res = []
        for l, u in zip(self.lower, self.upper):
            res.append([l.value, u.value])

        return res

    @property
    def pyshape(self):
        res = []
        if self.lower is None:
            return []

        for l, u in zip(self.lower, self.upper):
            res.append(u.value - l.value + 1)

        return res

    @property
    def size(self):
        return np.product(self.pyshape)


@dataclass(init=False)
class component:
    id: int = -1
    name: str = ""
    ts: typespec = None
    array_spec: arrayspec = None
    expr: expression = None
    actual_arg: actual_arglist = None
    attr: attribute = None
    access: str = ""
    initializer: expression = None
    proc_ptr: typebound_proc = None

    def __init__(self, *args):
        args = list(args)

        self.id = int(args[0])
        self.name = string_clean(args[1])
        self.ts = typespec(*args[2])
        self.array_spec = arrayspec(*args[3])
        if len(args[4]):
            self.expr = expression(*args[4])
        if len(args[5]):
            self.actual_arg = actual_arglist(*args[5])
        self.attr = attribute(*args[6])
        self.access = string_clean(args[7])

        if self.name == "_final" or self.name == "_hash":
            self.initializer = expression(*args[8])
            _ = args.pop(8)

        if not self.attr.proc == "UNKNOWN-PROC":
            self.proc_ptr = typebound_proc(args[8])


@dataclass(init=False)
class components:
    comp: t.List[component] = None

    def __init__(self, *args):
        self.comp = []
        for i in args:
            self.comp.append(component(*i))

    def __len__(self):
        return len(self.comp)

    def __iter__(self):
        return iter(self.comp)


@dataclass(init=False)
class namelist:
    sym_ref: t.List[symbol_ref] = None

    def __init__(self, *args):
        self.sym_ref = []
        if len(args):
            for i in args:
                self.sym_ref.append(symbol_ref(i))


@dataclass(init=False)
class simd_dec:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@dataclass(init=False)
class data:
    attr: attribute
    comp: components = None
    comp_access: str = ""  # Only for DT's
    ts: typespec = None
    ns: namespace = None
    common_link: symbol_ref = None
    formal_arg: formal_arglist = None
    parameter: expression = None  # If parameter
    array_spec: arrayspec = None
    sym_ref: symbol_ref = None
    sym_ref_cray: symbol_ref = None  # If cray_pointer
    derived: derived_ns = None
    actual_arg: actual_arglist = None
    nml: namelist = None
    intrinsic: int = -1
    intrinsic_symbol: int = -1
    hash: int = -1
    simd: simd_dec = None

    def __init__(self, *args):
        args = list(
            args
        )  # Do it this was as there are optional terms we may need to pop
        self.attr = attribute(*args[0])
        self.comp = components(*args[1])

        if isinstance(args[2], str):
            self.comp_access = args[2]
            _ = args.pop(2)

        self.ts = typespec(*args[2])
        self.ns = namespace(args[3])
        self.common_link = symbol_ref(args[4])
        self.formal_arg = formal_arglist(*args[5])
        if self.attr.flavor == "PARAMETER":
            self.parameter = expression(*args[6])
            _ = args.pop(6)
        self.array_spec = arrayspec(*args[6])
        if True:
            self.sym_ref = symbol_ref(args[7])
        else:
            pass  # Ignore cray pointers
        self.derived = derived_ns(*args[8])
        self.actual_arg = actual_arglist(*args[9])
        self.nml = namelist(*args[10])
        self.intrinsic = int(args[11])
        if len(args) > 12:
            self.intrinsic_symbol = int(args[12])
        if len(args) > 13:
            self.hash = int(args[13])
        if len(args) > 14:
            if args[15] is not None:
                self.simd = simd_dec(*args[14])


@dataclass(init=False)
class symbol:
    head: header = None
    sym: data = None
    raw: str = ""

    def __init__(self, *args):
        self.head = header(*args[0:5])
        self.sym = data(*args[5])
        self.raw = args


class module(object):
    version = 15

    def __init__(self, filename, load_only=False):
        self.filename = filename

        with gzip.open(self.filename) as f:
            x = f.read().decode()

        self.mod_info = x[: x.index("\n")]

        v = int(self.mod_info.split("'")[1])

        if v != self.version:
            raise VersionError("Unsupported module version")

        data = x[x.index("\n") + 1 :].replace("\n", " ")

        self.parsed_data = OneOrMore(nestedExpr()).parseString(data)

        if not load_only:
            self.interface = self.parsed_data[0]

            self.operators = self.parsed_data[1]
            self.generics = self.proc_generics(self.parsed_data[2])

            self.common = self.proc_common(self.parsed_data[3])
            self.equivalence = self.parsed_data[4]

            self.omp = self.parsed_data[5]

            self.symbols = self.parse_symbols(self.parsed_data[6])
            self.summary = Summary(self.parsed_data[7])

    def parse_symbols(self, data):
        result = {}
        for i in range(0, len(data), 6):
            s = symbol(*data[i : i + 6])
            result[s.head.id] = s

        return result

    def proc_common(self, data):
        result = {}
        for i in data:
            d = c_item(*i)
            result[d.id] = d
        return result

    def proc_generics(self, data):
        result = {}
        for i in data:
            d = generics(*i)
            result[d.name] = d
        return result

    def keys(self):
        return self.summary.names()

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        try:
            return self.symbols[self.summary[key].id]
        except KeyError:
            # Not a global variable maybe a function argument?
            return self.symbols[key]


if __name__ == "__main__":
    m = module(filename=sys.argv[1])
    for i in m.keys():
        pprint.pprint(m[i])
