# SPDX-License-Identifier: GPL-2.0+

import types
from pyparsing import OneOrMore, nestedExpr
from dataclasses import dataclass

import gzip

import typing as t


filename = 'basic.mod'

with gzip.open(filename) as f:
    x = f.read().decode()


header = x[:x.index('\n')]

data = x[x.index('\n')+1:].replace('\n',' ')




parsed_data = OneOrMore(nestedExpr()).parseString(data)


operators = parsed_data[0]
generics = parsed_data[1]
dt_types = parsed_data[2]
common = parsed_data[3]
overloads = parsed_data[4]
equivalence = parsed_data[5]
symbols = parsed_data[6]
summary = parsed_data[7]

#################################

@dataclass
class s_item:
    name: str
    ambiguous: bool
    id: int

    def __post_init__(self):
        self.name = self.name.replace("'","")
        self.ambiguous = self.ambiguous != '0'
        self.id = int(self.id)

def proc_summary(data):
    result = {}
    for i in range(0, len(data), 3):
        d = s_item(*data[i:i+3])
        result[d.id] = d
    return result

proc_summary(summary)


#################################

@dataclass
class c_item:
    name: str
    id: int
    saved_flag: bool
    _unknown: int

    def __post_init__(self):
        self.name = self.name.replace("'","")
        self.id = int(self.id)
        self.saved_flag  = int(self.saved_flag)
        self._unknown = int(self._unknown)

def proc_common(data):
    result = {}
    for i in data:
        d = c_item(*i)
        result[d.id] = d
    return result

proc_common(common)

#################################

@dataclass
class dt_type:
    name: str
    module: str
    id: int

    def __post_intit__(self):
        self.name = self.name.replace("'","")
        self.module = self.module.replace("'","")
        self.id = int(self.id)

def proc_dt_type(data):
    result = {}
    for i in data:
        d = dt_type(*i)
        result[d.id] = d
    return result

proc_dt_type(dt_types)

#################################

#####################################

@dataclass(init=False)
class attribute:
    flavor: str = ''
    intent: str = ''
    proc: str = ''
    if_source: str = ''
    save: str = '' 
    ext_attr: int = -1
    extension: int = -1
    attributes: t.Tuple[str] = None

    def __init__(self,*args):
        self.flavor = args[0]
        self.intent = args[1]
        self.proc = args[2]
        self.if_source = args[3]
        self.save = args[4]
        self.ext_attr = int(args[5])
        self.extension = int(args[6])
        self.attributes = args[7:]


@dataclass(init=False)
class component:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

@dataclass(init=False)
class namespace:
    args: t.Tuple[int] = None

    def __init__(self,*args):
        self.args = args

@dataclass
class header:
    id: int
    name: str # If first letter is captialised then its a dt
    module: str
    bindc: str
    parent_id: int

    def __post_init__(self):
        self.id = int(self.id)
        self.parent_id = int(self.parent_id)
        self.name = self.name.replace("'","")
        self.module = self.module.replace("'","")
        self.bindc = len(self.bindc.replace("'","")) > 0

@dataclass
class symbol_ref:
    reference: int = -1 

@dataclass(init=False)
class formal_arglist:
    symbol: t.List[symbol_ref] = None

    def __init__(self,*args):
        self.symbol = []
        for i in args:
            self.symbol.append(symbol_ref(i))

@dataclass(init=False)
class derived_namepsace:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

@dataclass(init=False)
class actual_arglist:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@dataclass(init=False)
class typespec:
    type: str = ''
    kind: int = -1 # If class symbol_ref else kind
    class_ref: symbol_ref = None # If class symbol_ref else kind
    interface: symbol_ref = None
    is_c_interop: int = -1
    is_iso_c: int = -1
    type2: str = '' # Repeat of type
    charlen: int = -1 # If character
    deferred_cl: int = -1 #if character and deferred length

    def __init__(self,*args):
        self.type = args[0]
        if self.type == 'CLASS':
            self.class_ref = symbol_ref(*args[1])
        else:
            self.kind = int(args[1])

        if len(args[2]):
            self.interface = symbol_ref(*args[2])

        self.is_c_interop = bool(int(args[3]))
        self.is_iso_c = bool(int(args[4]))
        self.type2 = args[5]
        try:
            self.charlen = int(args[6])
        except TypeError:
            self.charlen = -1
        try:
            self.deferred_cl = int(args[7])
        except (TypeError,IndexError):
            self.deferred_cl = -1

@dataclass
class expression:
    exp_type: str = ''
    ts: typespec = None
    rank: int = -1
    value: t.Any = None
    arglist: actual_arglist = None # PDT's? 

    def __post_init__(self):
        self.ts = typespec(*self.ts)
        self.rank = int(self.rank)

        self.value = self.value.replace("'","")

        if self.ts.type == 'REAL':
            self.value = self.hextofloat(self.value)
        elif self.ts.type == 'INTEGER':
            self.value = int(self.value)
        elif self.ts.type == 'CHARACTER':
            self.value = self.value
        #TODO: Handle arrays, complex etc

    def hextofloat(self,s):
        # Given hex like parameter '0.12decde@9' returns 5065465344.0
        man, exp = s.split("@")
        exp = int(exp)
        decimal = man.index(".")
        man = man[decimal + 1 :]
        man = man.ljust(exp, "0")
        man = man[:exp] + "." + man[exp:]
        man = man + "P0"
        return float.fromhex(man)


@dataclass
class arrayspec:
    rank: int = -1
    corank: int = -1
    array_type: str = ''
    bounds: t.List[t.List[expression]] = None # LIst of lower and upper bounds

@dataclass
class namelist:
    sym_ref: t.List[symbol_ref] = None

@dataclass(init=False)
class simd_dec:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

@dataclass(init=False)
class data:
    attr: attribute
    comp: component = None
    ts: typespec = None
    ns: namespace = None
    common_link: symbol_ref = None
    formal_arg: formal_arglist = None
    parameter: expression = None #If parameter
    array_spec: arrayspec = None
    sym_ref: symbol_ref = None
    sym_ref_cray: symbol_ref = None # If cray_pointer
    derived: derived_namepsace = None
    actual_arg: actual_arglist = None
    nl: namelist = None
    intrinsic: int = -1
    intrinsic_symbol: int = -1
    hash: int = -1
    simd: simd_dec = None

    def __init__(self,*args):
        args = list(args) # Do it this was as there are optional terms we may need to pop
        self.attr = attribute(*args[0])
        self.comp = component(*args[1])
        self.ts = typespec(*args[2])
        self.ns = namespace(*args[3])
        self.common_link = symbol_ref(*args[4])
        self.formal_arg = formal_arglist(*args[5])
        if self.attr.flavor == 'PARAMETER':
            self.parameter = expression(*args[6])
            _ = args.pop(6)
        self.array_spec = arrayspec(*args[7])
        if True:
            self.sym_ref = symbol_ref(*args[8])
        else:
            pass # Ignore cray pointers
        self.derived = derived_namepsace(*args[9])
        self.actual_arg = actual_arglist(*args[10]) 
        self.nl = namelist(*args[11])
        self.intrinsic = int(args[12])
        if len(args) > 13:
            self.intrinsic_symbol = int(args[13])
        if len(args) > 14:
            self.hash = int(args[14])
        if len(args) > 15:
            if args[15] is not None:
                self.simd = simd_dec(*args[15]) 

@dataclass(init=False)
class symbol:
    head: header = None
    sym: data = None

    def __init__(self,*args):
        self.head = header(*args[0:5])
        self.sym = data(*args[5])

def parse_symbols():
    result = {}
    for i in range(0, len(symbols), 6):
        s = symbol(*symbols[i:i+6])
        result[s.head.id] = s

    return result

z=parse_symbols()

z[20].sym.parameter
