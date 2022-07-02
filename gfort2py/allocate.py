import ctypes
import os

from .generate_allocations import compile, wd

_libname = "liballoc.so"

_clib = None


def setup():
    global _clib
    with wd() as w:
        compile("allocation.f90", "alloc")
    _clib = ctypes.CDLL(f"{os.path.dirname(__file__)}/{_libname}")


def _make_name(type, kind, ndims):
    type = type.lower()

    if type == "integer":
        kind = f"int{kind*8}"
    elif type == "real" or type == "complex":
        kind = f"real{kind*8}"
    else:
        kind = "4"

    return f"{type}_{kind}_{ndims}"


def _make_bounds(shape):
    bounds = ctypes.c_int32 * len(shape)
    return bounds(*shape)


def alloc(module, ftype, type, kind, shape, clen=None):

    fname = f"allocate_{_make_name(type,kind,len(shape))}"
    func = getattr(_clib, f"__{module}_MOD_{fname}")

    args = [
        ftype,
        _make_bounds(shape),
    ]
    if clen is not None:
        args.append(ctypes.c_int32(clen))

    fargs = [ctypes.pointer(i) for i in args]

    func(*fargs)

    return


def dealloc(module, ftype, type, kind, shape, clen=None, head=0):

    if ftype.base_addr is None or ftype.base_addr == 0:
        return

    fname = f"deallocate_{_make_name(type,kind,len(shape))}"

    func = getattr(_clib, f"__{module}_MOD_{fname}")

    args = [
        ftype,
    ]
    if clen is not None:
        args.append(ctypes.c_int32(clen))

    fargs = [ctypes.pointer(i) for i in args]

    func(*fargs)

    ftype.base_addr = 0
    return
