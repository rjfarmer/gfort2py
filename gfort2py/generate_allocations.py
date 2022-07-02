import subprocess
import os


dbg = True

types = [
    {
        "name": "integer",
        "kinds": ["INT8", "INT16", "INT32", "INT64"],
        "needs_len": False,
        "needs_type": False,
    },
    {
        "name": "real",
        "kinds": ["REAL32", "REAL64", "REAL128"],
        "needs_len": False,
        "needs_type": False,
    },
    {
        "name": "complex",
        "kinds": ["REAL32", "REAL64", "REAL128"],
        "needs_len": False,
        "needs_type": False,
    },
    {
        "name": "logical",
        "kinds": ["4"],
        "needs_len": False,
        "needs_type": False,
    },
    {
        "name": "character",
        "kinds": ["4"],
        "needs_len": True,
        "needs_type": False,
    },
]

if dbg:
    types.append(
        {
            "name": "abc",
            "kinds": ["4"],
            "needs_len": False,
            "needs_type": True,
        }
    )


def make_subs(f, name, kind, dims, needs_len, needs_type):
    dim_spec = ",".join(":" * dims)

    bounds = ""
    for b in range(1, dims + 1):
        bounds += f"1:bounds({b}),"
    bounds = bounds[0:-2]

    if needs_type:
        type = f"type({name})"
    else:
        type = name

    if needs_len:
        type = type + "(len=n)"

    if not needs_len and not needs_type:
        type = f"{type}({kind})"

    if needs_len:
        dec = f"{type},allocatable :: x({dim_spec})\n"
        sub = "(x, bounds, n)\n"
        de_sub = "(x, n)\n"
    else:
        dec = f"{type},allocatable :: x({dim_spec})\n"
        sub = "(x, bounds)\n"
        de_sub = "(x)\n"

    f.write(f"subroutine allocate_{name}_{kind}_{dims}" + sub)
    if needs_len:
        f.write("integer,intent(in) :: n\n")
    f.write(dec)
    f.write(f"integer,intent(in) :: bounds({dims})\n")
    f.write("if(allocated(x)) deallocate(x)\n")
    f.write(f"allocate(x({bounds})))\n")
    f.write("end subroutine\n\n")

    f.write(f"subroutine deallocate_{name}_{kind}_{dims}" + de_sub)
    if needs_len:
        f.write("integer,intent(in) :: n\n")
    f.write(dec)
    f.write("if(allocated(x)) deallocate(x)\n")
    f.write("end subroutine\n\n")


def make_flags(opt, flags):
    line = []
    if flags is not None:
        for f in flags:
            flags.append(f"{opt}{f}")
    return line


def compile(filename, module, inc_flags=None, lib_dir=None, lib_incs=None):
    flags = []
    flags = (
        make_flags("-I", inc_flags)
        + make_flags("-L", lib_dir)
        + make_flags("-l", lib_incs)
    )

    subprocess.call(["gfortran", "-ggdb", "-fPIC", "-shared", *flags, "-c", filename])
    subprocess.call(
        [
            "gfortran",
            "-ggdb",
            "-fPIC",
            "-shared",
            *flags,
            "-o",
            f"lib{module}.so",
            filename,
        ]
    )


def make_allocs(
    types, filename="allocation.f90", module="alloc", max_ndims=5, includes=None
):

    with open(filename, "w") as f:
        f.write(f"module {module}\n\n")
        f.write("use iso_fortran_env\n")
        if includes is not None:
            for i in includes:
                f.write(f"use {i}\n")

        f.write("implicit none\n\n")

        if dbg:
            f.write("type abc\n")
            f.write("integer :: x\n")
            f.write("end type\n")

        f.write("contains\n\n")

        for t in types:
            name = t["name"]
            if t["kinds"] is not None:
                for k in t["kinds"]:
                    for n in range(1, max_ndims + 1):
                        make_subs(f, name, k, n, t["needs_len"], t["needs_type"])

        f.write(f"end module {module}\n")


class wd(object):
    def __init__(self, *args, **kwargs):
        self.old = None

    def __enter__(self, *args, **kwargs):
        self.old = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        return None

    def __exit__(self, *args, **kwargs):
        os.chdir(self.old)


if __name__ == "__main__":
    with wd() as w:
        filename = "allocation.f90"
        module = "alloc"
        make_allocs(types, filename, module)
        compile(filename, module)
