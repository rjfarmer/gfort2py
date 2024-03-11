import subprocess
import tempfile
import os
import platform
import random
import string
import shutil
import hashlib
import platformdirs
from pathlib import Path
import logging

from .utils import library_ext, fc_path

_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None


def compile_and_load(
    string=None,
    file=None,
    FC=None,
    FFLAGS="-O2",
    LDLIBS="",
    LDFLAGS="",
    output=None,
):
    if string is None and file is None:
        raise AttributeError("Must set either string or file")

    if FC is None:
        FC = fc_path()

    if _TEST_FLAG is True:
        logging.debug(f"Found FC={FC}")
        r = subprocess.run([FC, "-v"], capture_output=True)
        logging.debug(r.stdout)
        logging.debug(r.stderr)
        logging.debug(f"Environ = {os.environ.get('FC')}")

    output_dir = output_folder(output)
    output_file = output_filename(file, output_dir)

    if string is not None:
        str2file(string, output_file)
    else:
        shutil.copy(file, output_file)
        moduleize_file(file, output_file)

    mname = mod_name(output_file)

    lib_name = library_name(mname)

    library(lib_name, output_file, output_dir, FC, FFLAGS, LDLIBS, LDFLAGS)

    return os.path.join(output_dir, lib_name), module_filename(mname, output_dir)


def common_compile(
    string=None,
    gfort=None,
    FC=None,
    FFLAGS="-O2",
    LDLIBS="",
    LDFLAGS="",
    output=None,
):
    if "\n" in string:
        string = string.split("\n")

    name = "c" + hashlib.md5(b"".join([i.encode() for i in string])).hexdigest()

    string = "\n".join([f"module {name}", *string, "contains", "end module"])

    lib_path = Path(os.path.realpath(gfort._lib._name))

    LDLIBS = (
        LDLIBS + f"-l:{lib_path.name}"
    )  # colon is needed to search for exact name not lib{name}
    LDFLAGS = LDFLAGS + f"-L{lib_path.parent}"

    FFLAGS += f" -Wl,-rpath='.',-rpath='{lib_path.parent}' -Wl,--no-undefined"

    return compile_and_load(
        string=string,
        FC=FC,
        FFLAGS=FFLAGS,
        LDLIBS=LDLIBS,
        LDFLAGS=LDFLAGS,
        output=output,
    )


class CompileError(Exception):
    pass


def shared_lib_flags():
    os_platform = platform.system()
    if os_platform == "Darwin":
        return ["-dynamiclib"]
    elif os_platform == "Windows":
        return ["-shared"]
    else:
        return ["-fPIC", "-shared"]


def library(lib, file, output, FC, FFLAGS, LDLIBS, LDFLAGS):
    local_file = os.path.basename(file)

    line = " ".join(
        [FC, FFLAGS, *shared_lib_flags(), LDFLAGS, LDLIBS, "-o", lib, local_file]
    )

    res = subprocess.check_output(
        line,
        stderr=subprocess.STDOUT,
        cwd=output,
        shell=True,
    )


def moduleize_file(file, output):
    string = []
    with open(file, "r") as f:
        for line in f.readlines():
            string.append(line.strip("\n"))

    str2file(string, output)


def str2file(string, output):
    string = moduleize(string)
    with open(output, "w") as f:
        f.write("\n".join(string))


def moduleize(string):
    if "\n" in string:
        string = string.split("\n")

    # Check if already a module
    count = 0
    for line in string:
        line = line.strip()
        if line.startswith("module"):
            count += 1
        if line.startswith("contains"):
            count += 1
        if line.startswith("end module"):
            count += 1

    if count == 3:
        return string  # Already a module

    if count != 0:
        raise ValueError("Malformed module")

    # Module names must start with a letter
    # hashing the contents means the name is static so
    # module_parse caching should speed things up
    name = "a" + hashlib.md5(b"".join([i.encode() for i in string])).hexdigest()

    return [f"module {name}\n", "contains\n", *string, "end module\n"]


def mod_name(file):
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.lower()
            if line.strip().startswith("module "):
                n = line.split()
                name = n[n.index("module") + 1]
                if "!" in name:
                    return name[: name.index("!")]
                else:
                    return name

    raise ValueError(f"Could not determine module name for {file}")


def output_folder(output):
    if output is None:
        # return platformdirs.user_cache_dir("gfort2py")
        return tempfile.mkdtemp(prefix="gfort2py")
    else:
        return os.path.realpath(output)


def output_filename(file, folder):
    if file is None:
        f, name = tempfile.mkstemp(suffix=".f90", prefix="mod_", dir=folder)
        os.close(f)
        return name
    else:
        p = Path(file)
        return os.path.join(folder, f"mod_{p.name}")


def library_name(file):
    return f"lib{file}.{library_ext()}"


def module_filename(module_name, output_folder):
    return os.path.join(output_folder, f"{module_name}.mod")


def random_string(N):
    return "".join(random.choices(string.ascii_lowercase, k=N))
