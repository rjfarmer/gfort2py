from utils import *


modname = "vars_modules"
libname = "t_vars_modules"
filename_f90 = "../t_vars_modules.f90"
filename_py = "../t_vars_modules_test.py"


def make_func(func_name, name, v, n):
    f = []
    f.append(fort_func_start.substitute(name=func_name, args="", result="x"))
    f.append(INDENT + fort_bool_var.substitute(var="x"))
    f.append(INDENT + fort_var_set.substitute(name="x", value=".false."))
    f.append(INDENT + fort_var_comp.substitute(name=name, value=f"{v}_{n}"))
    f.append(INDENT + fort_var_set.substitute(name="x", value=".true."))
    f.append(INDENT + fort_func_end.substitute(name=func_name))

    return "\n".join(f)


def create_ints():
    fort_strs = []
    fort_funcs = []
    py_strs = []
    values = [-1, 0, 1]

    for n, k in fort_int_kinds.items():
        for indv, v in enumerate(values):
            name = f"int_{n}_{indv}"

            fort_strs.append(
                fort_set_var.substitute(
                    type="integer", kind=n, var=name, value=f"{v}_{n}"
                )
            )

            py_strs.append(py_value_comp.substitute(name=name, value=v))

            v = 2 * v
            py_strs.append(py_value_set.substitute(name=name, value=v))
            py_strs.append(py_value_comp.substitute(name=name, value=v))

            func_name = f"check_{name}"
            py_strs.append(py_proc_call.substitute(proc=func_name, args=""))
            py_strs.append(py_proc_return.substitute())

            fort_funcs.append(make_func(func_name, name, v, n))

    return fort_strs, fort_funcs, py_strs


def create_reals():
    fort_strs = []
    fort_funcs = []
    py_strs = []
    values = [-3.140000104904175, 0, 3.140000104904175]

    for n, k in fort_real_kinds.items():
        for indv, v in enumerate(values):
            name = f"real_{n}_{indv}"

            fort_strs.append(
                fort_set_var.substitute(type="real", kind=n, var=name, value=f"{v}_{n}")
            )

            py_strs.append(py_value_comp.substitute(name=name, value=v))

            v = 2 * v
            py_strs.append(py_value_set.substitute(name=name, value=v))
            py_strs.append(py_value_comp.substitute(name=name, value=v))

            func_name = f"check_{name}"
            py_strs.append(py_proc_call.substitute(proc=func_name, args=""))
            py_strs.append(py_proc_return.substitute())

            fort_funcs.append(make_func(func_name, name, v, n))

    return fort_strs, fort_funcs, py_strs


with open(filename_f90, "w") as file_f90, open(filename_py, "w") as file_py:
    write_line(file_f90, fort_module_start.substitute(name=modname), 0)
    write_line(file_py, py_module_start.substitute(modname=modname, libname=libname), 0)

    f_ints, ffuncs_ints, py_ints = create_ints()
    write_lines(file_f90, f_ints, 1)
    write_line(file_py, py_proc.substitute(function="check_ints"), 1)
    write_lines(file_py, py_ints, 2)

    f_reals, ffuncs_reals, py_reals = create_reals()
    write_lines(file_f90, f_reals, 1)
    write_line(file_py, py_proc.substitute(function="check_reals"), 1)
    write_lines(file_py, py_reals, 2)

    write_line(file_f90, fort_module_mid.substitute(), 1)

    write_lines(file_f90, ffuncs_ints, 1)
    write_lines(file_f90, ffuncs_reals, 1)

    # print(fort_module_mid.substitute(),file=file_f90)
    write_line(file_f90, fort_module_end.substitute(name=modname), 0)
