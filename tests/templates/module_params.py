from utils import *


modname = "params_modules"
libname = "t_params_modules"
filename_f90 = "../t_params_modules.f90"
filename_py = "../t_params_modules_test.py"


def create_ints():
    fort_strs = []
    py_strs = []
    values = [-1, 0, 1]

    for n, k in fort_int_kinds.items():
        for indv, v in enumerate(values):
            name = f"int_{n}_{indv}"

            fort_strs.append(
                fort_param.substitute(
                    type="integer", kind=n, var=name, value=f"{v}_{n}"
                )
            )

            py_strs.append(py_value_comp.substitute(name=name, value=v))
            py_strs.append(py_fail.substitute(var=name, bad=-99))

    return fort_strs, py_strs


def create_ints_array_1d():
    fort_strs = []
    py_strs = []
    values = np.array([-10, -1, 0, 1, 10])
    v = f"{make_array_values(values)}"
    vstr = f"np.array([{make_array_values(values)}])"
    for n, k in fort_int_kinds.items():
        name = f"int_{n}_1d"

        fort_strs.append(
            fort_param_array_multi.substitute(
                type="integer", kind=n, var=name, shape=len(values), value=v
            )
        )

        py_strs.append(py_array_check.substitute(name=name, value=vstr))
        py_strs.append(py_fail.substitute(var=name, bad="np.array([1,2,3])"))

    return fort_strs, py_strs


def create_ints_array_2d():
    fort_strs = []
    py_strs = []
    values = np.array([-10, -1, 0, 1, 10, 50]).reshape(2, 3)
    v = f"{make_array_values(values)}"
    vstr = f'np.array([{make_array_values(values)}]).reshape(2,3, order="F")'
    for n, k in fort_int_kinds.items():
        name = f"int_{n}_2d"

        fort_strs.append(
            fort_param_array_multi.substitute(
                type="integer", kind=n, var=name, shape="2,3", value=v
            )
        )

        py_strs.append(py_array_check.substitute(name=name, value=vstr))
        py_strs.append(py_fail.substitute(var=name, bad="np.array([1,2,3])"))

    return fort_strs, py_strs


def create_reals():
    fort_strs = []
    py_strs = []
    values = [-3.140000104904175, 0, 3.140000104904175]

    for n, k in fort_real_kinds.items():
        for indv, v in enumerate(values):
            name = f"real_{n}_{indv}"

            fort_strs.append(
                fort_param.substitute(type="real", kind=n, var=name, value=f"{v}_{n}")
            )

            py_strs.append(py_value_comp.substitute(name=name, value=v))
            py_strs.append(py_fail.substitute(var=name, bad=-99.9))

    return fort_strs, py_strs


def create_reals_array_1d():
    fort_strs = []
    py_strs = []
    values = np.array([-3.140000104904175, 0, 3.140000104904175])
    v = f"{make_array_values(values)}"
    vstr = f"np.array([{make_array_values(values)}])"

    for n, k in fort_real_kinds.items():
        name = f"real_{n}_1d"
        v = f"{make_array_values(values,n)}"

        fort_strs.append(
            fort_param_array_multi.substitute(
                type="real", kind=n, var=name, shape=len(values), value=v
            )
        )

        py_strs.append(py_array_check.substitute(name=name, value=vstr))
        py_strs.append(py_fail.substitute(var=name, bad="np.array([1,2,3])"))

    return fort_strs, py_strs


def create_reals_array_2d():
    fort_strs = []
    py_strs = []
    values = np.array(
        [
            -2 * 3.140000104904175,
            -3.140000104904175,
            0.0,
            1.1111,
            3.140000104904175,
            2 * 3.140000104904175,
        ]
    ).reshape(2, 3)
    vstr = f'np.array([{make_array_values(values)}]).reshape(2,3, order="F")'
    for n, k in fort_real_kinds.items():
        name = f"real_{n}_2d"
        v = f"{make_array_values(values,n)}"

        fort_strs.append(
            fort_param_array_multi.substitute(
                type="real", kind=n, var=name, shape="2,3", value=v
            )
        )

        py_strs.append(py_array_check.substitute(name=name, value=vstr))
        py_strs.append(py_fail.substitute(var=name, bad="np.array([1,2,3])"))

    return fort_strs, py_strs


def create_logicals():
    fort_strs = []
    py_strs = []
    fvalues = [".false.", ".true."]
    pyvalues = [False, True]

    n = "logical"
    for indv, v in enumerate(fvalues):
        name = f"logicals_{indv}"

        fort_strs.append(fort_bool_param.substitute(var=name, value=f"{v}"))

        if v == ".true.":
            py_strs.append(py_bool_true.substitute(name=name))
            py_strs.append(py_fail.substitute(var=name, bad=False))
        else:
            py_strs.append(py_bool_false.substitute(name=name))
            py_strs.append(py_fail.substitute(var=name, bad=True))

    return fort_strs, py_strs


def create_logicals_1d():
    fort_strs = []
    py_strs = []
    values = np.array([True, False, True, False])
    v = f".true., .false., .true., .false."
    vstr = f"np.array([{make_array_values(values)}])"

    n = "logical"
    name = f"logicals_0_1d"

    fort_strs.append(
        fort_bool_param_array_multi.substitute(var=name, shape="4", value=v)
    )

    py_strs.append(py_array_check.substitute(name=name, value=vstr))

    return fort_strs, py_strs


def create_logicals_2d():
    fort_strs = []
    py_strs = []
    values = np.array([True, False, True, False, True, False]).reshape(2, 3)
    v = f".true., .false., .true., .false., .true., .false."
    vstr = f'np.array([{make_array_values(values)}]).reshape(2,3, order="F")'

    n = "logical"
    name = f"logicals_0_2d"

    fort_strs.append(
        fort_bool_param_array_multi.substitute(var=name, shape="2,3", value=v)
    )

    py_strs.append(py_array_check.substitute(name=name, value=vstr))

    return fort_strs, py_strs


def create_cmplx():
    fort_strs = []
    py_strs = []

    for n, k in fort_real_kinds.items():
        values = [
            f"(-3.140000104904175_{n},-3.140000104904175_{n})",
            f"(0_{n},0_{n})",
            f"(3.140000104904175_{n},3.140000104904175_{n})",
        ]
        pyvalues = [
            "complex(-3.140000104904175,-3.140000104904175)",
            "complex(0.0,0.0)",
            "complex(3.140000104904175,3.140000104904175)",
        ]
        for indv, v in enumerate(values):
            name = f"complex_{n}_{indv}"

            fort_strs.append(
                fort_param.substitute(type="complex", kind=n, var=name, value=v)
            )

            py_strs.append(py_value_comp.substitute(name=name, value=pyvalues[indv]))
            py_strs.append(py_fail.substitute(var=name, bad=-99.9))

    return fort_strs, py_strs


def create_cmplx_array_1d():
    fort_strs = []
    py_strs = []

    for n, k in fort_real_kinds.items():
        name = f"complex_{n}_1d"
        # Kind of a component is different to the kind of the overall complex variable so skip setting the kind to make things easier
        v = f"(-3.140000104904175,-3.140000104904175), (0,0), (3.140000104904175,3.140000104904175)"
        pyvalues = "complex(-3.140000104904175,-3.140000104904175), complex(0.0,0.0), complex(3.140000104904175,3.140000104904175)"

        fort_strs.append(
            fort_param_array_multi.substitute(
                type="complex", kind=n, var=name, shape=3, value=v
            )
        )

        py_strs.append(
            py_array_check.substitute(name=name, value=f"np.array([{pyvalues}])")
        )
        py_strs.append(py_fail.substitute(var=name, bad="np.array([1,2,3])"))

    return fort_strs, py_strs


def create_cmplx_array_2d():
    fort_strs = []
    py_strs = []

    for n, k in fort_real_kinds.items():
        name = f"complex_{n}_2d"

        v = f"(-6.28000020980835,-6.28000020980835), (-3.140000104904175,-3.140000104904175), (0,0),(0,-1), (3.140000104904175,3.140000104904175),(-6.28000020980835,-6.28000020980835)"

        pyvalues = "complex(-6.28000020980835,-6.28000020980835), complex(-3.140000104904175,-3.140000104904175), complex(0.0,0.0),complex(0.0,-1.0), complex(3.140000104904175,3.140000104904175), complex(-6.28000020980835,-6.28000020980835)"

        fort_strs.append(
            fort_param_array_multi.substitute(
                type="complex", kind=n, var=name, shape="2,3", value=v
            )
        )

        py_strs.append(
            py_array_check.substitute(
                name=name, value=f'np.array([{pyvalues}]).reshape(2,3, order="F")'
            )
        )
        py_strs.append(py_fail.substitute(var=name, bad="np.array([1,2,3])"))

    return fort_strs, py_strs


def make_tests(file_f90, file_py, name, func):
    f, py = func()
    write_lines(file_f90, f, 1)
    write_line(file_py, py_proc.substitute(function=name), 1)
    write_lines(file_py, py, 2)


with open(filename_f90, "w") as file_f90, open(filename_py, "w") as file_py:
    write_line(file_f90, fort_module_start.substitute(name=modname), 0)
    write_line(file_py, py_module_start.substitute(modname=modname, libname=libname), 0)

    make_tests(file_f90, file_py, "check_ints", create_ints)
    make_tests(file_f90, file_py, "check_reals", create_reals)
    make_tests(file_f90, file_py, "check_logicals", create_logicals)

    make_tests(file_f90, file_py, "check_ints_1d", create_ints_array_1d)
    make_tests(file_f90, file_py, "check_reals_1d", create_reals_array_1d)
    make_tests(file_f90, file_py, "check_logicals_1d", create_logicals_1d)

    make_tests(file_f90, file_py, "check_ints_2d", create_ints_array_2d)
    make_tests(file_f90, file_py, "check_reals_2d", create_reals_array_2d)
    make_tests(file_f90, file_py, "check_logicals_2d", create_logicals_2d)

    make_tests(file_f90, file_py, "check_cmplx", create_cmplx)
    make_tests(file_f90, file_py, "check_cmplx_1d", create_cmplx_array_1d)
    make_tests(file_f90, file_py, "check_cmplx_2d", create_cmplx_array_2d)

    # print(fort_module_mid.substitute(),file=file_f90)
    write_line(file_f90, fort_module_end.substitute(name=modname), 0)
