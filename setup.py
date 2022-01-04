from setuptools import Extension,setup
from Cython.Build import cythonize
from setuptools.command.build_py import build_py as build_py_orig

import numpy as np
import sysconfig

PY_INCLUDE = sysconfig.get_paths()["include"]

ext = [
    Extension(
        "gfort2py.fnumpy",
        ["gfort2py/fnumpy.pyx"],
        include_dirs=[np.get_include(), PY_INCLUDE],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


class build_py(build_py_orig):
    def build_packages(self):
        pass


setup(
    ext_modules=cythonize(ext),
    cmdclass={"build_py": build_py},
)