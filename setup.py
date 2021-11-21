
from setuptools import Extension,setup

from Cython.Build import cythonize
from setuptools.command.build_py import build_py as build_py_orig

import numpy as np
import sysconfig

PY_INCLUDE = sysconfig.get_paths()['include']

ext = [
Extension("gfort2py.parseMod.utils_cpython",
        ["gfort2py/parseMod/utils_cpython.pyx"],
        include_dirs=[PY_INCLUDE]),
Extension("gfort2py.fnumpy",
        ["gfort2py/fnumpy.pyx"],
        include_dirs=[np.get_include(),PY_INCLUDE])
]

class build_py(build_py_orig):
    def build_packages(self):
        pass

setup(
    ext_modules=cythonize(ext),
    cmdclass={'build_py': build_py},
)
