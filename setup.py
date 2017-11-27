#!/usr/bin/env python

import os
from setuptools import setup, find_packages

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
import os
import sysconfig

PY_INCLUDE = sysconfig.get_paths()['include']

ext = [
Extension("gfort2py.parseMod.utils_cpython",["gfort2py/parseMod/utils_cpython.pyx"],include_dirs=[PY_INCLUDE]),
Extension("gfort2py.fnumpy",["gfort2py/fnumpy.pyx"],include_dirs=[np.get_include(),PY_INCLUDE])
]				


def get_version():
	with open("gfort2py/version.py") as f:
		l=f.readlines()
	return l[0].split("=")[-1].strip().replace("'","")


setup(name='gfort2py',
      version=get_version(),
      description='Python bindings for Fortran',
      license="GPLv2+",
      author='Robert Farmer',
      author_email='r.j.farmer@uva.nl',
      url='https://github.com/rjfarmer/gfort2py',
      keywords='python fortran binding',
      packages=find_packages(),
      exclude_package_data={'':['new_release.sh']},
      classifiers=[
			"Development Status :: 3 - Alpha",
			"License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
			"Programming Language :: Fortran",
		    'Programming Language :: Python :: 2.7',
		    'Programming Language :: Python :: 3',
		    'Programming Language :: Python :: 3.2',
		    'Programming Language :: Python :: 3.3',
		    'Programming Language :: Python :: 3.4',
		    'Programming Language :: Python :: 3.5',
		    'Programming Language :: Python :: 3.6',
		    'Programming Language :: Python :: 3.7',
		    'Topic :: Software Development :: Code Generators',
		    
      ],
      test_suite = 'tests',
      ext_modules=cythonize(ext),
      extras_require={
		'dev': [
			'coveralls',
			'unittest2'
			]
		}
     )
