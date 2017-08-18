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

ext = Extension("**/*",["**/*.pyx"],include_dirs=[np.get_include(),PY_INCLUDE])
				

setup(name='gfort2py',
      version='1.0.0',
      description='Python bindings for Fortran',
      license="GPLv2+",
      author='Robert Farmer',
      author_email='rjfarmer@asu.edu',
      url='https://github.com/rjfarmer/gfort2py',
      keywords='python fortran binding',
      packages=find_packages(),
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
		    'Intended Audience :: Science/Research',
		    'Topic :: Software Development :: Code Generators',
		    
      ],
      python_requires='>2.6, >=3.3',
      test_suite = 'tests',
      ext_modules=cythonize(ext),
      extras_require={
		'dev': [
			'coveralls',
			'unittest2'
			]
		}
     )
