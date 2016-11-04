#!/usr/bin/env python

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()




setup(name='gfort2py',
      version='0.0',
      description='Python bindings for Fortran',
      license="GPLv2+",
      author='Robert Farmer',
      author_email='rjfarmer@asu.edu',
      url='https://github.com/rjfarmer/gfort2py',
      classifiers=[
			"Development Status :: 1 - Planning",
			"License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
			"Programming Language :: Fortran",
      ]
     )
