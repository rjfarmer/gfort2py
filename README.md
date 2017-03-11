# gfort2py
Library to allow calling fortran code from python.

Compile code normnally (parsing -shared -fPIC to make a shared library at the end)

````python

import gfort2py as gf

SHARED_LIB_NAME='./test_mod.so'
MOD_FILE_NAME='tester.mod'

x=gf.fFort(SHARED_LIB_NAME,MOD_FILE_NAME)

````

x now contains all variables, parameters and functions from the module (tab completable)


## Testing

````bash
ipython3 testPy.py
````

To run unit tests
