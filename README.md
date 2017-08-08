![Build status](https://travis-ci.org/rjfarmer/gfort2py.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/rjfarmer/gfort2py/badge.svg?branch=master)](https://coveralls.io/github/rjfarmer/gfort2py?branch=master)

# gfort2py
Library to allow calling fortran code from python. Requires gfortran>=6.0

## Build
````bash
ipython3 setup.py install --user
````

## Using

Compile code normnally (parsing -shared -fPIC as compile options to make a shared library at the end)

````python

import gfort2py as gf

SHARED_LIB_NAME='./test_mod.so'
MOD_FILE_NAME='tester.mod'

x=gf.fFort(SHARED_LIB_NAME,MOD_FILE_NAME)

````

x now contains all variables, parameters and functions from the module (tab completable)

````python
y = x.func_name(a,b,c)
````

Will call the fortran function with varaibles a,b,c and will return the result in y,
subroutines will return a dict (possibly empty) with any intent out, inout or undefined intent variables.


Most of the time the function will copy the intent out variables before returning,
(arrays sometimes copy) and derive types are copied into a dict. To stop this from
happening (say with very large arrays/derived types)

````python
x.func_name.saveArgs(True)
y = x.func_name(a,b,c)
````

Now the return value in y is a dict of only the intent out names. The data
is stored in:

````python
x.func_name.args_out
````

Which is a dict containg ctype data, derived types can be accessed 
(lets say variable *a* as was derived type with components (x,y)) using
a decimal point followed by the component name *x*,*y*

````python
z=x.func_name.args_out`["a"]`
z.x
z.y
````


````python
x.some_var = 1
````

Sets a module variable to 1, will attempt to coerce it to the fortran type

````python
x.some_var
x.some_var.get()
````

First will print the value in some_var while get() will return the value


## Testing

````bash
ipython3 setup.py tests
````

To run unit tests

## Things that work

### Module variables

- [x] Scalars
- [x] Parameters
- [x] Characters
- [x] Explicit size arrays
- [X] Complex numbers (Scalar and parameters)
- [ ] Getting a pointer
- [x] Getting the value of a pointer
- [x] allocatable arrays
- [x] Derived types
- [ ] Nested derived types
- [ ] Functions in derived types
- [ ] Other complicated derived type stuff (abstract etc)

### Functions/subroutines

- [X] Basic calling (no arguments)
- [x] Argument passing (scalars)
- [x] Argument passing (strings)
- [X] Argument passing (explicit arrays)
- [x] Argument passing (assumed size arrays)
- [x] Argument passing (assumed shape arrays)
- [x] Argument passing (allocatable arrays)
- [x] Argument passing (derived types)
- [x] Argument intents (in, out, inout and none)
- [x] Passing characters
- [x] Pointer Arguments 
- [ ] Optional arguments
- [ ] Keyword arguments
- [ ] Generic/Elemental functions
- [ ] Functions as an argument





