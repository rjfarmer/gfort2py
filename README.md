[![Continuous Integration](https://github.com/rjfarmer/gfort2py/actions/workflows/linux.yml/badge.svg)](https://github.com/rjfarmer/gfort2py/actions/workflows/linux.yml)
[![Coverage Status](https://coveralls.io/repos/github/rjfarmer/gfort2py/badge.svg?branch=main)](https://coveralls.io/github/rjfarmer/gfort2py?branch=main)
[![PyPI version](https://badge.fury.io/py/gfort2py.svg)](https://badge.fury.io/py/gfort2py)
[![DOI](https://zenodo.org/badge/72889348.svg)](https://zenodo.org/badge/latestdoi/72889348)
[![Python versions](https://img.shields.io/pypi/pyversions/gfort2py.svg)](https://img.shields.io/pypi/pyversions/gfort2py.svg)
[![gfortran versions](https://img.shields.io/badge/gfortran-8%7C9%7C10%7C11%7C12%7C13-blue)](https://img.shields.io/badge/gfortran-8%7C9%7C10%7C11%7C12%7C13-blue)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gfort2py)


# gfort2py
Library to allow calling Fortran code from Python. Requires gfortran>=8.0, Works with python >= 3.7

## Build
Installing locally:
````bash
python -m pip install .
````

or install via pypi
````bash
python -m pip install --upgrade --user gfort2py
````

For a full list of supportetd platforms [see the support documentation](docs/support.md).

## Why use this over other Fortran to Python translators?

gfort2py has three main aims:

1. Make it trivially easy to call Fortran code from Python
2. Minimise the number of changes needed in the Fortran code to make this work.
3. Support as many Fortran features as possible.

We achieve this by tightly coupling the code to the gfortran compiler, by doing so we can easily embed assumptions about how advanced Fortran features work which makes development easier and minimises the number of changes needed on the Fortran side. 

gfort2py use the gfortran ``mod`` files to translate your Fortran code's ABI to Python-compatible types using Python's ctype library.
By using the ``mod`` file we can determine the call signature of all procedures, components of derived types, and the size and shapes of all module-level variables. As long as your code is inside a Fortran module, no other changes are needed to your Fortran code.

The downside to this approach is that we are tightly tied to gfortran's ABI, which means we can not support other non-gfortran compilers and we do not support all versions of gfortran. When gfortran next breaks its ABI (which happens rarely, the last break was gfortran 8) we will re-evaluate our supported gfortran versions.

## Using

There are two ways to load Fortran code into Python.
Either ``fFort`` or ``compile``. The recommended way
is via ``fFort`` for interfacing with existing code,
while ``compile`` is more suitable for wrapping short snippets of Fortran code

### fFort

Your Fortran code must be inside a module and then compiled as a shared library.

On linux: 
````bash
gfortran -fPIC -shared -c file.f90
gfortran -fPIC -shared -o libfile.so file.f90
````

On MacOS: 
````bash
gfortran -dynamiclib -c file.f90
gfortran -dynamiclib -o libfile.dylib file.f90
````

On Windows:
````bash
gfortran -shared -c file.f90
gfortran -shared -o libfile.dll file.f90
````

If the shared library needs other
shared libraries you may need to set the ``LD_LIBRARY_PATH`` environment variable, and it is also recommended to run chrpath on the shared 
libraries so you can access them from anywhere.

#### Python side
````python

import gfort2py as gf

SHARED_LIB_NAME=f'./test_mod.{gf.lib_ext()}' # Handle whether on Linux, Mac, or Windows
MOD_FILE_NAME='tester.mod'

x=gf.fFort(SHARED_LIB_NAME,MOD_FILE_NAME) 

````

> **_NOTE:_** The mod data is cached to speed up re-reading the data. To control this pass cache_folder to ``fFort``.
A value of False disables caching, a string sets the folder location, while leaving the argument as None defaults to platformdirs ``user_cache_dir``



### compile

````python

import gfort2py as gf


fstr = """
            integer function myfunc(x,y)
                integer :: x,y
                myfunc = x+y
            end function myfunc
"""

x  = gf.compile(string=fstr)

````

The Fortran code can also be in a file in which case:


````python

import gfort2py as gf

x  = gf.compile(file='my_fortran_file.f90')

````

In either casee the code will be compilied into a
Fortran module and then into a shared library. Any Fortran code is valid as long as it can be inserted into a Fortran Module (Its optional whether you need to wrap things in ``module``/``end module``, if you do not then that is done automatically for you).

Additional options available for ``compile``:

- FC: str Path to gfortran compilier
- FFLAGS: str Additional Fortran compile options. This defaults to -O2.
- LDLIBS: str Any additional libraries needed to be linked in (-l)
- LDFLAGS: str Locations of addtional libraries (-L)
- ouput: str Location to save intermediate files to. Defaults to ``None`` which saves files in a temporary location. Otherwise save to the location specified.

> **_NOTE:_** The interface to compile is currently considered unstable and may change.

### Interface


``x`` now contains all variables, parameters and procedures from the module (tab completable), and is independant on how the Fortran code was loaded.


### Functions
````python
y = x.func_name(a,b,c)
````

Will call the Fortran function with variables ``a,b,c`` and returns the result in ``y``.

``y`` will be  named tuple which contains (result, args). Where ``result`` is a python object for the return value (0 if a subroutine) and where args is a dict containing all arguments passed to the procedure (both those with intent (in) which will be unchanged and intent(inout/out) which may have changed).


### Variables

````python
x.some_var = 1
````

Sets a module variable to 1, will attempt to coerce it to the Fortran type

````python
x.some_var
````

Will return a Python object 


Optional arguments that are not present should be passed as a Python ``None``.


### Arrays

Arrays should be passed as a NumPy array of the correct size and shape.


Remember that Fortran by default has 1-based array numbering while Numpy
is 0-based.


If a procedure expects an unallocated array, then pass None as the argument, otherwise pass an array of the correct shape.

### Derived types

Derived types can be set with a dict 
````python
x.my_dt={'x':1,'y':'abc'}
````

````python
y=x.my_dt
y['x']
y['y']
````
If the derived type contains another derived type then you can set a dict in a dict

````python
x.my_dt={'x':1,'y':{'a':1}}
````

When setting the components of a derived type you do not need to specify
all of them at the same time.


If you have an array of derived types

````fortran
type(my_type), dimension(5) :: my_dt
type(my_type), dimension(5,5) :: my_dt2
````

Elements can be accessed via an index:

````python
x.my_dt[0]['x']
x.my_dt2[0,0]['x']
````

You can only access one component at a time (i.e no striding [:]). Allocatable derived types are not yet supported.

Derived types that are dummy arguments to a procedure are returned as a ``fDT`` type. This is a dict-like object where the components
can only be accessed via the item interface ``['x']`` and not as attributes ``.x``.  This was done so that we do not have a name collision
between Python functions (``keys``, ``items`` etc) and any Fortran-derived type components.

You can pass a ``fDT`` as an argument to a procedure.


### Quad precision variables

Quad precision (REAL128) variables are not natively supported by Python thus we need a different way to handle them. For now that is the [pyQuadp library](https://github.com/rjfarmer/pyQuadp) which can be installed from PyPi with:

````bash
python -m pip install pyquadp
````

or from a git checkout:

````bash
python -m pip install .[qaud]
````

For more details see pyQuadp's documentation, but briefly you can create a 
quad precision variable from an ``int``, ``float``, or ``string``. On return you will receive a ``qfloat`` type. This ``qfloat`` type acts like a Python Number, so you can do things like add, multiply, subtract etc this Number with other Numbers (including non-``qfloat`` types).

We currently only support scalar Quad's and scalar complex Quad's. Arrays of
quad precision values is planned but not yet supported. Quad values can also not be returned as a function result (this is a limitation in ``ctypes`` which we have no control over). Thus a quad precision value can only occur in:

- Module variables
- Parameters
- Procedure arguments

``pyQuadp`` is currently an optional requirement, you must manually install it, it does not get auto-installed when ``gfort2py`` is installed. If you try to access a quad precision variable without ``pyQuadp`` you should get a ``TypeError``.


### Callback arguments

To pass a Fortran function as a callback argument to another function then pass the function directly:

````python

y = x.callback_function(1)

y = x.another_function(x.callback_function)

````

Currently only Fortran functions can be passed. No checking is done to ensure that the callback function has the 
correct signature to be a callback to the second function.

The callback and also be created in Python at runtime (but must be valid Fortran):

````python

fstr = """
        integer function callback(x)
            integer :: x
            write(*,*) x
            callback = 3*x
        end function callback

        """

f = gf.compile(fstr)


y = x.another_function(f.callback)

````



## Testing

````bash
python -m pip install .[test]
pytest -v
````

To run unit tests

## Things that work

### Module variables

- [x] Scalars
- [x] Parameters
- [x] Characters
- [x] Explicit size arrays
- [X] Complex numbers (Scalar and parameters)
- [x] Getting a pointer
- [x] Getting the value of a pointer
- [x] Allocatable arrays
- [x] Derived types
- [x] Nested derived types
- [X] Explicit Arrays of derived types
- [ ] Allocatable Arrays of derived types
- [ ] Procedure pointers inside derived types
- [x] Derived types with dimension(:) array components (pointer, allocatable, target)
- [x] Allocatable strings (partial)
- [x] Explicit Arrays of strings
- [x] Allocatable arrays of strings
- [ ] Classes
- [ ] Abstract interfaces
- [ ] Common blocks
- [ ] Equivalences 
- [ ] Namelists
- [ ] Quad precision variables
- [ ] function overloading 

### Procedures

- [X] Basic calling (no arguments)
- [x] Argument passing (scalars)
- [x] Argument passing (strings)
- [X] Argument passing (explicit arrays)
- [x] Argument passing (assumed size arrays)
- [x] Argument passing (assumed shape arrays)
- [x] Argument passing (allocatable arrays)
- [x] Argument passing (derived types)
- [x] Argument intents (in, out, inout and none)
- [x] Passing characters of fixed size (len=10 or len=* etc)
- [x] Functions that return a character as their result
- [x] Allocatable strings (Only for things that do not get altered inside the procedure)
- [x] Explicit arrays of strings
- [x] Allocatable arrays of strings
- [x] Pointer arguments 
- [x] Optional arguments
- [x] Value arguments
- [x] Keyword arguments
- [ ] Generic/Elemental functions
- [ ] Functions as an argument
- [x] Unary operations (arguments that involve an expression to evaluate like dimension(n+1) or dimension((2*n)+1))
- [x] Functions returning an explicit array as their result 

### Common block elements

There is no way currently to access components of a common block.


## Accessing module file data

For those wanting to explore the module file format, there is a routine ``mod_info`` available from the top-level ``gfort2py`` module:

````python
module = gf.mod_info('file.mod')
````

That will parse the mod file and convert it into an intermediate format inside ``module``.

Variables or procedures can be looked up via the item interface (I also recommend using pprint for easier viewing):

````
from pprint import pprint

pprint(module['a_variable'])
````

Accessing the list of all available components can be had via ``module.keys()``.

You can also do:
````python
module = gf.mod_info('file.mod',json=True)
module['a_variable']
````

Then when you access each component the return value will be JSON-formatted. Note you can currently only access each component as JSON not the whole module file as JSON at the moment.


## Contributing

Bug reports are of course welcome and PR's should target the main branch.

For those wanting to get more involved, adding Fortran examples to the test suite of currently untested or unsupported features would be helpful. Bonus points if you also provide a Python test case (that can be marked ``@pytest.mark.skip`` if it does not work) that demonstrates the proposed interface to the new Fortran feature. Features with test cases will move higher in the order of things I add to the code.

See [how to write a test case](docs/test_suite.md) for details on how to write test cases.

For those wanting to go further and add the new feature themselves open a bug report and we can chat about what needs doing.

## Debugging

[Debugging instructions are here](docs/debugging.md)
