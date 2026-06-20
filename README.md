[![Continuous Integration](https://github.com/rjfarmer/gfort2py/actions/workflows/linux.yml/badge.svg)](https://github.com/rjfarmer/gfort2py/actions/workflows/linux.yml)
[![Coverage Status](https://coveralls.io/repos/github/rjfarmer/gfort2py/badge.svg?branch=main)](https://coveralls.io/github/rjfarmer/gfort2py?branch=main)
[![PyPI version](https://badge.fury.io/py/gfort2py.svg)](https://badge.fury.io/py/gfort2py)
[![DOI](https://zenodo.org/badge/72889348.svg)](https://zenodo.org/badge/latestdoi/72889348)
[![Python versions](https://img.shields.io/pypi/pyversions/gfort2py.svg)](https://img.shields.io/pypi/pyversions/gfort2py.svg)
[![gfortran versions](https://img.shields.io/badge/gfortran-8%7C9%7C10%7C11%7C12%7C13%7C14%7C15-blue)](https://img.shields.io/badge/gfortran-8%7C9%7C10%7C11%7C12%7C13%7C14%7C15-blue)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gfort2py)


# gfort2py
Library to allow calling Fortran code from Python. Requires gfortran>=8.0, Works with python >= 3.10

## Build
Installing locally:
````bash
python -m pip install .
````

or install via pypi
````bash
python -m pip install --upgrade --user gfort2py
````

For a full list of supported platforms [see the support documentation](docs/support.md).

## Why use this over other Fortran to Python translators?

gfort2py has three main aims:

1. Make it trivially easy to call Fortran code from Python
2. Minimise the number of changes needed in the Fortran code to make this work.
3. Support as many Fortran features as possible.

We achieve this by tightly coupling the code to the gfortran compiler, by doing so we can easily embed assumptions about how advanced Fortran features work which makes development easier and minimises the number of changes needed on the Fortran side. 

gfort2py uses the gfortran ``mod`` files to translate your Fortran code's ABI to Python-compatible types using Python's ``ctypes`` library.
By using the ``mod`` file we can determine the call signature of all procedures, components of derived types, and the size and shapes of all module-level variables. As long as your code is inside a Fortran module, no other changes are needed to your Fortran code.

The downside to this approach is that we are tightly tied to gfortran's ABI, which means we cannot support other non-gfortran compilers and we do not support all versions of gfortran. When gfortran next breaks its ABI (which happens rarely, the last break was gfortran 15, before that was 8) we will re-evaluate our supported gfortran versions.

## Using

There are two ways to load Fortran code into Python:
``fFort`` or ``compile``. The recommended way
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
shared libraries you may need to set the ``LD_LIBRARY_PATH`` environment variable, and it is also recommended to run ``chrpath`` on the shared
libraries so you can access them from anywhere.

#### Python side
````python

import gfort2py as gf

SHARED_LIB_NAME=f'./test_mod.{gf.lib_ext()}' # Handle whether on Linux, Mac, or Windows
MOD_FILE_NAME='tester.mod'

x=gf.fFort(SHARED_LIB_NAME,MOD_FILE_NAME) 

````

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

In either case the code will be compiled into a
Fortran module and then into a shared library. Any Fortran code is valid as long as it can be inserted into a Fortran Module. This code MUST NOT be inside
a module.

Additional options available for ``compile``:

- FC: str Path to gfortran compiler
- FFLAGS: str Additional Fortran compile options.
- LDLIBS: str Any additional libraries needed to be linked in (-l)
- LDFLAGS: str Locations of additional libraries (-L)
- INCLUDE_FLAGS: str Include directory flags (-I)


> **_NOTE:_** The interface to compile is currently considered unstable and may change.

### Interface


``x`` now contains all variables, parameters and procedures from the module (tab completable), and is independent of how the Fortran code was loaded.


### Functions
````python
y = x.func_name(a,b,c)
````

Will call the Fortran function with variables ``a,b,c`` and returns the result in ``y``.

``y`` will be a named tuple containing ``(result, args)``. ``result`` is a Python object for the return value (``None`` for subroutines), and ``args`` is a dict containing all arguments passed to the procedure (both those with ``intent(in)``, which are unchanged, and those with ``intent(inout/out)``, which may change).


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

For CHARACTER arrays, use NumPy string dtypes that match the Fortran kind:

- ``character(kind=1, len=N)``: pass a byte-string array (for example ``dtype="S10"``).
- ``character(kind=4, len=N)``: pass a unicode array (for example ``dtype=np.str_``).

For fixed-length CHARACTER arrays, values are truncated or space-padded to the declared Fortran length.

### Unicode CHARACTER support

Unicode CHARACTER values using ``selected_char_kind('ISO_10646')`` are supported for:

- Module variables and parameters
- Scalar procedure arguments and function return values
- Explicit-size CHARACTER arrays
- Allocatable module CHARACTER arrays

Example Fortran declarations:

````fortran
integer, parameter :: CK = selected_char_kind('ISO_10646')
character(kind=CK, len=100) :: uni_set
character(kind=CK, len=100), dimension(3) :: uni_arr
character(kind=CK, len=100), dimension(:), allocatable :: uni_alloc_arr
````

Example Python usage:

````python
import numpy as np
import gfort2py as gf

x = gf.fFort("./tests/build/unicode." + gf.lib_ext(), "./tests/build/unicode.mod")

x.uni_set = "漢字Ω"
print(x.uni_set.strip())

x.uni_arr = np.array(["🚀🚀🚀", "🌍🌍🌍", "✨✨✨"], dtype=np.str_)
print(x.uni_arr)

x.alloc_uni_alloc_arr()
print(x.uni_alloc_arr)
````

Current limitation: unicode CHARACTER dummy arguments with ``dimension(:)`` (assumed-shape) or ``dimension(..)`` (assumed-rank) are not considered stable yet.

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

You can only access one component at a time (i.e no striding [:]).

Derived types that are dummy arguments to a procedure are returned as a ``fDT`` type. This is a dict-like object where the components
can only be accessed via the item interface ``['x']`` and not as attributes ``.x``.  This was done so that we do not have a name collision
between Python functions (``keys``, ``items`` etc) and any Fortran-derived type components.

You can pass a ``fDT`` as an argument to a procedure.

Arrays of derived types can be set (or allocatad) with a list of dicts:

````python
x.my_dt= [
            [{"a_int": 1}, {"a_int": 2}],
            [{"a_int": 3}, {"a_int": 4}],
        ]
````

### Allocatable characters

On Windows deallocation of allocatable characters can be flaky and crash. Thus we raise a NotImplementedError if we find one. Rather than returning it, pass
an allocatable character as an argument or a module variable to avoid this.

### Type-bound procedures

Type-bound procedures declared inside a Fortran derived type are available as
Python callables on derived-type objects.

For ``nopass`` bindings, call the method with its declared arguments:

````python
y = x.p_proc.proc_no_pass(3)
````

For ``pass(this)`` bindings, the passed object is inserted automatically.
Do not pass ``this`` explicitly:

````python
x.p_proc.proc_pass(9)
````

Type-bound methods are also resolved on extended types:

````python
x.p_proc_extend.proc_no_pass(4)
x.p_proc_extend.proc_pass(6)
````

Polymorphic ``CLASS(...)`` passed-object dummies are supported for
type-bound ``PASS`` calls.

Current limitation: procedures with polymorphic ``CLASS(...)`` array dummy
arguments (for example ``class(t), dimension(:)``) still require class-wrapper
objects produced by gfort2py. Passing plain Python placeholders (like ``[]``)
for these dummies raises ``TypeError``.


### Quad precision variables

Quad precision (INT128, REAL128, and COMPLEX128) variables are not natively supported by Python thus we need a different way to handle them. For now that is the [pyQuadp library](https://github.com/rjfarmer/pyQuadp) which can be installed from PyPi with:

````bash
python -m pip install pyquadp
````

or from a git checkout:

````bash
python -m pip install .[quad]
````

For more details see pyQuadp's documentation, but briefly you can create a 128 bit variable with either ``qint``, ``qfloat``, or ``qcmplx``. These act like a Python Number, so you can do things like add, multiply, subtract etc this Number with other Numbers (including non-128 bit types).
There are also array versions ``qiarray``, ``qarray``, ``qcarray`` that support many of Numpy array features.

Guaranteed support currently includes:

- Module variables
- Parameters
- Procedure arguments (including array arguments)

Quad functions returning a scalar are not supported. Either 
return a 1-element quad array, return the value as a dummy argument, or set a module variable with the return value.

``pyQuadp`` is currently an optional requirement, you must manually install it, it does not get auto-installed when ``gfort2py`` is installed. If you try to access a quad precision variable without ``pyQuadp`` you should get a ``TypeError``.


### Callback arguments

To pass a Fortran function as a callback argument to another function then pass the function directly:

````python

y = x.callback_function(1)

y = x.another_function(x.callback_function)

````

Currently only Fortran functions can be passed. No checking is done to ensure that the callback function has the 
correct signature to be a callback to the second function.

The callback can also be created in Python at runtime (but must be valid Fortran):

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

## Supported features

The items below are the currently supported with test-covered behavior.

### Module symbols

- Scalars, parameters, and characters
- Explicit-size and allocatable arrays
- Assumed shape arrays
- Derived types, including nested derived types
- Explicit and allocatable arrays of derived types
- Strings and unicode strings (kind=4), including allocatable arrays
- Quad module symbols and arguments when ``pyquadp`` is installed and compiler support is available
- Polymorphic class values used via the documented object wrappers

### Procedure calls

- Scalar, string, explicit-array, assumed-size, assumed-shape, assumed-rank, and allocatable-array arguments
- Derived type arguments and returns
- Pointer, optional, value, and keyword arguments
- Functions passed as callback arguments
- Type-bound procedures (including ``pass``/``nopass``)
- Unary expression-based shape resolution (for example ``dimension(n+1)``)

### Known issues/missing features

- Unicode assumed-shape/assumed-rank character dummy arrays
- Elemental/generic procedure behavior
- Common blocks
- Operator overload and procedure overloading
- Paramterised derived types


## Accessing module file data

For direct parsing and inspection of ``.mod`` files, use [https://github.com/rjfarmer/gfModParser](gfModParser).

````python
import gfModParser as gp

module = gp.Module("file.mod")
print(module.keys())
print(module["a_variable"])
````


## Contributing

Bug reports are of course welcome and PR's should target the main branch.

For those wanting to get more involved, adding Fortran examples to the test suite of currently untested or unsupported features would be helpful. Bonus points if you also provide a Python test case (that can be marked ``@pytest.mark.skip`` if it does not work) that demonstrates the proposed interface to the new Fortran feature. Features with test cases will move higher in the order of things I add to the code.

See [how to write a test case](docs/test_suite.md) for details on how to write test cases.

AI contributions are welcome, though please disclose if you did use an AI. Keep changes minimal and focused and remember to add test cases.

For those wanting to go further and add the new feature themselves open a bug report and we can chat about what needs doing.

## Debugging

[Debugging instructions are here](docs/debugging.md)
