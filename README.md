[![Continuous Integration](https://github.com/rjfarmer/gfort2py/actions/workflows/ci.yml/badge.svg)](https://github.com/rjfarmer/gfort2py/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/rjfarmer/gfort2py/badge.svg?branch=main)](https://coveralls.io/github/rjfarmer/gfort2py?branch=main)
[![PyPI version](https://badge.fury.io/py/gfort2py.svg)](https://badge.fury.io/py/gfort2py)
[![DOI](https://zenodo.org/badge/72889348.svg)](https://zenodo.org/badge/latestdoi/72889348)
[![Python versions](https://img.shields.io/pypi/pyversions/gfort2py.svg)](https://img.shields.io/pypi/pyversions/gfort2py.svg)
[![gfortran versions](https://img.shields.io/badge/gfortran-8%7C9%7C10%7C11%7C12-blue)](https://img.shields.io/badge/gfortran-8%7C9%7C10%7C11%7C12-blue)


# gfort2py
Library to allow calling Fortran code from Python. Requires gfortran>=8.0, Works with python >= 3.7

## Build
````bash
pip3 install -r requirements.txt
python3 setup.py install --user
````

or install via pip
````bash
python -m pip install --upgrade --user gfort2py
````

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
### Fortran side
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

If your code comes as program that does everything, then just turn the program into a function call inside a module,
then create a new file with your program that uses the module and calls the function you just made.

If the shared library needs other
shared libraries you will need to set LD_LIBRARY_PATH environment variable, and it is also recommended to run chrpath on the shared libraries so you can access them from anywhere.

### Python side
````python

import gfort2py as gf

SHARED_LIB_NAME=f'./test_mod.{gf.lib_ext()}' # Handle whether on Linux or Mac
MOD_FILE_NAME='tester.mod'

x=gf.fFort(SHARED_LIB_NAME,MOD_FILE_NAME)

````

``x`` now contains all variables, parameters and procedures from the module (tab completable). 

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


## Testing

````bash
pytest
````

or 

````bash
tox
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
- [x] Common blocks (partial)
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

### Accessing common block elements

There's no direct way to access the common block elements, but if you declare the common block as a module variable you may access the elements by their name:


````fortran
module my_mod
    implicit none
    
    integer :: a,b,c
    common /comm1/ a,b,c
    
````

Elements in the common block can thus be accessed as:

````python
x.a
x.b
x.c
````

<!-- ### Procedure pointers:

#### Procedures as arguments

Consider:

````fortran
integer function my_func(func_arg)
    integer func_arg
    
    my_func = func_arg(5)
end function my_func
    
````

Assuming that func_arg is another fortran function then we can call my_func as:


````python
x.my_func(x.func_arg) # With the function itself
````

It is left the the user to make sure that the function func_arg takes the correct inputs and returns the correct output -->


<!-- Needs readding once fixed
#### Procedure pointers

Consider a procedure like:

````fortran
procedure(my_func), pointer:: func_ptr => NULL()
````

This can be set similar to how we handle functions as arguments:


````python
x.func_ptr = x.func_arg # With the function itself
````

Its left the the user to make sure that the function func_arg takes the correct inputs and returns the correct output. If you have a function
that accepts a function pointer then its the same as if the it just accepted a function argument

If func_ptr already points a a function at compile time:

````fortran
procedure(my_func), pointer:: func_ptr => my_func
````

You must still first set it to something

````python
x.func_ptr = x.func_arg # With the function itself
```` -->

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

## Contributing

Bug reports are of course welcome and PR's should target the main branch.

For those wanting to get more involved, adding Fortran examples to the test suite of currently untested or unsupported features would be helpful. Bonus points if you also provide a Python test case (that can be marked ``@pytest.mark.skip`` if it does not work) that demonstrates the proposed interface to the new Fortran feature. Features with test cases will move higher in the order of things I add to the code.

See [how to write a test case](tests/README.md) for details on how to write test cases.

For those wanting to go further and add the new feature themselves open a bug report and we can chat about what needs doing.

## Debugging

For those wanting to dig further into ``gfort2py``.

Assuming that you loaded things as:

````python
x = gf.fFort(SO, MOD)
````

You can find out the available Fortran variables/procedures module information with:

````python
var = x._module['variable_name']
````

### Variables

For variables you can create a ``fVar`` (the object that handles converting too and from Python to Fortran) with:

````python
fvar = gf.fVar.fVar(var,x._module)
````

Note that at this point the ``fvar`` has no idea where to look for the variable. If you want to access its value in a module then

````python
fvar.in_dll(x._lib)
print(fvar.value)
````

and you can then set the value (after calling ``in_dll``) with:

````python
fvar.value = value
````

### Procedures

For a procedure you can do:

````python
proc = x._module['procedure_name']
````

and its Fortran object is:

````python
fproc = gf.fProc.fProc(x._lib,proc,x._module)
````

Calling the procedure is then:

````python
fproc(*args,**kwargs)
````

To access the arguments of a procedure then:

````python
args = [x._module[i.ref] for i in proc.args().symbol]
````

The return value of a function is accessed via:

````python
return_arg = x._module[proc.return_arg()]
````

### Derived types

To access a derived type (the type definition not an instance of the type)

````python
dt_type = x._module['Derived_name']
````

Note the capitalization, derived types start with a capital.

Similar to a variable, to access an instance of a derived type variable its:

````python
dt_var = x._module['dt_variable_name']
dt_fvar = gf.fVar.fVar(dt_var,x._module)
````
