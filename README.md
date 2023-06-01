[![Continuous Integration](https://github.com/rjfarmer/gfort2py/actions/workflows/ci.yml/badge.svg)](https://github.com/rjfarmer/gfort2py/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/rjfarmer/gfort2py/badge.svg?branch=main)](https://coveralls.io/github/rjfarmer/gfort2py?branch=main)
[![PyPI version](https://badge.fury.io/py/gfort2py.svg)](https://badge.fury.io/py/gfort2py)
[![DOI](https://zenodo.org/badge/72889348.svg)](https://zenodo.org/badge/latestdoi/72889348)



# gfort2py
Library to allow calling fortran code from python. Requires gfortran>=8.0, Works with python >= 3.6

Current stable version is 1.1.7

## Build
````bash
pip3 install -r requirements.txt
python3 setup.py install --user
````

or install via pip
````bash
pip install --user gfort2py
````

## Why use this over other fortran to python translators?

gfort2py use gfortran .mod files to translate your fortran code's ABI to python compatible types using python's ctype library. The advantage here is that it can (in principle) handle anything the compiler can compile. gfort2py is almost entirely python and there are no changes needed to your fortran source code (some changes in the build process may be needed, as gfort2py needs your code compiled as a shared library). The disadvantage however is that we are tied to gfortran and can't support other compilers and may break when gfortran updates its .mod file format, though this happens rarely.


## Using
### Fortran side
Compile code with -fPIC and -shared as options, then link together as a shared lib at the end

````bash
gfortran -fPIC -shared -c file.f90
gfortran -fPIC -shared -o libfile file.f90
````
If your code comes as program that does everything, then just turn the program into a function call inside a module,
then create a new file with your program that uses the module and calls the function you just made.

If the shared library needs other
shared libraries you will need to set LD_LIBRARY_PATH environment variable, and its also recommended is to run chrpath on the 
shared libraries so you can access them from anywhere.

### Python side
````python

import gfort2py as gf

SHARED_LIB_NAME='./test_mod.so'
MOD_FILE_NAME='tester.mod'

x=gf.fFort(SHARED_LIB_NAME,MOD_FILE_NAME)

````

x now contains all variables, parameters and procedures from the module (tab completable). 

### Functions
````python
y = x.func_name(a,b,c)
````

Will call the fortran function with variables a,b,c and will return the result in y.

Y will be  named tuple which contains (result, args). Where result is a python object for the return value (0 if a subroutine) and where args is a dict containing all arguments passed to the procedure (both those with intent (in) which will be unchanged and intent(inout/out) which may have changed).


### Variables

````python
x.some_var = 1
````

Sets a module variable to 1, will attempt to coerce it to the fortran type

````python
x.some_var
````

Will return a python object 


Optional arguments that are not present should be passed as a python ``None``.


### Arrays

Arrays should be passed as a numpy array of the correct size and shape.


Remember that fortran by default has 1-based array numbering while numpy
is 0-based.


If a procedure expects an unallocted array, then pass None as the argument, otherwise pass an array of the correct shape.

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

Elements can be access via a notation similar to a slice:

````python
x.my_dt[0]['x']
x.my_dt2[0,0]['x']
````

You can only access one component at a time (i.e no striding [:]). Allocatable derived types are not yet supported.

Derived types that are dummy arguments to a procedure are returned as a ``fDT`` type. This is a dict-like object where the components
can only be accessed via the item interface ``['x']`` and not as attributes ``.x``.  This was done so that we do not have a name collision
between python functions (``keys``, ``items`` etc) and any fortran derived type components.

It is currently not possible to pass a ``fDT`` as an argument to a procedure.



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
- [ ] Arrays of strings
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
- [ ] Arrays of strings
- [x] Pointer arguments 
- [x] Optional arguments
- [x] Value arguments
- [x] Keyword arguments
- [ ] Generic/Elemental functions
- [ ] Functions as an argument
- [ ] Unary operations (arguments that involve an expresson to evaluate like dimension(n+1))

### Accessing common block elements

There's no direct way to access the common block elements, but if you declare the the common block as a module variable you may access the elements by their name:


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

### Procedure pointers:

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

It is left the the user to make sure that the function func_arg takes the correct inputs and returns the correct output


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


## Contributing

Bug reports are of course welcome and PR's should target the current dev branch.

For those wanting to get more involved, adding Fortran examples to the test suite of currently untested or unsupported features would be helpful. Bonus points if you also provide a python test case (that can be marked ``@pytest.mark.skip`` if it does not work) that demonstrates the proposed interface to the new fortran feature.

For those wanting to go further and add the new feature themselves open a bug report and we can chat about what needs doing.




