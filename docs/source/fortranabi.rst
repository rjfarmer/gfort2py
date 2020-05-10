##########################
gfortran ABI
##########################


This is what i have reverse engineered for gfortran's ABI, I'm focusing on gfortran > 8, as there was an ABI break then.
this also assume you are working with python's *ctype* library.


Module file format
===========================

:doc:`mod_files`


Symbol Names
===========================

All functions have variables follow the following convention:

* '__'
* module name
* '_MOD_'
* function or variable

Thus a variable  *x* in the module *my_module* is formatted as::

    __my_module_MOD_x
    
 
This is the name you must use to find the variable in the shared library (with either in_dll or getattr)


Parameters
===========================

These are stored in the mod file and can not be accessed via the shared library

Numbers
----------------------------------------------------------

Strings
----------------------------------------------------------

Arrays
----------------------------------------------------------



Integer
===========================

Integer's map to C's int class (32 or 64 bit)

Function argument
----------------------------------------------------------

These are passed as a pointer to an int. 


Real
===========================

Real's map to C's float (single precision) and double (double precision)

Function argument
----------------------------------------------------------

These are passed as a pointer to an float/double.


Complex
===========================

Use a two element struct with real, imag::

    struct complex {
		float real
		float imag
	}

Where float is either float or double depending on precision.


Strings
===========================

Fortran's strings are arrays of char's that are not null terminated.

For a fortran declaration in a module::

    character(len=5) :: my_str


That maps to c_char * 5.


Function argument
----------------------------------------------------------

These are passed as a pointer to the first character. 
Gfortran also inserts an extra hidden argument at the *end* of the argument list which is size int64 (and passed by value) which contains the size of the string


Function result
----------------------------------------------------------

Given::

    function my_func(x) result(y)
    integer :: x
    character(len=x) :: y
    
Gfortran inserts a character array and the size of the string (of type int64 and passed by value) at the *start* of the argument list. 
It seems to be up to the caller to pre-allocate the string and the length does not seem to be used.



Allocatable strings
----------------------------------------------------------
.. note::
  
  TODO



Arrays
===========================

There are two main classes of arrays explicit (where we know the size) and dummy (where we do not).


Explicit
----------------------------------------------------------

Given an array like::

    real, dimension(5) :: x
    

The array *x* is stored as pointer to the first element 


Function argument
^^^^^^^^^^^^^^^^^^^^^^^^^^

If we have either::

    real, dimension(5) :: x
    or
    real, dimension(N) :: x

We pass a pointer to the first element. No size information is added to the function arguments (like for strings). Instead the compiler propagates the size at compile time


Function result
^^^^^^^^^^^^^^^^^^^^^^^^^^

Given::

    	function my_function(N) result(array)
		integer, intent(in) :: N
		integer(dp) :: array(N+1)

Gfortran inserts a dummy array as the first argument that holds the result of the function (array).


Assumed Size
----------------------------------------------------------

::

	real,dimension(*) :: x
	
We pass a pointer to the first element, no one knows the size so be careful about out-of-bounds access



Dummy
----------------------------------------------------------

These include any array declared::

	dimension(:)
	
this includes allocatable, pointer, target, or assumed shape.

These arrays are stored in a struct called the array descriptor::

	struct array_descriptor {
		void *data 
		size_t offset
		dtype dtype
		index_t span
		dims bounds * rank
	}
	
	
	struct dtype {
		size_t elem_len
		int32 version
		byte rank
		byte type
		ushort attribute
	}
	
	struct dims{
		index_t stride
		index_t lbound
		index_t ubound
	}


index_t and size_t seem to be int64's.


data
^^^^^^^^^^^^^^^^^^^^^^^^^^
Poiter to first element of array


offset
^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the negative sum of all the strides


span
^^^^^^^^^^^^^^^^^^^^^^^^^^

Size in bytes of one element in the array (int32 => 4 bytes, double => 8 bytes)


bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an array of size the number of dimensions.


elem_len
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same as span?


version
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Appears to be 0 for now


rank
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Number of dimensions


type
^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the following

========= ==========
Type                    Value     
========= ==========
UNKNOWN        0     
INTEGER            1     
LOGICAL            2     
REAL                  3     
COMPLEX          4     
DERIVED            5     
CHARACTER      6     
CLASS                7      
PROCEDURE      8     
HOLLERITH        9     
VOID                  10     
ASSUMED          11     
========= ==========


attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Appears to be 0 for now


stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of elements you must stride to get to the next element in the same dimension.
For dimension i this is the product(strides[:i]). Thus the first dimension arrays stride is 1, 
the second dimension is N (where N is the size of the first dimension), third is (N*m where M is the size of the second dimension).

This is not in bytes, but numpy arrays want this is stride*elem_len
 


lbound
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lower bound of the array for the dimension


ubound
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Upper bound of the array for the dimension



Allocatable array
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data is a null pointer when the array is not allocated.



Derived types
===========================

Handle them similar to a C-struct.



Functions
===========================

Functions expect most arguments to be pointers (except those passed by value or pointer arguments which are pointers to pointers).
Return values are passed by value

Subroutine return None.


Intent does not matter when calling the function, it is a a compile time only check.


Procedure Pointers
===========================



Misc
===========================

value
----------------------------------------------------------
Arguments declared *value* are passed by values


optional
----------------------------------------------------------

Optional arguments that are not present should be passed a null value (in python this is None)


pointer
----------------------------------------------------------

Pointer arguments get another pointer, so its a pointer to a pointer to a variable.









