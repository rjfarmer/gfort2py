##########################
Module files
##########################


Gfortran module files are just gzipped text files 


Format
==========================

An example file::


	GFORTRAN module version '15' created from file.f90   #1
	(() () () () () () () () () () () () () () () () () () () () () () () ()  
	() () ())  #2

	() #3

	() #4

	() #5

	() #6

	() #7

	(.....)  #8


	(....)  #9


Note the #Number are not present in the file, this is for easy referencing


Line #1
==========================

Metadata about the module::

	GFORTRAN module version '15' created from file.f90
	
This contains the module version '15' and the file the module came from. 

+---------------+----------------------+
| Mod Version   | Gfortran Version     |
+===============+======================+
| 14            | <8                   |
+---------------+----------------------+
| 15            | >=8                  |
+---------------+----------------------+


Line #2
==========================

This handles operator overload (+,- etc)

+----------------+----------------------+
| Bracket number | Operator             |
+================+======================+
| 1              |                      |
+----------------+----------------------+
| 2              |                      |
+----------------+----------------------+
| 3              |       plus           |
+----------------+----------------------+
| 4              |       minus          |
+----------------+----------------------+
| 5              |                      |
+----------------+----------------------+
| 6              |                      |
+----------------+----------------------+
| 7              |                      |
+----------------+----------------------+
| 9              |                      |
+----------------+----------------------+
| 10             |                      |
+----------------+----------------------+
| 11             |                      |
+----------------+----------------------+
| 12             |                      |
+----------------+----------------------+
| 13             |                      |
+----------------+----------------------+
| 14             |                      |
+----------------+----------------------+
| 15             |                      |
+----------------+----------------------+
| 16             |                      |
+----------------+----------------------+
| 17             |                      |
+----------------+----------------------+
| 18             |                      |
+----------------+----------------------+
| 19             |                      |
+----------------+----------------------+
| 20             |                      |
+----------------+----------------------+
| 21             |                      |
+----------------+----------------------+
| 22             |                      |
+----------------+----------------------+
| 23             |                      |
+----------------+----------------------+
| 24             |                      |
+----------------+----------------------+
| 25             |                      |
+----------------+----------------------+
| 26             |    equal             |
+----------------+----------------------+
| 27             |                      |
+----------------+----------------------+

Thus::

	interface operator(+)
		procedure :: my_add
	end interface 

into::

	(() () (2))

Thus its (ID of functions). Where ID is that of the named procedure(s)



Line #3
==========================

Maps unary operators::

	interface operator(.MYUNARY.)
		procedure :: my_unnary
	end interface 
	
into::

	(('myunary' '' 5))

Thus its (name, Unknown, ID)


Line #4
==========================

The name of derived types::

	type(my_dt1)
		...
	end type 

	type(my_dt2)
		...
	end type 


Shows as::

	(('my_dt1' 'module_name' 2) ('my_dt2' 'module_name' 3))

Thus each tuple contains the (name, module, ID).


Line #5
==========================

This maps between overloaded functions, thus ::

		interface my_func
			module procedure my_func_int
			module procedure my_func_real
			module procedure my_func_real_dp
			module procedure my_func_str
			module procedure my_func_cmplx
		end interface my_func

Shows as::

	(('my_func' 'module' 2 3 4 5 6))

Where each tuple is the (name, module, ID of each function inside the interface block)



Line #6
==========================

Names of common blocks::

	common  x_int
	common  /com_mod/ x_int1
	
Shows as::

	(('__BLNK__' 2 0 0 '') ('com_mod' 3 0 0 ''))
	
Where __BLNK__ is the name given to the one unnamed common block allowed

Thus each tuple is the (name, ID, saved flag)

ID appears to only be the ID of the first element of the common block.

..todo::
	what is saved flag?



Line #7
==========================

Equivalence list


Line #8
==========================

Lists all symbols


Line #9
==========================

This is a summary of all declared variables and functions on the module. It does not include function arguments::

	('var1' 0 2 'var2' 0 3 .... )
	
Where this is the (name, ambiguous flag, ID)

..todo::
	what is Ambiguous flag

