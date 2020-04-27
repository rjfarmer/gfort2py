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


Line #3
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

Thus each tuple contains the name of the derived type, the module is declared in, and the id.


Line #4
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

Where each tuple is the name of the interface block, module name, and the id of each function inside the interface block



Line #5
==========================

Names of common blocks::

	common  x_int
	common  /com_mod/ x_int1
	
Shows as::

	(('__BLNK__' 2 0 0 '') ('com_mod' 3 0 0 ''))
	
Where __BLNK__ is the name given to the one unnamed common block allowed

Thus each tuple is the name of the common block, its id, unknown int, unknown int.



Line #6
==========================



Line #7
==========================



Line #8
==========================


Line #9
==========================

This is a summary of all declared variables and functions on the module. It does not include function arguments::

	('var1' 0 2 'var2' 0 3 .... )
	
Where this is the name of the variable/function/subroutine, unknown int, id of variable






