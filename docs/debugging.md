For those wanting to dig further into ``gfort2py``.

Assuming that you loaded things as:

````python
x = gf.fFort(SO, MOD)
````

You can find out the available Fortran variables/procedures module information with:

````python
var = x._module['variable_name']
````

## Variables

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

## Procedures

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

## Derived types

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
