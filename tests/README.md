# How many tests do we actually need?

The main difficulty in writing gfort2py is handling the huge number of combinations of options that are available in Fortran. 
If we consider a simple integer: How many different ways can we declare an integer?

First we must consider where the integer is declared:

- module variable
- dummy argument to a procedure
- return result of a function
- component of a derived type

Then we need to consider its storage:

- Is it a parameter?
- Optional?
- pass by value?
- pointer?
- kind?

Is it an array? If so:

- Explicit sized (dimension(5))
- Explicit but runtime sized (dimension(n) where n is another integer)
- Assumed size (dimension(*))
- Assumed shape (dimension(:))
- allocatable
- pointer/target

Then of course there are the combinations of the previous options. As well, for arrays its helpful to test multiple dimensions to ensure the
ordering is correct.

In an ideal world we would test every possible combination of valid options. But in reality we test as many as we can (and that we remember to test).


## Module variables

The simplest things to test are module variables. The basic structure of these tests is that python should set the value, a Fortran procedure should be
called to check the value is correct, then the Fortran procedure should alter the variable, and then python checks the final value again. This way
we check that python can read and set these values and that Fortran can interpret what python set.

A basic example:

Fortran side

````
integer :: a_int

logical function check_a_int()

    check_a_int = .false.

    if(a_int/=99) return

    a_int = 50
end function check_a_int
````


python side:

````
    def test_a_int(self):
        # Set value
        x.a_int = 99

        # Check python set it correctly
        assert x.a_int == 99

        # Call function to check variable was set
        res = x.check_a_int()

        # Check the result of the function call
        assert res.result
    
        # Check the variable again
        assert x.a_int == 50
````

Sometimes it is better to write to stdout than the explicitly check the value in Fortran.

````
integer :: a_int

subroutine check_a_int()

    write(*,*) a_int
    
end subroutine check_a_int
````

and

````
    # Note python tests must start with test_
    def test_a_int(self,capfd):
        # Set value
        x.a_int = 99

        # Check python set it correctly
        assert x.a_int == 99

        # This captures the stdout and stderr
        out, err = capfd.readouterr()

        # Call function to check variable was set
        res = x.check_a_int()

        assert int(strip(out)) == 99
````

## Procedure arguments

Similar to module variables but now instead of procedures without arguments there should be dummy arguments. It is not needed to test various intents
simply assume everything is intent(inout)

````
logical function check_args(x)
    integer :: x

    check_args = .false.

    if(x/=99) return

    x = 50
end function check_args
````

## Function result


````
integer function check_result(x)
    integer :: x

    check_args = -1

    if(x/=99) return

    check_result = 50
end function check_result
````


## Derived types

Same as above.


# Naming things

Generally names should be unique and describe what the thing is i.e if its an integer include ``int`` if its an array include the number of dimensions (``_3d``). Things that
test similar things should have similar names, i.e all thing testing explicit arrays of integers should be named the same except for the number of dimensions.


# Skipping tests

For things that don't work yet you can use the python decorator ``@pytest.mark.skip`` to skip the test. Nothing needs to be done on the Fortran side for tests that don't work yet.

# Bug reports

Bug reports make excellent test cases. If you can reduce the problem done to its minimal working example and add it to the test suite then there is a much better chance that the bug wont be re-introduced once it has been fixed. It also makes fixing things easier if there is a minimal working example.
