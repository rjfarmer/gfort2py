
all:
	gfortran -fPIC -shared -o tester.so test_mod.f90
#	python3 parseMod.py tester.mod
