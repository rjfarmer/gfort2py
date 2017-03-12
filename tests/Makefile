
FC = gfortran

ifdef TRAVIS
	FC = gfortran-6
endif

all:
	$(FC) -fPIC -shared -o tester.so test_mod.f90
