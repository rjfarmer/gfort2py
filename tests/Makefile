
FC = gfortran

ifdef TRAVIS
	FC = gfortran-6
endif

all:
	$(FC) -fPIC -shared -o tester.so test_mod.f90
	cp tester.mod tester.gz
	gunzip -f tester.gz
