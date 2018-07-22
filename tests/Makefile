
FC = gfortran

ifdef TRAVIS
	FC = $(TRAVIS_GFORTRAN)
endif

OPTIONS=-fvar-tracking-assignments -ggdb -fdump-tree-all -fPIC -shared


all:
	$(FC) $(OPTIONS) -c test2.f90
	$(FC) $(OPTIONS) -o tester.so test_mod.f90 test2.f90
	$(FC) $(OPTIONS) -o test_mod.s -S test_mod.f90
	cp tester.mod tester.gz
	gunzip -f tester.gz
