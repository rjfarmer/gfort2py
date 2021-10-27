
.PHONY = all clean

FC = gfortran

ifdef TRAVIS
	FC = $(TRAVIS_GFORTRAN)
endif

OPTIONS=-ggdb -fdump-tree-all -fPIC -shared -D_FORTIFY_SOURCE=2 -fstack-clash-protection -fstack-protector-all -fstack-protector

SRCS = $(wildcard *.f90)
MODS =  $(wildcard *.mod)

LIBS = $(patsubst %.f90,%.so,$(SRCS))
GZ = $(patsubst %.mod,%.gz,$(MODS))

all: $(LIBS) $(GZ)


%.so: %.f90
	$(FC) $(OPTIONS) -o  $@ $<
	-mv $<.004t.original $<.original
	-rm $<.*.*

%.gz: %.mod
	-cp $< $@
	-gunzip -f $@ -c > $@.extract
	-rm $@

clean:
	-rm -f *.o *.mod *.gz *.pyc *.f90.*.* *.so *.original *.extract *.smod *.fpy
