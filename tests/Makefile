
.PHONY = all clean

FC = gfortran

OPTIONS=-ggdb -fdump-tree-original -D_FORTIFY_SOURCE=2 -ffree-line-length-none  -ffree-form -fstack-clash-protection -fstack-protector-all -fstack-protector

SRCS = $(wildcard *.f90)
MODS =  $(wildcard *.mod)

GZ = $(patsubst %.mod,%.gz,$(MODS))

UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
   LIB_FLAGS = -shared
   LIB_SUFFIX = dll
   LIBS = $(patsubst %.f90,%.dll,$(SRCS))
else ifeq ($(UNAME_S),Linux)
   LIB_FLAGS = -fPIC -shared
   LIB_SUFFIX = so
   LIBS = $(patsubst %.f90,%.so,$(SRCS))
else ifeq ($(UNAME_S),Darwin)
   LIB_FLAGS = -dynamiclib
   LIB_SUFFIX = dylib
   LIBS = $(patsubst %.f90,%.dylib,$(SRCS))
else
$(error Unknown OS)
endif


all: $(LIBS) $(GZ) kinds


%.$(LIB_SUFFIX): %.f90
	$(FC) $(OPTIONS) $(LIB_FLAGS) -o  $@ $<

%.gz: %.mod
	@cp $< $@
	@gunzip -f $@ -c > $@.extract
	@rm $@

%.o : %.mod

%.mod : %.o
	@true

%.o : %.f90
	$(FC) $(OPTIONS) $(LIB_FLAGS) -o  $@ $<

kinds: kinds_check.o
	$(FC) -o kinds kinds_check.f90

clean:
	-rm -f *.o *.mod *.gz *.pyc *.f90.*.* *.$(LIB_SUFFIX) *.original *.extract *.smod *.fpy
