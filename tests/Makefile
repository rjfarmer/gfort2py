
.PHONY = all clean

FC = gfortran

OPTIONS=-ggdb -fdump-tree-original -D_FORTIFY_SOURCE=2 -ffree-line-length-none  -ffree-form -fstack-clash-protection -fstack-protector-all -fstack-protector

SRCS = $(wildcard *.f90)
MODS =  $(wildcard *.mod)

GZ = $(patsubst %.mod,%.gz,$(MODS))

ifeq ($(OS),Windows_NT)
    uname_S := Windows
else
    uname_S := $(shell uname -s)
endif

ifneq (,$(findstring _NT,$(uname_S)))
   uname_S := Windows
endif

ifeq ($(uname_S), Windows)
   LIB_FLAGS = -shared
   LIB_SUFFIX = dll
   LIBS = $(patsubst %.f90,%.dll,$(SRCS))
else ifeq ($(uname_S),Linux)
   LIB_FLAGS = -fPIC -shared
   LIB_SUFFIX = so
   LIBS = $(patsubst %.f90,%.so,$(SRCS))
else ifeq ($(uname_S),Darwin)
   LIB_FLAGS = -dynamiclib
   LIB_SUFFIX = dylib
   LIBS = $(patsubst %.f90,%.dylib,$(SRCS))
else
$(error Unknown OS $(OS) $(uname_S))
endif


all: $(LIBS) $(GZ)

%.$(LIB_SUFFIX): %.f90
	$(FC) $(OPTIONS) $(LIB_FLAGS) -cpp -o  $@ $<

%.gz: %.mod
	@cp $< $@
	@gunzip -f $@ -c > $@.extract
	@rm $@

%.o %.mod %.smod: %.f90
	$(FC) $(OPTIONS) $(LIB_FLAGS) -cpp -o  $@ $<

clean:
	-rm -f *.o *.mod *.gz *.pyc *.f90.*.* *.$(LIB_SUFFIX) *.original *.extract *.smod *.fpy
