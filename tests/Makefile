
.PHONY = all clean

FC = gfortran

OPTIONS=-ggdb -fdump-tree-all -D_FORTIFY_SOURCE=2 -fstack-clash-protection -fstack-protector-all -fstack-protector

SRCS = $(wildcard *.f90)
MODS =  $(wildcard *.mod)

GZ = $(patsubst %.mod,%.gz,$(MODS))

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
   # Rpaths need to resolve relative to module/test folder plus the final location of the library
   LIB_FLAGS = -fPIC -shared
   LIB_SUFFIX = so
   LIBS = $(patsubst %.f90,%.so,$(SRCS))
endif
ifeq ($(UNAME_S),Darwin)
   LIB_FLAGS = -dynamiclib
   LIB_SUFFIX = dylib
   LIBS = $(patsubst %.f90,%.dylib,$(SRCS))
endif


all: $(LIBS) $(GZ)


%.$(LIB_SUFFIX): %.f90
	$(FC) $(OPTIONS) $(LIB_FLAGS) -o  $@ $<

%.gz: %.mod
	@cp $< $@
	@gunzip -f $@ -c > $@.extract
	@rm $@

clean:
	-rm -f *.o *.mod *.gz *.pyc *.f90.*.* *.$(LIB_SUFFIX) *.original *.extract *.smod *.fpy
