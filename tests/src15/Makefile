
.PHONY: all clean

FC       = gfortran
BUILDDIR = ../build

# Skip if gfortran version < 15
FC_VERSION := $(shell $(FC) -dumpversion 2>/dev/null | cut -d. -f1)
FC_GE_15   := $(shell [ -n "$(FC_VERSION)" ] && [ "$(FC_VERSION)" -ge 15 ] 2>/dev/null && echo 1 || echo 0)

FFLAGS = -ggdb -fdump-tree-original -D_FORTIFY_SOURCE=2 \
         -ffree-line-length-none -ffree-form \
         -fstack-clash-protection -fstack-protector-all -fstack-protector

SRCS = $(wildcard *.f90)

# All sources in src15 require -funsigned
FFLAGS += -funsigned

ifeq ($(OS),Windows_NT)
    uname_S := Windows
else
    uname_S := $(shell uname -s)
endif

ifneq (,$(findstring _NT,$(uname_S)))
    uname_S := Windows
endif

ifeq ($(uname_S),Windows)
    LIB_FLAGS  = -shared
    LIB_SUFFIX = dll
else ifeq ($(uname_S),Linux)
    LIB_FLAGS  = -fPIC -shared
    LIB_SUFFIX = so
else ifeq ($(uname_S),Darwin)
    LIB_FLAGS  = -dynamiclib
    LIB_SUFFIX = dylib
else
    $(error Unknown OS: $(uname_S))
endif

LIBS = $(patsubst %.f90,$(BUILDDIR)/%.$(LIB_SUFFIX),$(SRCS))

ifeq ($(FC_GE_15),0)
all:
	@echo "Skipping src15: $(FC) version $(FC_VERSION) is less than 15"
else
all: $(LIBS) | $(BUILDDIR)
	@for modfile in $(BUILDDIR)/*.mod; do \
		name=$$(basename "$$modfile" .mod); \
		gunzip -f -c "$$modfile" > "$(BUILDDIR)/$$name.gz.extract"; \
	done

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.$(LIB_SUFFIX): %.f90 | $(BUILDDIR)
	$(FC) $(FFLAGS) $(LIB_FLAGS) -cpp -J$(BUILDDIR) -o $@ $<

endif # FC_GE_15

clean:
	-rm -rf $(BUILDDIR)
