.PHONY: all clean

SUBDIRS := $(wildcard */.)

all:
	@for dir in $(SUBDIRS); do \
		if [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C $$dir; \
		fi \
	done

clean:
	@for dir in $(SUBDIRS); do \
		if [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C $$dir clean; \
		fi \
	done
