# Supported platforms

Here we describe the support available for different platforms, where platform means the combination of the CPU architecture, OS, and Python interpreter. Support for new platforms is welcome, but will be limited by what can be run on a Github action or other free CI testing system.


## Level 1 Support

These platforms are fully supported and all features should work. We test for all combinations of supported Python (>=3.7) and gfortran (>=8) versions when available on the platform.

- x86_64/Linux/CPython
- x86_64/MacOS/CPython
- arm/MacOS/CPython (M-Chips) (Only gfortan >=9)
- x64/Windows/CPython (via Choco)


## Level 2 Support

These platforms are fully supported and all features should work. However we only test a single supported Python and gfortran version.

- x86_64/Linux/PyPy
- x64/Windows/CPython (via Cygwin)
- arm32v7/Linux/CPython
- riscv64/Linux/CPython

## Level 3 Support

These platforms are partially supported and not all features work. We only test a single supported Python and gfortran version. To find unsupported features search the test suite for the given error message.

- s390x/Linux/CPython ('Skip on big endian systems')
- ppc64le/Linux/CPython ('Skip on ppc64le systems')

# Unsupported platforms

No platforms are currently unsupported.

# Planned platforms

These platforms may work but we have not yet gotten the test suite running on these platforms. Support is planned for the future.

No additional platforms are currently planned.
