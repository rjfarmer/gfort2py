# Supported platforms

Here we describe the support available for different platforms, where platform means the combination of the CPU architecture, OS, and Python interpreter. Support for new platforms is welcome, but will be limited by what can be run on a Github action or other free CI testing system.


## Level 1 Support

These platforms are fully supported and all features should work. We test for all combinations of supported Python and gfortran versions.

- x86/Linux/CPython
- x86/MacOS/CPython
- x86/Windows/CPython (via Choco)


## Level 2 Support

These platforms are fully supported and all features should work. However we only test a single supported Python and gfortran version.

- x86/Linux/PyPy
- x86/Windows/CPython (via Cygwin)
- arm32v7/Linux/CPython
- riscv64/Linux/CPython

## Level 3 Support

These platforms are partially supported and not all features work. We only test a single supported Python and gfortran version. To find unsupported features search the test for the given error message.

- s390x/Linux/CPython ('Skip on big endian systems')
- ppc64le/Linux/CPython ('Skip on ppc64le systems')

# Unsuporrted platforms

No platforms are currently unsupoorted.

# Planned platforms

These paaltforms may work but we have not yet gotten the test suite running on these platforms. But support is planned for the future.

- M*-chips/MacOS/CPython