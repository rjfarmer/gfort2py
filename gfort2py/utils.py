import ctypes


def copy_array(src, dst, length, size):
    ctypes.memmove(
        dst,
        src,
        length * size,
    )
