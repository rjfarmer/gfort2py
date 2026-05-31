# SPDX-License-Identifier: GPL-2.0+

import numpy as np


def to_numpy_array_with_dtype(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert array to dtype, handling ctypes structured complex scalars."""
    if (
        array.dtype.fields is not None
        and "real" in array.dtype.fields
        and "imag" in array.dtype.fields
    ):
        return (array["real"] + 1j * array["imag"]).astype(dtype, copy=False)

    return array.astype(dtype)
