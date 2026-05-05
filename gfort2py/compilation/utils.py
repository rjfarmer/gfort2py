# SPDX-License-Identifier: GPL-2.0+

import os
import tempfile
from functools import cache
from pathlib import Path


@cache
def output_folder(output=None) -> Path:
    if output is None:
        sys_temp = Path(tempfile.gettempdir())
        temp_folder = sys_temp.joinpath("gfort2py")

        os.makedirs(temp_folder, exist_ok=True)

        return temp_folder
    else:
        return Path(output).resolve()
