from __future__ import print_function
import os
import select


def find_key_val(list_dicts, key, value):
    v = value.lower()
    for idx, i in enumerate(list_dicts):
        if i[key].lower() == v:
            return idx


def read_pipe(pipe_out):
    def more_data():
        r, _, _ = select.select([pipe_out], [], [], 0)
        return bool(r)
    out = b''
    while more_data():
        out += os.read(pipe_out, 1024)
    return out.decode()
