# SPDX-License-Identifier: GPL-2.0+

import hashlib
from pathlib import Path

# Given a code snippet turn it into a module


class Modulise:
    def __init__(self, text: str):
        self.text = text
        self._module: str | None = None

    def as_module(self) -> str:
        name = self.strhash()
        if self._module is None:
            self._module = "\n".join(
                [
                    f"module {name}",
                    "contains",
                    self.text,
                    f"end module {name}",
                ]
            )

        return self._module

    def strhash(self):
        # Must make sure hash starts with a letter as we use that in the module name
        return (
            "a"
            + hashlib.md5(self.text.encode(), usedforsecurity=False).hexdigest()[:15]
        )

    def to_file(self, filename: Path) -> None:
        module = self.as_module()
        with open(filename, "w") as f:
            f.write(module)
