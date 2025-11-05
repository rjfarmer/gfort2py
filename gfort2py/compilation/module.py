# SPDX-License-Identifier: GPL-2.0+

import hashlib


# Given a code snippet turn it into a module


class Modulise:
    def __init__(self, text: str):
        self.text = text
        self._module = None

    def as_module(self) -> list[str]:
        name = self.strhash()
        if self._module is None:
            self._module = [
                f"module {name}",
                "contains",
                self.text,
                f"end module {name}",
            ]

        return "\n".join(self._module)

    def strhash(self):
        # Must make sure hash starts with a letter as we use that in the module name
        return (
            "a"
            + hashlib.md5(
                b"".join([i.encode() for i in self.text]), usedforsecurity=False
            ).hexdigest()[:15]
        )
