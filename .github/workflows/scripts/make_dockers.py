import subprocess
import os
import sys

platforms = [
    [
        "armhf",
        "arm-linux-gnueabihf",
        "arm32v7/ubuntu:22.04",
        "linux/arm/v7",
    ],
    [
        "ppc64le",
        "powerpc64le-linux-gnu",
        "ppc64le/ubuntu:22.04",
        "linux/ppc64le",
    ],
    [
        "s390x",
        "s390x-linux-gnu",
        "s390x/ubuntu:22.04",
        "linux/s390x",
    ],
    [
        "riscv64",
        "riscv64-linux-gnu",
        "riscv64/ubuntu:22.04",
        "linux/riscv64",
    ],
]


for p in platforms:
    NAME = p[0]
    TOOLCHAIN_NAME = p[1]
    DOCKER_CONTAINER = p[2]
    PLATFORM_NAME = p[3]

    subprocess.run(
        f"""docker run --name {TOOLCHAIN_NAME} --platform {PLATFORM_NAME} --interactive -v $(pwd):/gfort2py {DOCKER_CONTAINER} /bin/bash -c "
          apt-get update &&
          apt-get install -y cmake git python3 python-is-python3 python3-dev python3-venv python3-pip python3-numpy automake libc6-dev linux-libc-dev gcc gfortran &&
          git config --global --add safe.directory /gfort2py &&
          python -m pip install build wheel pytest dataclasses_json cpyparsing platformdirs "
          """,
        shell=True,
    )

    subprocess.run(f"docker commit {TOOLCHAIN_NAME} {TOOLCHAIN_NAME}", shell=True)
    subprocess.run(
        f"docker save -o docker_{TOOLCHAIN_NAME}.tar {TOOLCHAIN_NAME}", shell=True
    )
