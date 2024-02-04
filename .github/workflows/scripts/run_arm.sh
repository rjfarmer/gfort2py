#!/bin/bash

export NAME="armhf"
export TOOLCHAIN_NAME="arm-linux-gnueabihf"
export DOCKER_CONTAINER="arm32v7/ubuntu:22.04"
export PLATFORM_NAME="linux/arm/v7"

cd ~/src/gfort2py/tests
make clean
cd ~/src/gfort2py

docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker load -i docker_${TOOLCHAIN_NAME}.tar

docker start ${TOOLCHAIN_NAME}

docker run -v $(pwd):/gfort2py --platform ${PLATFORM_NAME} ${TOOLCHAIN_NAME} /bin/bash -c "cd /gfort2py && python -m pip install . && python -m pytest -v"
