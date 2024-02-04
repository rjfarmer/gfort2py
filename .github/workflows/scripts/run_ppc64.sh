#!/bin/bash

export NAME="ppc64le"
export TOOLCHAIN_NAME="powerpc64le-linux-gnu"
export DOCKER_CONTAINER="ppc64le/ubuntu:22.04"
export PLATFORM_NAME="linux/ppc64le"


cd ~/src/gfort2py/tests
make clean
cd ~/src/gfort2py

docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker load -i docker_${TOOLCHAIN_NAME}.tar

docker start ${TOOLCHAIN_NAME}

docker run -v $(pwd):/gfort2py --platform ${PLATFORM_NAME} ${TOOLCHAIN_NAME} /bin/bash -c "cd /gfort2py && python -m pip install . && python -m pytest -v"
