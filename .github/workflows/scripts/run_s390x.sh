#!/bin/bash

export NAME="s390x"
export TOOLCHAIN_NAME="s390x-linux-gnu"
export DOCKER_CONTAINER="s390x/ubuntu:22.04"
export PLATFORM_NAME="linux/s390x"

cd ~/src/gfort2py/tests
make clean
cd ~/src/gfort2py

docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker load -i docker_${TOOLCHAIN_NAME}.tar

docker start ${TOOLCHAIN_NAME}

docker run -v $(pwd):/gfort2py  --ulimit core=0  --platform ${PLATFORM_NAME} ${TOOLCHAIN_NAME}  /bin/bash -c "cd /gfort2py && python -m pip install . && python -m pytest -v"
