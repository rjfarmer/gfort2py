#!/usr/bin/env bash

cd tests

make clean
make

export PYTHONFAULTHANDLER=1
pytest

