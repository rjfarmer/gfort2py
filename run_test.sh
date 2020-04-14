#!/usr/bin/env bash

cd tests

make clean
make

export PYTHONFAULTHANDLER=1
python3 -m unittest discover .  "*_test.py"

