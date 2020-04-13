#!/usr/bin/env bash

cd tests

make clean
make

python3 -m unittest discover .  "*_test.py"

