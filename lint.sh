#!/bin/bash

python -m black gfort2py/ tests/
isort .
mypy -p gfort2py