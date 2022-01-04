#!/bin/bash

rm out.log
touch out.log
for i in $(grep 'def test_' tests/dummy_arrays_test.py | sed 's/    def//g' | sed 's/(.*//');
do
    echo $i >> out.log
    pytest tests/dummy_arrays_test.py -k "$i" 2>&1 | tee -a out.log
done
