#!/bin/bash

echo "Testing calculator with expressions:"
echo "2+3" | ./target/debug/fine_calc
echo ""
echo "(2+3)*4" | ./target/debug/fine_calc
echo ""
echo "10/2" | ./target/debug/fine_calc