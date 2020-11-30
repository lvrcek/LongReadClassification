#!/bin/bash

conda create --name pileograms python=3.8 pip
conda install --file requirements.txt

cd vendor/raven
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
