#!/bin/bash

conda create --name pileograms python=3.8 pip
conda install --file requirements.txt

cd vendor

wget https://downloads.sourceforge.net/project/quast/quast-5.0.2.tar.gz
tar -xzf quast-5.0.2.tar.gz
cd ..

cd MUMmer3.23
make install
cd ..

cd raven
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make

