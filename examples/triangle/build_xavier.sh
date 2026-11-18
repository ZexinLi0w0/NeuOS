#!/bin/bash
# classify.bin for JetPack 5 (L4T R35.4.1) on AGX Xavier; same sources and libs as ./Makefile.
set -e
g++ -std=c++11 -O2 -DUSE_OPENCV -DUSE_CUDNN -I/mnt/ssd/neuos/src/build/include -I/mnt/ssd/neuos/src/include $(pkg-config --cflags opencv4) -I/usr/local/cuda/include -I/mnt/ssd/neuos/env/include -I/mnt/ssd/neuos/prefix/include \
  classification.cpp -o classify.bin \
  -L/mnt/ssd/neuos/src/build/lib -lcaffe /mnt/ssd/neuos/src/build/lib/libcaffeproto.a \
  -L/mnt/ssd/neuos/env/lib -lglog -lgflags -lprotobuf -lboost_system -lboost_thread -lboost_filesystem \
  $(pkg-config --libs opencv4) -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand -lcudnn \
  /mnt/ssd/neuos/prefix/lib/libenergymon-odroid.a -lpthread \
  -Wl,-rpath,/mnt/ssd/neuos/src/build/lib:/mnt/ssd/neuos/env/lib
