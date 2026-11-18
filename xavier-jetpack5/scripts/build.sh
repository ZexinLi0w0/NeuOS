#!/bin/bash
# Build NeuOS (AGX variant) on Jetson AGX Xavier with JetPack 5.1.2 (L4T R35.4.1,
# CUDA 11.4, cuDNN 8.6, OpenCV 4.5.4, CMake 3.16, g++ 9.4), as run for this report.
# The Caffe dependencies come from a conda-forge env on the NVMe because the eMMC
# root had < 600 MB free; apt packages work as well.
#
# usage: bash build.sh NEUOS_SRC WORKDIR      (e.g. bash build.sh ~/NeuOS /mnt/ssd/neuos)
set -euo pipefail
SRC=$(realpath "$1"); W=$(realpath "$2")
E=$W/env           # conda env with the Caffe deps
P=$W/prefix        # energymon install prefix
mkdir -p "$W/pkgs"

# 1. dependencies (protobuf <= 3.21 and glog 0.5 keep C++11 working)
CONDA_PKGS_DIRS=$W/pkgs conda create -y -p "$E" --override-channels -c conda-forge \
  "libprotobuf=3.21" "protobuf=4.21" "glog=0.5" gflags "boost-cpp=1.74" \
  lmdb leveldb snappy hdf5 openblas numpy

# 2. energymon (AGX)
cmake -S "$SRC/energymon/AGX" -B "$SRC/energymon/AGX/build" -DCMAKE_INSTALL_PREFIX="$P"
make -C "$SRC/energymon/AGX/build" -j"$(nproc)" install

# 3. AGX overlay (README "Special Instructions for AGX")
cp -r "$SRC/AGX/include/." "$SRC/include/"
cp -r "$SRC/AGX/src/." "$SRC/src/"

# 4. Caffe (sm_72 is not in Caffe's auto list; pass it by hand)
PKG_CONFIG_PATH=$P/lib/pkgconfig cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
  -DCMAKE_PREFIX_PATH="$E" -DHDF5_ROOT="$E" -DProtobuf_PROTOC_EXECUTABLE="$E/bin/protoc" \
  -DBLAS=Open -DOpenBLAS_INCLUDE_DIR="$E/include" -DOpenBLAS_LIB="$E/lib/libopenblas.so" \
  -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=72 -DCUDA_ARCH_PTX= \
  -DBUILD_python=OFF -DBUILD_python_layer=OFF -DBUILD_docs=OFF \
  -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
  -DCMAKE_CXX_FLAGS="-I$P/include" \
  -DCMAKE_INSTALL_RPATH="$E/lib:$P/lib" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
make -C "$SRC/build" -j6 caffe

# 5. classify.bin (examples/triangle/build_xavier.sh has these paths filled in)
cd "$SRC/examples/triangle"
g++ -std=c++11 -O2 -DUSE_OPENCV -DUSE_CUDNN -I"$SRC/build/include" -I"$SRC/include" \
  $(pkg-config --cflags opencv4) -I/usr/local/cuda/include -I"$E/include" -I"$P/include" \
  classification.cpp -o classify.bin \
  -L"$SRC/build/lib" -lcaffe "$SRC/build/lib/libcaffeproto.a" \
  -L"$E/lib" -lglog -lgflags -lprotobuf -lboost_system -lboost_thread -lboost_filesystem \
  $(pkg-config --libs opencv4) -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand -lcudnn \
  "$P/lib/libenergymon-odroid.a" -lpthread -Wl,-rpath,"$SRC/build/lib:$E/lib"

# 6. CaffeNet weights and the low-rank twin (no pycaffe needed)
M=$SRC/models
curl -L -o "$M/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel" \
  http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
mkdir -p "$W/pb2" && "$E/bin/protoc" -I"$SRC/src/caffe/proto" --python_out="$W/pb2" "$SRC/src/caffe/proto/caffe.proto"
"$E/bin/python" -I "$(dirname "$(realpath "$0")")/make_lowrank_weights.py" "$W/pb2" \
  "$M/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel" \
  "$M/lowrank/caffenet_lowrank_deploy.prototxt" "$M/lowrank/config.json" \
  "$M/lowrank/caffenet_lowrank.caffemodel"
