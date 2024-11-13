FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="johannes.dieterich@amd.com"
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#Install packages
RUN apt upgrade -y
RUN apt update && apt install -y sudo wget gnupg2 git gcc gfortran libboost-dev bzip2 openmpi-bin flex build-essential bison libboost-all-dev vim libsqlite3-dev numactl sqlite3 gdb libgtest-dev libgflags-dev libssl-dev swig python3


#ROCm 6.2.0
RUN mkdir --parents --mode=0755 /etc/apt/keyrings
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
RUN echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.2 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
RUN apt update && apt install -y rocm-dev6.2.2 rocm-libs6.2.2

RUN pip install pytest scipy numpy==1.26.4

# Install pyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

COPY target.lst /opt/rocm/bin/

# recent cmake required for FAISS 
WORKDIR /root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.4/cmake-3.24.4.tar.gz && tar -zxvf cmake-3.24.4.tar.gz && cd cmake-3.24.4 && ./bootstrap && make -j && make install
 
# ROCm-enabled BLAS (BLIS)
WORKDIR /root
RUN git clone https://github.com/ROCmSoftwarePlatform/blis
WORKDIR blis
RUN git checkout rocm
RUN ./configure -p /opt/blis -t openmp --enable-cblas --enable-amd-offload amd64
RUN make -j
RUN make install

# ROCm-enabled LAPACK (libflame)
WORKDIR /root
RUN git clone https://github.com/ROCmSoftwarePlatform/libflame
WORKDIR libflame
RUN git checkout rocm
RUN ./configure --prefix=/opt/libflame --enable-lapack2flash --enable-vector-intrinsics=sse --enable-supermatrix --enable-hip --enable-blis-use-of-fla-malloc --enable-dynamic-build --enable-static-build --enable-verbose-make-output --enable-multithreading=pthreads --enable-lto
RUN make -j
RUN make install

ENV LD_LIBRARY_PATH=/opt/rocm/lib
RUN ldconfig

# FAISS
WORKDIR /root
RUN git clone https://github.com/facebookresearch/faiss.git
WORKDIR /root/faiss
RUN cmake -B build \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_ROCM=ON \
    -DBLAS_LIBRARIES=/opt/blis/lib/libblis.so \
    -DLAPACK_LIBRARIES=/opt/libflame/lib/libflame.so \
    -DBUILD_TESTING=ON \
    -DFAISS_ENABLE_C_API=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DBUILD_SHARED_LIBS=ON \
    #-DCMAKE_BUILD_TYPE=Release \
    #-DCMAKE_BUILD_TYPE=RelWithDebInfo \
    .
RUN make -k -C build -j$(nproc)

# Tests
RUN (cd build/faiss/python && python3 setup.py build)
RUN cp tests/common_faiss_tests.py faiss/gpu-rocm/test/
#RUN make -C build test
#RUN PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py
#RUN PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/torch_test_*.py
#RUN PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest faiss/gpu-rocm/test/test_*.py
#RUN PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest -v faiss/gpu-rocm/test/torch_test_contrib_gpu.py

# get rpd
RUN apt install -y libfmt-dev
WORKDIR /root
RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData
WORKDIR rocmProfileData
RUN make
RUN make install

#Enable if running on a system with an igpu
#ENV HIP_VISIBLE_DEVICES=0
WORKDIR /root/faiss
