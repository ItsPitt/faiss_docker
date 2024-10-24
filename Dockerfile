FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="johannes.dieterich@amd.com"
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#Install packages
RUN apt upgrade -y
RUN apt update && apt install -y sudo wget gnupg2 git gcc gfortran libboost-dev bzip2 openmpi-bin flex build-essential bison libboost-all-dev vim libsqlite3-dev numactl sqlite3 gdb libgtest-dev libssl-dev

WORKDIR /root
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh

ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda create --name faiss_build python=3.11 -y
RUN conda config --set solver libmamba
RUN conda update -y -q conda
RUN conda install -y -q python=3.11 cmake make swig numpy scipy pytest gflags
RUN conda install -y -q -c conda-forge gxx_linux-64 sysroot_linux-64
RUN conda install -y -q mkl=2023 mkl-devel=2023
RUN conda init bash && . ~/.bashrc && conda activate

#ROCm 6.2.2
RUN mkdir --parents --mode=0755 /etc/apt/keyrings
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
RUN echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.2 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
RUN apt update && apt install -y rocm-dev6.2.2 rocm-libs6.2.2

# Install pyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

COPY target.lst /opt/rocm/bin/
ENV LD_LIBRARY_PATH=/opt/rocm/lib
RUN ldconfig

# symblink system libraries for HIP compiler
RUN ln -s /lib/x86_64-linux-gnu/libc.so.6 /lib64/libc.so.6
RUN ln -s /lib/x86_64-linux-gnu/libc_nonshared.a /usr/lib64/libc_nonshared.a
RUN ln -s /usr/lib/x86_64-linux-gnu/libpthread.so.0 /lib64/libpthread.so.0
RUN ln -s /root/miniconda3/x86_64-conda-linux-gnu/sysroot/usr/lib64/libpthread_nonshared.a /usr/lib64/libpthread_nonshared.a

# FAISS
WORKDIR /root
RUN git clone https://github.com/ItsPitt/faiss.git
WORKDIR /root/faiss
RUN cmake -B build \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_ROCM=ON \
    -DBUILD_TESTING=ON \
    -DFAISS_ENABLE_C_API=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    #-DCMAKE_BUILD_TYPE=Release \
    #-DCMAKE_BUILD_TYPE=RelWithDebInfo \
    .
RUN make -C build -j faiss

# make the python wrapper
RUN make -C build -j swigfaiss

RUN make -C build -j install
#RUN make -C build test

RUN (cd build/faiss/python && python3 setup.py build)
RUN cp tests/common_faiss_tests.py faiss/gpu-rocm/test/
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
ENV HIP_VISIBLE_DEVICES=0
WORKDIR /root/faiss
