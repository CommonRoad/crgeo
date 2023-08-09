FROM ubuntu:22.04
# opengl
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    wget\
  && rm -rf /var/lib/apt/lists/*

# install python3.10
RUN apt-get update \
  && apt-get install -y software-properties-common \
&& add-apt-repository ppa:deadsnakes/ppa && apt install -y python3.10

# install pachkage for docker
RUN apt-get install -y build-essential cmake git git-lfs wget unzip libboost-dev libboost-thread-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev libomp-dev freeglut3-dev

# install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


COPY ./environment_cpu.yml ./environment_cpu.yml
COPY ./pyproject.toml ./pyproject.toml
COPY ./setup.cfg ./setup.cfg

RUN conda env create -f environment_cpu.yml

