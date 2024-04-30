FROM python:3.10.13

WORKDIR home/commonroad-geometric

# OpenGL and commonroad-drivability checker
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libxrender1 \
    wget\
  && apt-get install -y \
    build-essential \
    cmake \
    git git-lfs \
    wget \
    unzip \
    libboost-dev \
    libboost-thread-dev \
    libboost-test-dev \
    libboost-filesystem-dev \
    libeigen3-dev \
    libomp-dev \
    freeglut3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Files should be excluded through .dockerignore
COPY . .

# Build commonroad-drivability with more than one thread
ARG CMAKE_BUILD_PARALLEL_LEVEL=4
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.0.1 torch_geometric==2.3.1 \
    && pip install -e .[tests] \
    && pip cache purge

ENV SUMO_HOME=/usr/local/lib/python3.10/site-packages/sumo/
