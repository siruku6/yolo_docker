# https://github.com/zironycho/pytorch-docker/blob/main/1.12.0/Dockerfile

# FROM python:3.9.10-slim-bullseye
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    # # To avoid `Problem with the SSL CA cert (path? access rights?)` on git clone
    # ca-certificates \
    curl \
    wget \
    git \
    vim \
    # for opencv
    # ImportError: libGL.so.1: cannot open shared object file: No such file or directory
    libgl1-mesa-dev \
    # ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
    libopencv-dev \
    # NOTE: These 3 packages concerning python3 are necessary
    #       if you use `nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04` as docker image.
    python3.9 \
    python3-pip \
    python3-setuptools \
    \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# NOTE: This is required to downgrade protobuf
# https://stackoverflow.com/questions/72899948/how-to-downgrade-protobuf
RUN pip3 install protobuf==3.20.*

WORKDIR /opt

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
