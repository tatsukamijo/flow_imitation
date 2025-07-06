# Use Ubuntu 20.04 as base for better compatibility with robotics libraries
# FROM ubuntu:20.04
FROM ghcr.io/prefix-dev/pixi:0.41.1

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
ENV TZ=UTC

# Install system dependencies including build tools for robotics packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    cmake \
    build-essential \
    pkg-config \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Set environment variables for CUDA and MuJoCo
ENV CUDA_VISIBLE_DEVICES=0
ENV MUJOCO_EGL_DEVICE_ID=0
ENV PYOPENGL_PLATFORM=egl

# Use existing ubuntu user from pixi base image
USER ubuntu
# Working directory will be set by docker run

CMD ["/bin/bash"] 