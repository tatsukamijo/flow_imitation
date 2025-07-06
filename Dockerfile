# Use Ubuntu 20.04 as base for better compatibility with robotics libraries
FROM ubuntu:20.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
ENV TZ=UTC

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libfontconfig1 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo dependencies
RUN apt-get update && apt-get install -y \
    libglfw3-dev \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# pixi will manage all Python packages and dependencies

# Set working directory
WORKDIR /workspace

# Set environment variables for CUDA and MuJoCo
ENV CUDA_VISIBLE_DEVICES=0
ENV MUJOCO_EGL_DEVICE_ID=0
ENV PYOPENGL_PLATFORM=egl

# Create a user to avoid running as root
ARG USER_NAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd -g ${USER_GID} ${USER_NAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USER_NAME}

# Switch to user and install pixi for the user
USER ${USER_NAME}

# Install pixi for the user
RUN curl -fsSL https://pixi.sh/install.sh | sh

# Add pixi to PATH and enable completion for the user
RUN echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pixi completion -s bash)"' >> ~/.bashrc

# Set default command
CMD ["/bin/bash"] 