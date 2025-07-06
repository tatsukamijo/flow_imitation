#!/bin/bash

# Run script for flow_imitation Docker container
# Usage: ./RUN_DOCKER_CONTAINER.sh <NAME>

set -e

# Check if name argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a name for the Docker container"
    echo "Usage: ./RUN_DOCKER_CONTAINER.sh <NAME>"
    echo "Example: ./RUN_DOCKER_CONTAINER.sh kamijo"
    exit 1
fi

NAME=$1
IMAGE_NAME="flow_imitation_${NAME}"
CONTAINER_NAME="flow_imitation_${NAME}_container"
IMAGE_TAG="latest"

echo "=========================================="
echo "Running Docker container: ${CONTAINER_NAME}"
echo "From image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="

# Check if image exists
if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} >/dev/null 2>&1; then
    echo "Error: Docker image ${IMAGE_NAME}:${IMAGE_TAG} not found"
    echo "Please build the image first using:"
    echo "./BUILD_DOCKER_IMAGE.sh ${NAME}"
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a | grep -q ${CONTAINER_NAME}; then
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME} >/dev/null 2>&1 || true
    docker rm ${CONTAINER_NAME} >/dev/null 2>&1 || true
fi

# Get current directory for mounting
CURRENT_DIR=$(pwd)

# Check if nvidia-docker is available for GPU support
GPU_ARGS=""
if command -v nvidia-docker &> /dev/null; then
    GPU_ARGS="--runtime=nvidia"
elif docker info | grep -q "nvidia"; then
    GPU_ARGS="--gpus all"
else
    echo "Warning: No GPU support detected. Running without GPU acceleration."
fi

# Set up X11 forwarding for potential GUI applications
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Run the container
docker run -it \
    --name ${CONTAINER_NAME} \
    ${GPU_ARGS} \
    --privileged \
    --network host \
    --env DISPLAY=$DISPLAY \
    --env XAUTHORITY=$XAUTH \
    --volume $XSOCK:$XSOCK:rw \
    --volume $XAUTH:$XAUTH:rw \
    --volume ${CURRENT_DIR}:/workspace/flow_imitation \
    --workdir /workspace/flow_imitation \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env MUJOCO_EGL_DEVICE_ID=0 \
    --env PYOPENGL_PLATFORM=egl \
    --shm-size=8g \
    ${IMAGE_NAME}:${IMAGE_TAG}

echo "=========================================="
echo "Container exited. To restart, run:"
echo "docker start -i ${CONTAINER_NAME}"
echo "Or run this script again to create a new container."
echo "==========================================" 