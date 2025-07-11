#!/bin/bash

# Build script for flow_imitation Docker image
# Usage: ./BUILD_DOCKER_IMAGE.sh <NAME>

set -e

# Check if name argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a name for the Docker image"
    echo "Usage: ./BUILD_DOCKER_IMAGE.sh <NAME>"
    echo "Example: ./BUILD_DOCKER_IMAGE.sh kamijo"
    exit 1
fi

NAME=$1
IMAGE_NAME="flow_imitation_${NAME}"
IMAGE_TAG="latest"

echo "=========================================="
echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="

# Build the Docker image (using existing ubuntu user)
docker build \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f Dockerfile \
    .

echo "=========================================="
echo "Docker image built successfully!"
echo "Image name: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "To run the container, use:"
echo "./RUN_DOCKER_CONTAINER.sh ${NAME}"
echo "==========================================" 