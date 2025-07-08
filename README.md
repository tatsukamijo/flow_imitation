# Flow Imitation Learning Project

This project implements and compares flow-based and diffusion-based policies for robotic imitation learning.

## Quick Start with Docker

### Prerequisites
- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- Linux environment

### Setup Development Environment

0. **Setup submodule:**
   ```bash
   git submodule update --init --recursive
   ```

1. **Build the Docker image:**
   ```bash
   chmod +x BUILD_DOCKER_IMAGE.sh RUN_DOCKER_CONTAINER.sh
   ./BUILD_DOCKER_IMAGE.sh kamijo
   ```

2. **Run the Docker container:**
   ```bash
   ./RUN_DOCKER_CONTAINER.sh kamijo
   ```

The container will:
- Mount the current directory to `/workspace/flow_imitation`
- Set up GPU support (if available)
- Configure environment variables for CUDA and MuJoCo
- Provide X11 forwarding for GUI applications

### Development Workflow

The project uses [pixi](https://pixi.sh/) for package management. Once inside the container:

1. **Install project dependencies:**
   ```bash
   cd /workspace/flow_imitation
   pixi install  # Install all dependencies from pyproject.toml
   ```

2. **Work on your flow_imitation project:**
   ```bash
   cd /workspace/flow_imitation
   pixi shell  # Enter pixi environment
   # Your code development here
   
   # Or use pixi tasks directly:
   pixi run train    # Run training (once implemented)
   pixi run eval     # Run evaluation (once implemented)
   pixi run notebook # Start Jupyter Lab (once configured)
   ```

### Using Pixi

- **Add new dependencies:** `pixi add package_name`
- **Add PyPI packages:** `pixi add --pypi package_name`
- **Run commands:** `pixi run command`
- **Enter environment:** `pixi shell`
- **List tasks:** `pixi task list`

### Container Management

- **Restart existing container:**
  ```bash
  docker start -i flow_imitation_kamijo_container
  ```

- **Stop container:**
  ```bash
  docker stop flow_imitation_kamijo_container
  ```

- **Remove container:**
  ```bash
  docker rm flow_imitation_kamijo_container
  ```

## Project Structure

```
flow_imitation/
├── src/
│   └── flow_imitation/
├── Dockerfile
├── BUILD_DOCKER_IMAGE.sh
├── RUN_DOCKER_CONTAINER.sh
├── pyproject.toml
└── README.md
```

## Next Steps

1. Implement flow-based policies using conditional flow matching
2. Implement diffusion-based policies for comparison
3. Set up training and evaluation pipelines
4. Conduct comparative experiments on robotic tasks

## Resources

- [X-IL: Exploring the Design Space of Imitation Learning Policies](https://arxiv.org/abs/2502.12330)
- [FlowNav: Combining Flow Matching and Depth Priors for Efficient Navigation](https://arxiv.org/abs/2411.09524) 
