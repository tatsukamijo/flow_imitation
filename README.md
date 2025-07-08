# 🤖 Flow Imitation Learning

This project implements a **Conditional Flow Matching (CFM) based policy** in a **LeRobot-compatible style** for robotic imitation learning.

### 🧩 Main Components of the CFM Implementation
- `FlowPolicy`: High-level policy wrapper, normalization, and rollout logic
- `FlowModel`: Core model logic, loss computation, and action generation
- `FlowConditionalUnet1d`: 1D conditional UNet for sequence modeling
- `FlowRgbEncoder`: Image encoder (ResNet + SpatialSoftmax)
- `FlowSpatialSoftmax`: Keypoint extraction from feature maps
- `FlowSinusoidalPosEmb`: Sinusoidal positional embedding for time steps
- `FlowConv1dBlock` / `FlowConditionalResidualBlock1d`: Building blocks for the UNet
- Flow scheduler classes (`LinearFlowScheduler`, `VPFlowScheduler`): For time/weight sampling

---

## 🚀 Quick Start with Docker

### 🛠️ Prerequisites
- 🐳 Docker installed on your system
- ⚡ NVIDIA Docker runtime (for GPU support)
- 🐧 Linux environment

### 🏗️ Setup Development Environment

0. **Initialize submodules:**
   ```bash
   git submodule update --init --recursive
   ```

1. **Build the Docker image:**
   ```bash
   chmod +x BUILD_DOCKER_IMAGE.sh RUN_DOCKER_CONTAINER.sh
   ./BUILD_DOCKER_IMAGE.sh <YOUR_NAME>
   ```

2. **Run the Docker container:**
   ```bash
   ./RUN_DOCKER_CONTAINER.sh <YOUR_NAME>
   ```

The container will:
- 📂 Mount the current directory to `/workspace/flow_imitation`
- 🖥️ Set up GPU support (if available)
- ⚙️ Configure environment variables for CUDA and MuJoCo
- 🖼️ Provide X11 forwarding for GUI applications

---

## 🧑‍💻 Development Workflow

The project uses [pixi](https://pixi.sh/) for package management. Once inside the container:

1. **Install project dependencies:**
   ```bash
   cd /workspace/flow_imitation
   pixi install  # Install all dependencies
   ```

2. **Enter the virtual environment:**
   ```bash
   cd /workspace/flow_imitation
   pixi shell  # Enter pixi environment
   ```

3. **Run training:**

   - 🌀 **Train Flow Policy**
     ```bash
     pixi run train_flow_pusht
     ```
     _Runs: `python -m scripts.train --config_path configs/flow_pusht.yaml`_ inside the pixi  environment

   - 💨 **Train Diffusion Policy**
     ```bash
     pixi run train_dp_pusht
     ```
     _Runs: `python -m scripts.train --config_path configs/dp_pusht.yaml`_ inside the pixi environment

---

## 🧰 Using Pixi

- ➕ **Add new dependencies:** `pixi add package_name`
- 🐍 **Add PyPI packages:** `pixi add --pypi package_name`
- ▶️ **Run commands:** `pixi run <task>`
- 🐚 **Enter environment:** `pixi shell`
- 📋 **List tasks:** `pixi task list`

### 🎯 Defined Tasks

- 🌀 `train_flow_pusht`: Train the flow-based policy on the pusht dataset
- 💨 `train_dp_pusht`: Train the diffusion-based policy on the pusht dataset
- 📝 `notebook`: Start Jupyter Lab (if configured)

---

## 🐳 Container Management

- 🔄 **Restart existing container:**
  ```bash
  docker start -i flow_imitation_kamijo_container
  ```

- ⏹️ **Stop container:**
  ```bash
  docker stop flow_imitation_kamijo_container
  ```

- 🗑️ **Remove container:**
  ```bash
  docker rm flow_imitation_kamijo_container
  ```

---