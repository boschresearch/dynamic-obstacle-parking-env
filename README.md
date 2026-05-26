# Dynamic Obstacle Parking Environment (DOPE)

> This work was presented at the [ICRA 2026 Workshop](https://tum-avs.github.io/ICRA2026_Workshop/).

Dynamic Obstacle Parking Environment (DOPE) is a research framework for evaluating end-to-end autonomous parking systems in realistic **dynamic driving scenarios** using the CARLA simulator.

DOPE is inspired by the ideas and experimental setup introduced in the
[E2E Parking CARLA](https://github.com/qintonguav/e2e-parking-carla) project. However, this repository is an **independent reimplementation and substantial extension**, designed specifically for dynamic obstacle evaluation and reproducible autonomous parking research.

This project is **not a copy or redistribution** of the original repository. Instead, the system has been re-engineered and extended with new components, evaluation workflows, and scenario generation logic to support dynamic interactions between the ego vehicle and moving non-player character (NPC) vehicles.

The primary goal of DOPE is to study how autonomous parking models behave under challenging real-world conditions such as:
- Vehicles blocking the parking path
- Cars driving out of adjacent parking spots
- Vehicles approaching from the opposite direction
- Dynamic interactions during parking maneuvers
- Multi-agent interference in constrained environments

Compared to static parking benchmarks, DOPE provides a more realistic and adversarial testing environment for evaluating:
- Robustness of parking policies
- Collision avoidance behavior
- Reactive decision making
- Failure handling and recovery
- Generalization to unseen dynamic scenarios

## Features

### Dynamic NPC Modes

Four configurable NPC behavior modes are provided to simulate realistic parking scenarios:

- `none`  
  Static parking lot baseline without dynamic interference

- `drive_out`  
  NPC vehicle drives out of a parking spot during the ego vehicle’s maneuver

- `follow`  
  NPC vehicle follows or trails the ego vehicle during navigation

- `block`  
  NPC vehicle obstructs the path to the target parking location

### Supported Parking Agents

DOPE supports multiple pluggable autonomous parking backends:

- `e2e_parking_carla`  
  End-to-end parking architecture with multi-camera BEV perception and transformer-based feature fusion

- `caa_policy`  
  CAA policy-based parking agent

- `dino_diffusion_parking`  
  DINOv2 + diffusion-based autonomous parking policy

### Evaluation Metrics

The framework provides comprehensive evaluation and benchmarking metrics, including:

- Parking success rate
- Collision rate
- Final position error
- Final orientation error
- Parking completion time
- Scenario-specific failure statistics
- Dynamic obstacle interaction analysis

## Table of Contents  <!-- omit in toc -->

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [NPC Modes](#npc-modes)
- [Configuration](#configuration)
- [Contribution Guidelines](#contribution-guidelines)
- [Feedback](#feedback)
- [About](#about)
	- [Maintainers](#maintainers)
	- [Contributors](#contributors)
	- [3rd Party Licenses](#3rd-party-licenses)
	- [Used Encryption](#used-encryption)
	- [License](#license)

## Prerequisites

### Clone the Repository
```bash
git clone https://github.com/boschresearch/dynamic-obstacle-parking-env.git
cd dynamic-obstacle-parking-env
```

### Install Dependencies
```bash
conda env create -f environment.yml
conda activate dynamic-parking
```

Alternatively, run the provided setup script which also downloads and installs CARLA:
```bash
bash setup.sh
```

### Download Models
Each parking agent must be downloaded separately and placed in the `models/` directory:
- `models/e2e_parking_carla/` - E2E-Parking CARLA agent
- `models/caa_policy/` - CAA policy agent
- `models/dino_diffusion_parking/` - DINOv2 + diffusion agent

Each model directory must contain at minimum a `agent/parking_agent.py` (with a `ParkingAgent` class) and a `sensor_specs.yaml`.

### Download Pre-trained Weights
Download the model checkpoints and place them in the `weights/` directory:
- `weights/e2e_parking_carla.ckpt`
- `weights/caa_policy.ckpt`
- `weights/caa_policy_dynamics.ckpt`
- `weights/dino_diffusion_parking.ckpt`

### Install CARLA 0.9.11

Download and install [CARLA 0.9.11](https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.11.tar.gz) for Linux, or use `setup.sh` which automates this step.

## Getting Started

### Launch Simulation

**Terminal 1 - Start CARLA Server:**
```bash
cd carla
./CarlaUE4.sh -no-sound
```

**Terminal 2 - Run Evaluation:**
```bash
conda activate dynamic-parking
python benchmark/run.py -c benchmark/config/e2e_parking_carla.yaml
```

### Command Line Arguments
```bash
python benchmark/run.py --help
```

Key arguments:
- `-c`, `--config`: Path to configuration file (default: `benchmark/config/config.yaml`)
- `-v`, `--verbose`: Enable debug logging

## NPC Modes

| Mode | Description |
|------|-------------|
| `none` | Static parking lot with parked vehicles only |
| `drive_out` | An NPC vehicle drives out of a nearby parking spot during the maneuver |
| `follow` | An NPC vehicle follows the ego vehicle |
| `block` | An NPC vehicle temporarily blocks the path to the target parking spot |

Configure the NPC mode in the config file:
```yaml
evaluation:
  npc_mode: "block"  # options: ["none", "drive_out", "block", "follow"]
```

## Configuration

Main configuration files (in `benchmark/config/`):
- `e2e_parking_carla.yaml` - E2E-Parking agent settings
- `caa_policy.yaml` - CAA policy agent settings
- `dino_diffusion_parking.yaml` - DINOv2 + diffusion agent settings

Key configuration options:
```yaml
evaluation:
  agent_type: "e2e_parking_carla"  # options: ["e2e_parking_carla", "caa_policy", "dino_diffusion_parking"]
  npc_mode: "none"                 # options: ["none", "drive_out", "block", "follow"]
  model_path: "weights/e2e_parking_carla.ckpt"
  show_eva_imgs: false
  save_imgs: false
```

## Project Structure

```
dynamic-obstacle-parking-env/
├── benchmark/              # Benchmark runner and CARLA environment
│   ├── config/             # Configuration files (YAML + config loader)
│   ├── environment/        # CARLA world, sensors, NPC management, BEV rendering
│   ├── plots/              # Evaluation result plotting scripts
│   ├── resource/           # UI assets
│   └── run.py              # Main entry point
├── models/                 # Parking agent implementations
├── weights/                # Model checkpoints
├── environment.yml         # Conda environment specification
└── setup.sh                # Setup script (installs dependencies + CARLA)
```

## Open Source Software

| Package | Version | License |
|---------|---------|---------|
| [CARLA](https://github.com/carla-simulator/carla) | 0.9.11 | MIT |
| [Python](https://www.python.org/) | 3.7.16 | PSF-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | 1.13.1 | BSD-3-Clause |
| [torchvision](https://github.com/pytorch/vision) | 0.14.1 | BSD-3-Clause |
| [torchaudio](https://github.com/pytorch/audio) | 0.13.1 | BSD-3-Clause |
| [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) | 1.5.0 | Apache-2.0 |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | 0.11.4 | Apache-2.0 |
| [timm](https://github.com/huggingface/pytorch-image-models) | 0.9.7 | Apache-2.0 |
| [einops](https://github.com/arogozhnikov/einops) | 0.6.1 | MIT |
| [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) | 0.7.1 | Apache-2.0 |
| [numpy](https://github.com/numpy/numpy) | 1.21.5 | BSD-3-Clause |
| [opencv-python](https://github.com/opencv/opencv-python) | 4.8.0.76 | Apache-2.0 |
| [Pillow](https://github.com/python-pillow/Pillow) | 9.4.0 | HPND |
| [matplotlib](https://github.com/matplotlib/matplotlib) | 3.2.2 | PSF/BSD-compatible |
| [pandas](https://github.com/pandas-dev/pandas) | latest | BSD-3-Clause |
| [pygame](https://github.com/pygame/pygame) | 2.5.0 | LGPL-2.1 |
| [PyYAML](https://github.com/yaml/pyyaml) | 6.0.1 | MIT |
| [pyquaternion](https://github.com/KieranWynn/pyquaternion) | 0.9.9 | MIT |
| [loguru](https://github.com/Delgan/loguru) | 0.7.0 | MIT |
| [tqdm](https://github.com/tqdm/tqdm) | 4.66.1 | MIT / MPL-2.0 |
| [tensorboard](https://github.com/tensorflow/tensorboard) | 2.11.2 | Apache-2.0 |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | 0.16.4 | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | 0.4.0 | Apache-2.0 |
| [gymnasium](https://github.com/Farama-Foundation/Gymnasium) | latest | MIT |

## Contact

For any questions or issues, please contact [Min Hee Jo](mailto:MinHee.Jo@de.bosch.com).

## License

DOPE is open-sourced under the AGPL-3.0 license. See the LICENSE file for details.