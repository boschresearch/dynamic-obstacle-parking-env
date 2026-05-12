<!-- 
Copyright (c) Legal Entities in the BISL Group 
SPDX-License-Identifier: LicenseRef-BISL-1.0
-->

# Dynamic End-to-End Parking

[![License: BISL 1.0][license-badge]][license-docs]

This project extends the [E2E Parking CARLA](https://github.com/qintonguav/e2e-parking-carla) framework to evaluate end-to-end autonomous parking models in **dynamic scenarios** with moving NPC vehicles. The system tests how well parking models handle real-world challenges like vehicles blocking paths, driving out of parking spots, or approaching from the opposite direction.

## Features

- **Dynamic NPC Modes**: Four configurable NPC behavior modes to simulate realistic parking scenarios:
  - `none`: Static parking lot (baseline)
  - `drive_out`: NPC vehicle drives out of a parking spot during the parking maneuver
  - `opposite`: NPC vehicle approaches from the opposite direction
  - `block`: NPC vehicle blocks the path to the target parking spot

- **E2E Parking Model**: Based on the original E2E-Parking architecture with:
  - Multi-camera BEV (Bird's Eye View) perception
  - Transformer-based feature fusion
  - Tokenized control prediction (throttle, brake, steer, reverse)

- **Evaluation Metrics**: Comprehensive evaluation including success rate, collision rate, position/orientation errors, and parking time

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
git clone <repository-url>
cd dynamic-e2e-parking
```

### Install Dependencies
```bash
conda env create -f environment_rng.yml
conda activate avp-cc
```

### Download Pre-trained Models
Download the pre-trained E2E parking model checkpoint and place it in:
- `ckpt/e2e_parking.ckpt` - Pre-trained model checkpoint

### Install CARLA 0.9.16

#### Option A: GPU Cluster
```bash
cp setup_carla.sh /fs/scratch/to/your/path
cd /fs/scratch/to/your/path
bash setup_carla.sh
```

#### Option B: Local (Windows + WSL2)
1. Download [CARLA 0.9.16](https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/CARLA_0.9.16.zip)
2. Extract to `C:\carla\CARLA_0.9.16`
3. Install [DirectX End-User Runtimes](https://www.microsoft.com/en-us/download/details.aspx?id=8109)

## Getting Started

### Launch Simulation

#### Option A: GPU Cluster (LSF Scheduler)
```bash
bsub < run.bsub
```

#### Option B: Local (Windows + WSL2)

**Terminal 1 - Start CARLA Server (Windows PowerShell):**
```powershell
cd C:\carla\CARLA_0.9.16
.\CarlaUE4 -no-sound
```

**Terminal 2 - Run Evaluation (WSL2):**
```bash
conda activate avp-cc
python main.py --config config/dynamic.yaml
```

### Command Line Arguments
```bash
python main.py --help
```

Key arguments:
- `--config`: Path to configuration file (default: `config/dynamic.yaml`)
- `--npc_mode`: Override NPC mode (`none`, `drive_out`, `opposite`, `block`)
- `--show_eva_imgs`: Enable visualization of camera feeds and BEV

## NPC Modes

| Mode | Description |
|------|-------------|
| `none` | Static parking lot with parked vehicles only |
| `drive_out` | An NPC vehicle drives out of a nearby parking spot during the maneuver |
| `opposite` | An NPC vehicle approaches from the opposite direction in the driving lane |
| `block` | An NPC vehicle temporarily blocks the path to the target parking spot |

Configure the NPC mode in `config/dynamic.yaml`:
```yaml
evaluation:
  npc_mode: "block"  # options: ["none", "drive_out", "opposite", "block"]
```

## Configuration

Main configuration files:
- `config/dynamic.yaml` - Evaluation settings and NPC mode
- `config/e2e_training.yaml` - Model architecture configuration
- `config/sensor_specs.yaml` - Camera sensor specifications

## Project Structure

```
dynamic-e2e-parking/
├── agent/                  # Parking agent for inference
├── config/                 # Configuration files
├── data_generation/        # CARLA world, sensors, NPC management
├── dataset/                # Dataset loading and preprocessing  
├── e2e_model/              # E2E parking model (original architecture)
├── model/                  # Extended model components
├── tool/                   # Utilities (config, geometry, metrics)
├── main.py                 # Main entry point
└── ckpt/                   # Model checkpoints
```

## Contribution Guidelines

> Use this section to describe or link to documentation which explaining how users can make contributions to the contents of this repository. Consider adopting the [InnerSource way of solicitating and handing contributions][contributing-code].

Please read [our contribution guidelines][contribution].

## Feedback

> Consider using this section to describe how you would like other developers
> to get in contact with you or provide feedback.

## About

### Maintainers

> List the maintainers of this repository here. Consider linking to their Bosch
> Connect profile pages. Mention or link to their email as a minimum.

### Contributors

> Consider listing contributors in this section to give explicit credit. You
> could also ask contributors to add themselves in this file on their own.

### 3rd Party Licenses

> Declare all 3rd party software that is distributed with this repository along
> with their licenses. It is recommended to [use an SBoM][how-to-sbom]. If you
> do, please retain the following statement and add the SBoM file
> `sbom.spdx.json` or `sbom.cyclonedx.json` in the main directory:

Dependencies to 3rd party software are declared in the [SBoM](sbom.spdx.json).

> Alternatively, provide a list in the readme using a table like the following.

> | URL | Version | License |
> |----------|---------|-------------|
> |[Cobra](https://github.com/spf13/cobra) | 1.9.2 | [Apache 2.0 License](vendor/cobra/license.txt) |
>
> License texts of distributed dependencies should be stored in the `vendor`
> subdirectory.

### Used Encryption

> If the code in this repository **does not** contain or use encryption (other than TLS), please retain the following statement:

This repository does not contain or use encryption algorithms.

> If the code in this repository **does** contain or use encryption (other than TLS), please add the following statement to this readme:

The software in this repository uses non-custom, strong encryption (&lt;name of algorithm&gt;).
See [legal/&lt;name-of-algorithm&gt;-encryption.md] for more details.

> And provide the file `legal/<name-of-algorithm>-encryption.md` with details ([learn more][declaration-of-encrytion])

### License

[![License: BISL 1.0][license-badge]][license-docs]

We ❤️ to share this repository as [InnerSource][innersource-docs].

[license-docs]: https://docs.innersource.bosch.com/bisl-1/
[license-badge]: https://img.shields.io/badge/License-BISL--1.0-informational
[contribution]: CONTRIBUTING.md
[declaration-of-encrytion]: https://docs.innersource.bosch.com/developers-corner/start-new-project/add-innersource-metadata/#documenting-used-encryption
[contributing-code]: https://docs.innersource.bosch.com/developers-corner/run-project/ 
[how-to-sbom]: https://docs.innersource.bosch.com/developers-corner/start-new-project/how-to-sbom/
[innersource-docs]: https://docs.innersource.bosch.com/
