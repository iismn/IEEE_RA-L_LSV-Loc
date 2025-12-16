![header](https://capsule-render.vercel.app/api?type=rect&color=timeGradient&text=LSV-Loc&fontSize=20)

## Overview
**LSV-Loc** is a cross-modal place recognition framework for LiDAR-to-Street View localization. It leverages range image representations from various LiDAR sensors and matches them against street-view camera images using advanced deep learning techniques.

### Supported Features
- **Multi-LiDAR Support** — Compatible with various LiDAR sensors (HDL-32E, HDL-64E, OS1-32, OS1-64, OS2-32, OS2-64)
- **Cross-Modal Matching** — Efficient LiDAR-to-Camera place recognition
- **DDP Training** — Distributed Data Parallel training for multi-GPU acceleration
- **Multiple Backbones** — Support for DINOv2, CLIP, and SEA-lite models

### Supported Datasets
- **MulRan** — Multi-sensor urban dataset
- **ComplexUrban** — Complex urban driving scenarios
- **HeLiPR** — Heterogeneous LiDAR place recognition dataset
- **STheReo** — Stereo thermal dataset
- **InHouse** — Custom dataset support

## Key Components
| Module | Description |
|--------|-------------|
| **CLIP_AIO** | All-in-one CLIP-based backbone for cross-modal feature extraction |
| **match_SEA_lite** | Lightweight scene-aware encoder for efficient matching |
| **strvDataset_AIO** | Unified dataloader for street-view and range image pairs |
| **networkTool** | Training utilities including loss functions and optimizers |
| **evaluateTool** | Evaluation metrics for place recognition (Recall@K, etc.) |

## Environment Setup

### Docker Environment (Recommended)
This project requires a specific Docker environment with CUDA support:

```bash
sudo docker run -it \
    --name=SVR_Loc_NGC \
    --gpus=all \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --volume="$XAUTH:$XAUTH" \
    --runtime=nvidia \
    -v /home/$USER/Workspace_Share/:/home/$USER/Workspace/ \
    -v /home/$USER/Documents/:/home/$USER/Documents/ \
    -v /dev:/dev\
    -v /dev/shm:/dev/shm\
    --privileged\
    --net=host \
    --ipc=host \
    --pid=host \
    -p 8888:8888 \
    iismn/ubuntu_cuda_gl:LTS24-CUDA12-x86
```

### Dependencies Installation
After entering the Docker container:

```bash
git clone https://github.com/iismn/IEEE_RA-L_LSV-Loc.git
cd IEEE_RA-L_LSV-Loc
pip install -r requirements.txt
```

**Note:** For PyTorch and torchvision, please install them according to your specific CUDA version directly from [pytorch.org](https://pytorch.org).

## Quick Start

### Train
```bash
# Train with default configuration
python trainer.py --train_config strv_config

# Train with custom configuration
python trainer.py --train_config your_custom_config
```

### Evaluate Place Recognition
```bash
# Evaluate place recognition performance
python evaluate_PR.py

# Evaluate with visualization
python evaluate_VIS.py

# Evaluate PnP localization
python evaluate_PnP.py
```

## Configuration
Training configurations are located in `config/` directory. Key parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | GPU batch size | 10 |
| `epochs` | Training epochs | 50 |
| `image_size` | Input image resolution | 518 |
| `model_name` | Backbone model | match_SEA_lite |
| `threshold_dist` | Positive distance threshold (m) | 25 |
| `final_embedding_dim` | Feature embedding dimension | 512 |

## Repository Structure
```
LSV-Loc/
├── trainer.py                # Main training script
├── evaluate_PR.py            # Place recognition evaluation
├── evaluate_VIS.py           # Visualization evaluation
├── evaluate_PnP.py           # PnP localization evaluation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── config/
│   ├── strv_config.py        # Training configuration
│   └── strv_eval.py          # Evaluation configuration
├── utility/
│   ├── Backbone/             # Model backbones (CLIP, SEA-lite)
│   ├── Database/             # Dataset loaders
│   ├── Network/              # Network utilities
│   ├── Eval/                 # Evaluation tools
│   └── Etc/                  # Miscellaneous utilities
└── result/                   # Training outputs and checkpoints
```

## Citation
```bibtex
@article{lsv_loc_2025,
  title={LSV-Loc: Cross-Modal Place Recognition for LiDAR-to-Street View Localization},
  author={Sangmin Lee},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025}
}
```

## License
Released under the MIT License. See `LICENSE` for details.
