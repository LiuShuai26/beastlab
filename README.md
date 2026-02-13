# beastlab

Python training toolkit for Beast game environments. Supports both live
editor training (via TCP) and headless training with compiled `.so` env modules.

## Install

```bash
pip install -e .              # core (numpy only)
pip install -e ".[train]"     # + Stable Baselines3 & Gymnasium
pip install -e ".[sf]"        # + Sample Factory & Gymnasium
```

## Project structure

```
beastlab/
├── beastlab/              # Python package
│   ├── client.py          # TCP client env (connects to Beast editor)
│   ├── env_loader.py      # Dynamic .so loader from envs/
│   ├── sf.py              # Sample Factory integration
│   ├── validate_env.py    # Env validator for .so modules
│   ├── models/            # Custom SF network architectures
│   └── amp/               # AMP/CALM components (placeholder)
├── envs/                  # Compiled .so env modules go here
├── configs/               # Training hyperparameter configs (YAML)
├── scripts/               # Training & export scripts
└── Projects/              # Beast project files
```

## Headless training

Build a headless env `.so` from a Beast project, then drop it into `envs/`.

### Validate an env

```bash
python -m beastlab.validate_env HumanoidEnv
python -m beastlab.validate_env HumanoidEnv --steps 500
```

### Train with Sample Factory

```bash
python scripts/train_sf.py --cfg configs/humanoid_walk.yaml \
    --module_name HumanoidEnv \
    --project_path Projects/Humanoid/Humanoid.project
```

### Train with Stable Baselines3 (editor mode)

```bash
# 1. Open scene in Beast editor, click "Training" to start MLServer
# 2. Run:
python scripts/train_sb3.py --port 5555 --timesteps 10000000
```

## Loading envs from Python

```python
from beastlab import load_beast_env

module = load_beast_env("HumanoidEnv")  # loads from envs/
env = module.HumanoidEnv()
obs, info = env.reset()
obs, rewards, terminated, truncated, info = env.step(action)
```

## Export to ONNX

```bash
python scripts/export_onnx.py --model path/to/model.zip --output model.onnx
```
