# HW2 — BipedalWalker PPO

## Requirements

* Python >= 3.8 (recommended: 3.10)
* pip (latest)

Intended to support:

* Linux (Ubuntu 22.04)
* macOS
* Windows 11

---

## Installation

### Option 1: Conda (recommended)

```bash
conda create -n rl_hw2 python=3.10 -y
conda activate rl_hw2

pip install --upgrade pip
pip install "gymnasium[box2d]" "gymnasium[other]" matplotlib numpy torch
```

### Option 2: venv

```bash
python -m venv rl_hw2_env
```

Activate:

* Linux / macOS:

```bash
source rl_hw2_env/bin/activate
```

* Windows:

```bash
rl_hw2_env\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install "gymnasium[box2d]" "gymnasium[other]" matplotlib numpy torch
```

---

## Run

### Train + Evaluate (full pipeline)

```bash
python code.py
```

Trains for 2M timesteps, then records evaluation episodes.

### Evaluate only

Requires `agent.pth` from a prior training run.

```bash
python code.py --eval-only
```

---

## Project Structure

```
.
├── code.py
├── agent.pth
├── reward_curve.png
├── eval_video.mp4
├── video/
│   └── eval-episode-*.mp4
├── report.pdf
└── README.md
```

---

## Common Issues

### Box2D installation fails

Install system dependencies:

* Ubuntu:

```bash
sudo apt-get install swig build-essential python3-dev
```

* macOS:

```bash
brew install swig
```

* **Windows:**
Box2D requires C++ compilation, which often fails on Windows without the right tools.
  * **If using Conda (Recommended):** Install SWIG directly through Conda before installing Box2D:
    ```bash
    conda install swig
    ```
  * **If using venv:** You need to install the **Microsoft C++ Build Tools**. Download it from the official Microsoft website, run the installer, and make sure to check the box for **"Desktop development with C++"**.

**After installing the dependencies, try reinstalling:**
```bash
pip install "gymnasium[box2d]"
```

---

## Results

| Algorithm | Avg reward (eval) |
|-----------|-------------------|
| PPO       | ~297              |

---

## Notes

* No GPU is required
* Recommended Python version: 3.10 for best compatibility
