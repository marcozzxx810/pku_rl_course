# HW1 — LunarLander DQN & DDQN

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
conda create -n rl_hw1 python=3.10 -y
conda activate rl_hw1

pip install --upgrade pip
pip install "gymnasium>=1.0.0" matplotlib numpy torch
pip install "gymnasium[box2d]" "gymnasium[other]"
```

### Option 2: venv

```bash
python -m venv rl_hw1_env
```

Activate:

* Linux / macOS:

```bash
source rl_hw1_env/bin/activate
```

* Windows:

```bash
rl_hw1_env\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install "gymnasium>=1.0.0" matplotlib numpy torch
pip install "gymnasium[box2d]" "gymnasium[other]"
```

---

## Run

### Task 1 — DQN

```bash
python task1/code.py
```

### Task 2 — DDQN

```bash
python task2/code.py
```

Each script trains for 1000 episodes and saves to the same folder:
- `plot1.jpg` — reward curve (per episode + 50-ep smoothed)
- `plot2.jpg` — Q-value curves (initial-state max Q and average max Q)
- `video.mp4` — 35s+ rendering of the trained agent

---

## Project Structure

```
submission/
├── task1/
│   ├── code.py
│   ├── plot1.jpg
│   ├── plot2.jpg
│   └── video.mp4
├── task2/
│   ├── code.py
│   ├── plot1.jpg
│   ├── plot2.jpg
│   └── video.mp4
├── README.md
└── report.pdf
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

| Algorithm | Final avg reward (last 50 ep) | Test episode reward |
|-----------|------------------------------|---------------------|
| DQN       | ~247                         | ~241                |
| DDQN      | ~231                         | ~293                |

Both agents solve the environment (reward ≥ 200).

---

## Notes

* No GPU is required
* Recommended Python version: 3.10 for best compatibility
