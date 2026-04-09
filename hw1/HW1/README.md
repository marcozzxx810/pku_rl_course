# HW1

This project implements a basic tabular Q-learning agent using `gymnasium`. The agent is trained on `CliffWalking-v0` and includes environment for HW1 using `LunarLander-v3`.

Core files:

* `Q_learning_example.py` 
* `lunar_landing_env.py` 

---

## Requirements

* Python >= 3.8 (recommended: 3.9 or 3.10)
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
pip install "gymnasium>=1.0.0" matplotlib numpy
pip install "gymnasium[box2d]"
```


---

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
pip install gymnasium matplotlib numpy
pip install gymnasium[box2d]
```

---

## Run

### Train Q-learning agent

```bash
python Q_learning_example.py
```

Outputs:

* Training logs
* Plots saved as PNG files

---

### Test homework environment (LunarLander)

```bash
python lunar_landing_env.py
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

## Homework Project Structure

```
.
├── Q_learning_example.py
├── README.md
├── assets
│   └── lunar_landing_example.png
└── lunar_landing_env.py
```

---

## Notes

* No GPU is required
* Recommended Python version: 3.10 for best compatibility
