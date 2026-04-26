# HW3 — HalfCheetah-v5 Model-Based RL

## Requirements

- Python 3.10+
- pip (latest)
- MuJoCo (installed automatically via `gymnasium[mujoco]`)

Intended to support:

- Linux (Ubuntu 22.04)
- macOS

---

## Installation

### Option 1: Conda (recommended)

```bash
conda create -n rl_hw3 python=3.10 -y
conda activate rl_hw3

pip install --upgrade pip
pip install "gymnasium[mujoco]" torch matplotlib numpy "imageio[ffmpeg]"
```

### Option 2: venv

```bash
python -m venv rl_hw3_env
```

Activate:

- Linux / macOS:

```bash
source rl_hw3_env/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install "gymnasium[mujoco]" torch matplotlib numpy "imageio[ffmpeg]"
```

---

## Run

### Task 1 — MBRL v1.5 (CEM-MPC)

```bash
cd task1
python Code.py
```

Runs 5 000 random steps, then 40 outer iterations of (model train → CEM-MPC collect 1 000 steps). Saves `best_agent.pth` / `agent.pth`, writes plots to `plots/` and `Plot.jpg`, runs a 10-episode evaluation, and records 2 video episodes to `Video.mp4`.

### Task 2 — MBPO (Ensemble + SAC)

```bash
cd task2
python Code.py
```

Runs 5 000 random steps, then 40 000 policy steps with SAC + synthetic rollouts + ensemble retraining every 1 000 steps. Saves `best_agent.pth` / `agent.pth`, writes all plots to `plots/` and `Plot.jpg`, runs a 10-episode evaluation, and records 2 video episodes to `Video.mp4`.

---

## Project Structure

```
submission/
├── task1/
│   ├── Code.py
│   ├── Plot.jpg          # reward curve
│   ├── Video.mp4         # 2 evaluation episodes (~100 s)
│   ├── best_agent.pth    # best checkpoint (by checkpoint-eval mean)
│   ├── agent.pth         # final-step checkpoint
│   ├── train.log
│   └── plots/
│       ├── reward_curve.jpg
│       ├── model_train_loss.jpg
│       ├── model_val_loss.jpg
│       ├── rollout_pred_error.jpg
│       └── diagnostics_combined.jpg
├── task2/
│   ├── Code.py
│   ├── Plot.jpg          # reward curve
│   ├── Video.mp4         # 2 evaluation episodes (~100 s)
│   ├── best_agent.pth    # best checkpoint (by checkpoint-eval mean)
│   ├── agent.pth         # final-step checkpoint
│   ├── train.log
│   └── plots/
│       ├── reward_curve.jpg
│       ├── reward_vs_env_steps.jpg
│       ├── eval_return_vs_env_steps.jpg
│       ├── model_loss.jpg
│       ├── actor_loss.jpg
│       ├── critic_loss.jpg
│       ├── q_value.jpg
│       └── diagnostics_combined.jpg
├── report.pdf
└── README.md
```

---

## Common Issues

### MuJoCo installation fails

`gymnasium[mujoco]` bundles the MuJoCo binaries via the `mujoco` Python package — no separate MuJoCo license or manual download is required. If the install fails, try:

- Ubuntu — install OpenGL and rendering libraries:

```bash
sudo apt-get install libgl1-mesa-glx libglu1-mesa libosmesa6-dev
```

- macOS — should work out of the box with Xcode Command Line Tools:

```bash
xcode-select --install
```

### Headless rendering (video recording on a server)

If running on a headless machine, set the rendering backend before running:

```bash
export MUJOCO_GL=osmesa   # software rendering
python Code.py
```

Or use `egl` if an NVIDIA GPU is available:

```bash
export MUJOCO_GL=egl
python Code.py
```

---

## Results

| Task | Algorithm | Eval episodes | Mean reward | Std |
| ---- | --------- | :-----------: | ----------: | --: |
| 1    | MBRL v1.5 (CEM-MPC) | 10 det. | 2762.36 | ±33.49 |
| 2    | MBPO (Ensemble + SAC) | 10 det. | 5740.55 | ±123.87 |

Both tasks use 45 000 real environment steps. Task 1 trains in ~37 min; Task 2 trains in ~3 h 20 min (CUDA).

---

## Notes

- A CUDA GPU is strongly recommended (especially for Task 2). CPU-only training is possible but slow.
- Both tasks save a `best_agent.pth` (best checkpoint-eval weights) alongside `agent.pth` (final weights). Evaluation always loads `best_agent.pth`.
- Random seed is fixed to 42 for reproducibility.
