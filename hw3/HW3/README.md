# HW3

This release uses `HalfCheetah-v5` for model-based reinforcement learning.

Core files:

* `HW3.md`
* `assets/HaldCheetah.png`
* `half_cheetah_env.py`
* `reward_plot_example.py`

## Requirements

* Python >= 3.8, recommended 3.10
* pip or conda
* MuJoCo through Gymnasium's `mujoco` extra

## Installation

The release package only installs dependencies required by `half_cheetah_env.py` and `reward_plot_example.py`. Algorithm-specific frameworks, video tools, and logging libraries should be selected and installed by students according to their own implementation.

### Option 1: Conda

```bash
conda create -n rl_hw3 python=3.10 -y
conda activate rl_hw3
```

```bash
python -m pip install --upgrade pip
pip install "gymnasium[mujoco]>=1.0.0" matplotlib
```

### Option 2: venv

```bash
python -m venv rl_hw3_env
```

Activate:

Linux / macOS:

```bash
source rl_hw3_env/bin/activate
```

Windows:

```bash
rl_hw3_env\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install "gymnasium[mujoco]>=1.0.0" matplotlib
```

## Verify Installation

Run:

```bash
python - <<'PY'
import gymnasium as gym
env = gym.make("HalfCheetah-v5", exclude_current_positions_from_observation=False)
obs, info = env.reset(seed=42)
print("observation shape:", obs.shape)
print("action space:", env.action_space)
env.close()
PY
```

Expected:

```text
observation shape: (18,)
action space: Box(-1.0, 1.0, (6,), float32)
```

## Rendering Setup

MuJoCo rendering uses OpenGL. For local desktop usage, the default backend is usually enough:

```bash
python half_cheetah_env.py
```

If you are running on a headless Linux server, set one of the MuJoCo OpenGL backends before recording videos.

Try EGL first if a GPU driver is available:

```bash
export MUJOCO_GL=egl
python half_cheetah_env.py
```

If EGL is not available, use OSMesa CPU rendering:

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
export MUJOCO_GL=osmesa
python half_cheetah_env.py
```

For Ubuntu desktop/windowed rendering, these packages often resolve OpenGL or GLFW errors:

```bash
sudo apt-get update
sudo apt-get install -y libglfw3 libglew2.2 libgl1-mesa-glx
```

## Common Issues

### `gymnasium.error.DependencyNotInstalled: MuJoCo`

Install the MuJoCo extra:

```bash
pip install "gymnasium[mujoco]>=1.0.0"
```

### `GLFWError` or cannot open display

You are likely running without a display. Use:

```bash
export MUJOCO_GL=egl
```

or:

```bash
export MUJOCO_GL=osmesa
```

## Run

Test the environment:

```bash
python half_cheetah_env.py
```

Run the random baseline example:

```bash
python reward_plot_example.py
```

## Submission Structure

Your zip file must contain `task1/` and `task2/`. Each task folder must include code, a plot image, and a video:

```text
HW3_ID_Name.zip
+-- task1/
|   +-- Code.py
|   +-- Plot.jpg
|   +-- Video.mp4
|   +-- optional_helper_files.py
+-- task2/
|   +-- Code.py
|   +-- Plot.jpg
|   +-- Video.mp4
|   +-- optional_helper_files.py
+-- report.pdf
```

`Code.py` should be the main entry point for each task. Additional helper files are allowed; the internal code organization is not otherwise restricted.

## Notes

* `HalfCheetah-v5` is a MuJoCo environment and is more computationally demanding than classic control tasks.
* Use shorter smoke-test runs while debugging, then run longer training for final plots and videos.
* Gymnasium's current MuJoCo environments use the `mujoco` Python package; no `mujoco-py` license setup is needed for v4/v5 environments.
* The release installation intentionally does not prescribe a deep learning framework. Students should install any extra packages required by their own Task 1 and Task 2 implementations.
