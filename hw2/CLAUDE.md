# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python environment is managed with `uv`. The venv lives at `.venv/`. Always use `.venv/bin/python` to run scripts.

```bash
# Install dependencies
uv pip install "gymnasium[box2d]" "gymnasium[other]" matplotlib numpy torch

# Run training + eval (full pipeline)
.venv/bin/python submission/code.py

# Run eval only (requires submission/agent.pth from a prior training run)
.venv/bin/python submission/code.py --eval-only
```

## Architecture — `submission/code.py`

Single-file PPO implementation. All hyperparameters are constants at the top of the file.

**Data flow:**
1. `train()` — collects rollouts into `RolloutBuffer`, calls `PPOAgent.update()` every `ROLLOUT_STEPS` steps, saves weights to `submission/agent.pth`
2. `plot_rewards()` — saves `submission/reward_curve.png`
3. `evaluate_and_record()` — loads policy, records video episodes to `submission/video/` via `gymnasium.wrappers.RecordVideo`

**Key classes:**
- `Actor` — 2-layer MLP (256 hidden, ReLU), outputs `tanh`-squashed mean + learned `log_std` parameter for a diagonal Gaussian policy
- `Critic` — same MLP structure, scalar value output
- `RolloutBuffer` — stores a single rollout; `compute_gae()` does reverse-pass GAE computation
- `PPOAgent` — holds actor, critic, single shared Adam optimizer; `update()` runs K epochs of minibatch PPO updates with clipped surrogate loss + value loss + entropy bonus

**Outputs produced in `submission/`:**
- `agent.pth` — saved actor/critic state dicts
- `reward_curve.png` — episode reward plot with 10-ep moving average
- `video/eval-episode-*.mp4` — per-episode recordings
- `eval_video.mp4` — concatenated submission video (built manually with ffmpeg)

## Known Behaviour

- The agent tends to converge to a one-leg hopping gait (~297 reward) rather than true bipedal walking. This is a known PPO local optimum on this environment with a single training env.
- `NormalizeObservation` was tested but caused training collapse on single-env setup; it is intentionally excluded.
- `ENTROPY_COEFF=0.0` also caused collapse (designed for 32 parallel envs); keep at `0.01`.
- After training, concatenate only the good-walk episodes into `eval_video.mp4` using ffmpeg (episodes that fell early are not useful).
