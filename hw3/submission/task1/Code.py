#!/usr/bin/env python3
"""
Model-Based RL v1.5 for HalfCheetah-v5
Algorithm: MBRL v1.5 — iterative data aggregation with CEM MPC
  1. Collect initial data with a random policy
  2. For each iteration:
     a. Train a learned dynamics model f(s,a)->Δs on all collected data
     b. Execute CEM MPC in the real environment
  3. Save checkpoint, reward plot, diagnostics, and a video of the trained agent

Dependencies: gymnasium[mujoco] torch numpy matplotlib imageio imageio-ffmpeg
"""

# === Hyperparameters ===
SEED                  = 42

# Environment
ENV_ID                = "HalfCheetah-v5"
OBS_DIM               = None
ACT_DIM               = None
ACTION_LOW            = None
ACTION_HIGH           = None
MAX_EPISODE_STEPS     = None
DT                    = 0.05  # frame_skip * timestep = 5 * 0.01
CTRL_COST_WEIGHT      = 0.1

# Data collection
INIT_RANDOM_STEPS     = 5000
NUM_ITERATIONS        = 40
STEPS_PER_ITER        = 1000

# Dynamics model
MODEL_HIDDEN          = 256
MODEL_LAYERS          = 3
MODEL_LR              = 3e-4
MODEL_WEIGHT_DECAY    = 1e-5
MODEL_BATCH_SIZE      = 256
MODEL_EPOCHS_PER_ITER = 40
VAL_FRACTION          = 0.1

# CEM MPC
MPC_HORIZON           = 15
MPC_NUM_CANDIDATES    = 500
MPC_CEM_ITERS         = 3
MPC_NUM_ELITES        = 50
MPC_INIT_STD          = 1.0
MPC_MIN_STD           = 0.05
UPRIGHT_BONUS_WEIGHT  = 0.05

# Evaluation / video
N_EVAL_EPISODES       = 10
EVAL_VIDEO_EPISODES   = 2     # ~50 s each at 20 fps
VIDEO_FPS             = 20
VIDEO_MAX_FRAMES      = None

import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio

SUBMISSION_DIR = Path(__file__).parent
PLOTS_DIR      = SUBMISSION_DIR / "plots"
CKPT_PATH      = SUBMISSION_DIR / "agent.pth"
BEST_CKPT_PATH = SUBMISSION_DIR / "best_agent.pth"
LOG_PATH       = SUBMISSION_DIR / "train.log"


# =============================================================================
# Smoke-test overrides
# =============================================================================
_SMOKE = dict(
    INIT_RANDOM_STEPS=300,
    NUM_ITERATIONS=2,
    STEPS_PER_ITER=200,
    MODEL_EPOCHS_PER_ITER=5,
    MPC_NUM_CANDIDATES=100,
    MPC_HORIZON=5,
    MPC_CEM_ITERS=1,
    MPC_NUM_ELITES=10,
    MAX_EPISODE_STEPS=20,
    EVAL_VIDEO_EPISODES=1,
    N_EVAL_EPISODES=2,
    VIDEO_MAX_FRAMES=20,
)


def _apply_smoke():
    g = globals()
    for k, v in _SMOKE.items():
        g[k] = v
    print("=== SMOKE MODE: shortened run for pipeline verification ===", flush=True)


# =============================================================================
# Tee for both stdout and train.log
# =============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# =============================================================================
# Replay Buffer
# =============================================================================
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 600_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self._obs  = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._act  = np.zeros((max_size, act_dim), dtype=np.float32)
        self._nobs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._rew  = np.zeros((max_size,),         dtype=np.float32)
        self._done = np.zeros((max_size,),         dtype=np.float32)

    def add_batch(self, obs, act, nobs, rew, done):
        n = len(obs)
        idxs = np.arange(self.ptr, self.ptr + n) % self.max_size
        self._obs[idxs]  = obs
        self._act[idxs]  = act
        self._nobs[idxs] = nobs
        self._rew[idxs]  = rew
        self._done[idxs] = done
        self.ptr  = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def get_all(self):
        s = self.size
        return self._obs[:s], self._act[:s], self._nobs[:s], self._rew[:s], self._done[:s]

    def train_val_split(self, val_frac: float):
        obs, act, nobs, _, _ = self.get_all()
        n = len(obs)
        idx = np.random.permutation(n)
        n_val = max(1, int(n * val_frac))
        vi, ti = idx[:n_val], idx[n_val:]
        return (obs[ti], act[ti], nobs[ti]), (obs[vi], act[vi], nobs[vi])


# =============================================================================
# Normalizer
# =============================================================================
class Normalizer:
    def __init__(self, dim: int, device: torch.device):
        self.mean = torch.zeros(dim, device=device)
        self.std  = torch.ones(dim,  device=device)
        self.device = device

    def fit(self, data: np.ndarray):
        self.mean = torch.tensor(data.mean(0), dtype=torch.float32, device=self.device)
        self.std  = torch.tensor(data.std(0) + 1e-8, dtype=torch.float32, device=self.device)

    def norm(self, x):   return (x - self.mean) / self.std
    def unnorm(self, x): return x * self.std + self.mean


# =============================================================================
# Dynamics Model — predicts state delta Δs = s_next - s
# =============================================================================
class DynamicsModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, n_layers: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(obs_dim + act_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, obs_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


# =============================================================================
# HW3-spec analytic reward
# =============================================================================
def compute_reward_torch(s, a, s_next):
    # obs[0] is the torso x-position because the env includes current positions.
    forward_reward = (s_next[..., 0] - s[..., 0]) / DT
    ctrl_cost      = CTRL_COST_WEIGHT * (a ** 2).sum(dim=-1)
    return forward_reward - ctrl_cost


def planning_objective(s, a, s_next):
    base = compute_reward_torch(s, a, s_next)
    upright_bonus = -UPRIGHT_BONUS_WEIGHT * (s_next[..., 2] ** 2)
    return base + upright_bonus


# =============================================================================
# CEM MPC — fully batched on torch
# =============================================================================
@torch.no_grad()
def mpc_action_cem(state, model, obs_norm, delta_norm, device,
                   action_low_t, action_high_t,
                   horizon: Optional[int] = None,
                   n_candidates: Optional[int] = None,
                   cem_iters: Optional[int] = None,
                   n_elites: Optional[int] = None,
                   planning_score_fn: Optional[Callable] = None):
    """CEM MPC: sample action sequences, refit on elites, return the first action."""
    if ACT_DIM is None:
        raise RuntimeError("ACT_DIM is not initialized; call _detect_env_dims() first.")

    K = MPC_NUM_CANDIDATES if n_candidates is None else n_candidates
    H = MPC_HORIZON        if horizon      is None else horizon
    I = MPC_CEM_ITERS      if cem_iters    is None else cem_iters
    E = MPC_NUM_ELITES     if n_elites     is None else n_elites
    E = min(E, K)
    score = planning_score_fn or compute_reward_torch

    low = action_low_t.view(1, 1, -1)
    high = action_high_t.view(1, 1, -1)
    mid = (action_low_t + action_high_t) * 0.5
    half = (action_high_t - action_low_t) * 0.5
    mean = mid.expand(H, -1).clone()
    std = (half * MPC_INIT_STD).expand(H, -1).clone()

    s_init = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).expand(K, -1)

    for _ in range(I):
        noise = torch.randn(K, H, ACT_DIM, device=device)
        acts = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        acts = torch.max(torch.min(acts, high), low)

        cur_s = s_init.clone()
        total_r = torch.zeros(K, device=device)
        for h in range(H):
            a = acts[:, h, :]
            delta = delta_norm.unnorm(model(obs_norm.norm(cur_s), a))
            next_s = cur_s + delta
            total_r += score(cur_s, a, next_s)
            cur_s = next_s

        _, elite_idx = torch.topk(total_r, E)
        elites = acts[elite_idx]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0, unbiased=False).clamp(min=MPC_MIN_STD)

    return mean[0].cpu().numpy()


# =============================================================================
# Model training
# =============================================================================
def train_model(model, optimizer, buffer, obs_norm, delta_norm, device):
    train_data, val_data = buffer.train_val_split(VAL_FRACTION)
    t_obs, t_act, t_nobs = train_data
    v_obs, v_act, v_nobs = val_data
    t_delta = t_nobs - t_obs
    v_delta = v_nobs - v_obs

    obs_norm.fit(t_obs)
    delta_norm.fit(t_delta)

    def to_t(x): return torch.tensor(x, dtype=torch.float32, device=device)
    to, ta, td = to_t(t_obs), to_t(t_act), to_t(t_delta)
    vo, va, vd = to_t(v_obs), to_t(v_act), to_t(v_delta)

    n = len(to)
    train_losses: List[float] = []
    model.train()
    for _ in range(MODEL_EPOCHS_PER_ITER):
        perm = torch.randperm(n, device=device)
        ep_loss = ep_steps = 0
        for start in range(0, n, MODEL_BATCH_SIZE):
            b = perm[start: start + MODEL_BATCH_SIZE]
            pred = model(obs_norm.norm(to[b]), ta[b])
            loss = ((pred - delta_norm.norm(td[b])) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            ep_steps += 1
        train_losses.append(ep_loss / max(ep_steps, 1))

    model.eval()
    with torch.no_grad():
        pred_v  = delta_norm.unnorm(model(obs_norm.norm(vo), va))
        val_mse = ((pred_v - vd) ** 2).mean().item()

    return train_losses, val_mse


# =============================================================================
# Environment helpers
# =============================================================================
def make_env(render_mode: Optional[str] = None) -> gym.Env:
    kwargs = dict(
        render_mode=render_mode,
        exclude_current_positions_from_observation=False,
    )
    if MAX_EPISODE_STEPS is not None:
        kwargs["max_episode_steps"] = MAX_EPISODE_STEPS
    return gym.make(ENV_ID, **kwargs)


def _detect_env_dims():
    global OBS_DIM, ACT_DIM, ACTION_LOW, ACTION_HIGH
    if (OBS_DIM is not None and ACT_DIM is not None and
            ACTION_LOW is not None and ACTION_HIGH is not None):
        return

    env = make_env()
    try:
        OBS_DIM = int(env.observation_space.shape[0])
        ACT_DIM = int(env.action_space.shape[0])
        ACTION_LOW = env.action_space.low.astype(np.float32).copy()
        ACTION_HIGH = env.action_space.high.astype(np.float32).copy()
    finally:
        env.close()


def _action_bounds_tensors(device):
    _detect_env_dims()
    return (
        torch.tensor(ACTION_LOW, dtype=torch.float32, device=device),
        torch.tensor(ACTION_HIGH, dtype=torch.float32, device=device),
    )


def _clip_action(act):
    _detect_env_dims()
    return np.clip(act, ACTION_LOW, ACTION_HIGH)


def _preferred_checkpoint_path() -> Path:
    return BEST_CKPT_PATH if BEST_CKPT_PATH.exists() else CKPT_PATH


def collect_random(env, n_steps: int) -> Tuple[ReplayBuffer, List[float]]:
    _detect_env_dims()
    buf = ReplayBuffer(OBS_DIM, ACT_DIM)
    ep_returns: List[float] = []
    obs, _ = env.reset(seed=SEED)
    ep_ret = 0.0
    obsl, actl, nobsl, rewl, donel = [], [], [], [], []
    for _ in range(n_steps):
        act = env.action_space.sample()
        nobs, rew, term, trunc, _ = env.step(act)
        done = float(term or trunc)
        obsl.append(obs); actl.append(act); nobsl.append(nobs)
        rewl.append(rew); donel.append(done)
        ep_ret += rew
        obs = nobs
        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
    buf.add_batch(np.array(obsl), np.array(actl), np.array(nobsl),
                  np.array(rewl), np.array(donel))
    return buf, ep_returns


def run_mpc_iter(env, buffer, model, obs_norm, delta_norm, device):
    ep_returns: List[float] = []
    pred_errs:  List[float] = []
    action_low_t, action_high_t = _action_bounds_tensors(device)
    obs, _ = env.reset()
    ep_ret = 0.0
    obsl, actl, nobsl, rewl, donel = [], [], [], [], []
    for _ in range(STEPS_PER_ITER):
        act = mpc_action_cem(obs, model, obs_norm, delta_norm, device,
                             action_low_t, action_high_t,
                             planning_score_fn=planning_objective)
        act = _clip_action(act)
        with torch.no_grad():
            st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            at = torch.tensor(act, dtype=torch.float32, device=device).unsqueeze(0)
            pred_nobs = (st + delta_norm.unnorm(model(obs_norm.norm(st), at))).squeeze(0).cpu().numpy()
        nobs, rew, term, trunc, _ = env.step(act)
        done = float(term or trunc)
        pred_errs.append(float(np.mean((pred_nobs - nobs) ** 2)))
        obsl.append(obs); actl.append(act); nobsl.append(nobs)
        rewl.append(rew); donel.append(done)
        ep_ret += rew
        obs = nobs
        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
    buffer.add_batch(np.array(obsl), np.array(actl), np.array(nobsl),
                     np.array(rewl), np.array(donel))
    return ep_returns, float(np.mean(pred_errs))


# =============================================================================
# Checkpoint I/O
# =============================================================================
def save_checkpoint(model, obs_norm, delta_norm, path: Path):
    _detect_env_dims()
    torch.save({
        "model":      model.state_dict(),
        "obs_mean":   obs_norm.mean.detach().cpu(),
        "obs_std":    obs_norm.std.detach().cpu(),
        "delta_mean": delta_norm.mean.detach().cpu(),
        "delta_std":  delta_norm.std.detach().cpu(),
        "obs_dim":    OBS_DIM,
        "act_dim":    ACT_DIM,
        "action_low":  torch.tensor(ACTION_LOW, dtype=torch.float32),
        "action_high": torch.tensor(ACTION_HIGH, dtype=torch.float32),
        "hidden":     MODEL_HIDDEN,
        "n_layers":   MODEL_LAYERS,
    }, str(path))
    print(f"Checkpoint saved to {path}", flush=True)


def load_checkpoint(path: Path, device):
    global OBS_DIM, ACT_DIM, ACTION_LOW, ACTION_HIGH
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    OBS_DIM = int(ckpt["obs_dim"])
    ACT_DIM = int(ckpt["act_dim"])
    if "action_low" in ckpt and "action_high" in ckpt:
        ACTION_LOW = ckpt["action_low"].detach().cpu().numpy().astype(np.float32)
        ACTION_HIGH = ckpt["action_high"].detach().cpu().numpy().astype(np.float32)
    else:
        _detect_env_dims()

    model = DynamicsModel(ckpt["obs_dim"], ckpt["act_dim"],
                          ckpt["hidden"], ckpt["n_layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    obs_norm   = Normalizer(ckpt["obs_dim"], device)
    delta_norm = Normalizer(ckpt["obs_dim"], device)
    obs_norm.mean   = ckpt["obs_mean"].to(device)
    obs_norm.std    = ckpt["obs_std"].to(device)
    delta_norm.mean = ckpt["delta_mean"].to(device)
    delta_norm.std  = ckpt["delta_std"].to(device)
    print(f"Loaded checkpoint from {path}", flush=True)
    return model, obs_norm, delta_norm


# =============================================================================
# Plotting
# =============================================================================
def _smooth(x: List[float], w: int) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_rewards(returns: List[float], out: Path, smooth: int = 5):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(returns, alpha=0.35, color="steelblue", label="Episode return")
    ma = _smooth(returns, smooth)
    if len(returns) >= smooth:
        ax.plot(range(smooth - 1, len(returns)), ma,
                color="steelblue", lw=2, label=f"{smooth}-ep moving avg")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.set_title("MBRL v1.5 — HalfCheetah-v5 Training Reward")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(out), dpi=150); plt.close(fig)
    print(f"Saved: {out}", flush=True)


def save_all_curves(history: Dict, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_rewards(history["all_returns"], plots_dir / "reward_curve.jpg")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_losses_flat"], color="tomato", lw=1, alpha=0.8)
    ax.set_xlabel("Epoch (across all iterations)"); ax.set_ylabel("MSE (normalised)")
    ax.set_title("Dynamics Model — Training Loss"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(plots_dir / "model_train_loss.jpg"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["val_mses"], marker="o", color="darkorange")
    ax.set_xlabel("Outer iteration"); ax.set_ylabel("MSE (original space)")
    ax.set_title("Dynamics Model — Validation 1-Step Prediction MSE"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(plots_dir / "model_val_loss.jpg"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["rollout_errs"], marker="s", color="seagreen")
    ax.set_xlabel("Outer iteration"); ax.set_ylabel("Mean 1-step MSE (env space)")
    ax.set_title("MPC Rollout — Real vs Predicted 1-Step Error"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(plots_dir / "rollout_pred_error.jpg"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history["train_losses_flat"], color="tomato", lw=1, alpha=0.8)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_mses"], marker="o", color="darkorange")
    axes[1].set_title("Val 1-Step MSE"); axes[1].set_xlabel("Iteration"); axes[1].grid(alpha=0.3)
    axes[2].plot(history["rollout_errs"], marker="s", color="seagreen")
    axes[2].set_title("Rollout Pred Error"); axes[2].set_xlabel("Iteration"); axes[2].grid(alpha=0.3)
    fig.suptitle("MBRL v1.5 Diagnostics — HalfCheetah-v5")
    fig.tight_layout(); fig.savefig(str(plots_dir / "diagnostics_combined.jpg"), dpi=150); plt.close(fig)

    print(f"All diagnostic curves saved to {plots_dir}/", flush=True)


# =============================================================================
# Training
# =============================================================================
def train():
    set_seed(SEED)
    _detect_env_dims()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | OBS={OBS_DIM} ACT={ACT_DIM}", flush=True)
    print(f"Config: iters={NUM_ITERATIONS}, steps/iter={STEPS_PER_ITER}, "
          f"init={INIT_RANDOM_STEPS}, H={MPC_HORIZON}, K={MPC_NUM_CANDIDATES}, "
          f"I={MPC_CEM_ITERS}, E={MPC_NUM_ELITES}", flush=True)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model      = DynamicsModel(OBS_DIM, ACT_DIM, MODEL_HIDDEN, MODEL_LAYERS).to(device)
    obs_norm   = Normalizer(OBS_DIM, device)
    delta_norm = Normalizer(OBS_DIM, device)
    optimizer  = optim.Adam(model.parameters(), lr=MODEL_LR, weight_decay=MODEL_WEIGHT_DECAY)

    history: Dict = dict(all_returns=[], train_losses_flat=[],
                         val_mses=[], rollout_errs=[])
    best_val_mse = float("inf")

    # ---- Phase 0: random data collection ----
    print(f"\n[Phase 0] Collecting {INIT_RANDOM_STEPS} random transitions...", flush=True)
    env = make_env()
    buffer, rand_rets = collect_random(env, INIT_RANDOM_STEPS)
    env.close()
    history["all_returns"].extend(rand_rets)
    avg0 = np.mean(rand_rets) if rand_rets else 0.0
    print(f"  Buffer: {buffer.size} transitions | {len(rand_rets)} episodes | avg return={avg0:.1f}",
          flush=True)

    # ---- Outer MBRL loop ----
    for it in range(NUM_ITERATIONS):
        t0 = time.time()
        print(f"\n[Iter {it+1}/{NUM_ITERATIONS}] Training model on {buffer.size} transitions...",
              flush=True)
        train_losses, val_mse = train_model(model, optimizer, buffer,
                                            obs_norm, delta_norm, device)
        history["train_losses_flat"].extend(train_losses)
        history["val_mses"].append(val_mse)
        print(f"  Model: last_train_loss={train_losses[-1]:.4f}  val_mse={val_mse:.6f}",
              flush=True)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            print(f"  ** New best val_mse={best_val_mse:.6f}; saving best checkpoint",
                  flush=True)
            save_checkpoint(model, obs_norm, delta_norm, BEST_CKPT_PATH)

        print(f"  CEM MPC collecting {STEPS_PER_ITER} steps "
              f"(H={MPC_HORIZON}, K={MPC_NUM_CANDIDATES}, "
              f"I={MPC_CEM_ITERS}, E={MPC_NUM_ELITES})...", flush=True)
        env = make_env()
        ep_rets, mean_err = run_mpc_iter(env, buffer, model,
                                         obs_norm, delta_norm, device)
        env.close()
        history["all_returns"].extend(ep_rets)
        history["rollout_errs"].append(mean_err)
        avg_r = np.mean(ep_rets) if ep_rets else 0.0
        print(f"  CEM MPC: {len(ep_rets)} eps | avg_return={avg_r:.1f} | "
              f"pred_err={mean_err:.6f} | {time.time()-t0:.1f}s", flush=True)

        # Save checkpoint after each iteration so a long run can be resumed
        # for evaluation even if interrupted.
        save_checkpoint(model, obs_norm, delta_norm, CKPT_PATH)

    # ---- Final retrain on the full buffer ----
    print(f"\n[Final retrain] buffer={buffer.size} transitions", flush=True)
    _, final_val_mse = train_model(model, optimizer, buffer,
                                   obs_norm, delta_norm, device)
    if final_val_mse < best_val_mse:
        best_val_mse = final_val_mse
        print(f"  ** New best val_mse={best_val_mse:.6f}; saving best checkpoint",
              flush=True)
        save_checkpoint(model, obs_norm, delta_norm, BEST_CKPT_PATH)
    save_checkpoint(model, obs_norm, delta_norm, CKPT_PATH)

    # ---- Plots ----
    print("\n[Plotting] ...", flush=True)
    plot_rewards(history["all_returns"], SUBMISSION_DIR / "Plot.jpg")
    save_all_curves(history, PLOTS_DIR)

    return model, obs_norm, delta_norm, history


# =============================================================================
# Evaluation + video recording (mirrors hw2 split: measurement + video)
# =============================================================================
def evaluate_and_record(model, obs_norm, delta_norm, device):
    print(f"\n[Eval] Running {N_EVAL_EPISODES} measurement episodes...", flush=True)
    action_low_t, action_high_t = _action_bounds_tensors(device)
    env = make_env()
    rewards: List[float] = []
    for i in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=SEED + 1000 + i)
        done = False
        ep_ret = 0.0
        while not done:
            act = mpc_action_cem(obs, model, obs_norm, delta_norm, device,
                                 action_low_t, action_high_t,
                                 planning_score_fn=planning_objective)
            act = _clip_action(act)
            obs, rew, term, trunc, _ = env.step(act)
            done = term or trunc
            ep_ret += rew
        rewards.append(ep_ret)
        print(f"  Eval ep {i+1}: return={ep_ret:.2f}", flush=True)
    env.close()
    if rewards:
        arr = np.array(rewards)
        print(f"Eval over {len(rewards)} eps: mean={arr.mean():.2f} ± {arr.std():.2f}, "
              f"min={arr.min():.2f}, max={arr.max():.2f}", flush=True)

    print(f"\n[Video] Recording {EVAL_VIDEO_EPISODES} episode(s)...", flush=True)
    env = make_env(render_mode="rgb_array")
    frames: List[np.ndarray] = []
    for ep in range(EVAL_VIDEO_EPISODES):
        obs, _ = env.reset(seed=SEED + 200 + ep)
        done = False
        ep_ret = 0.0
        ep_frames = 0
        while not done:
            act = mpc_action_cem(obs, model, obs_norm, delta_norm, device,
                                 action_low_t, action_high_t,
                                 planning_score_fn=planning_objective)
            act = _clip_action(act)
            obs, rew, term, trunc, _ = env.step(act)
            done = term or trunc
            frames.append(env.render())
            ep_ret += rew
            ep_frames += 1
            if VIDEO_MAX_FRAMES is not None and ep_frames >= VIDEO_MAX_FRAMES:
                break
        print(f"  Video ep {ep+1}: return={ep_ret:.2f}, frames={ep_frames}", flush=True)
    env.close()

    out_path = SUBMISSION_DIR / "Video.mp4"
    frames_u8 = [f.astype(np.uint8) for f in frames]
    try:
        with imageio.get_writer(str(out_path), fps=VIDEO_FPS,
                                codec="libx264", quality=7) as writer:
            for f in frames_u8:
                writer.append_data(f)
    except Exception:
        imageio.mimsave(str(out_path), frames_u8, fps=VIDEO_FPS)

    duration = len(frames) / VIDEO_FPS
    print(f"Video: {out_path} ({len(frames)} frames, {duration:.1f}s)", flush=True)
    if duration < 30.0:
        print(f"WARNING: video is {duration:.1f}s (< 30 s)", flush=True)
    return rewards, duration


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Short sanity-check run (verifies pipeline only)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved checkpoint and record video")
    args = parser.parse_args()
    if args.smoke:
        _apply_smoke()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to train.log inside task1/
    log_file = open(LOG_PATH, "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"=== Run started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"args: {vars(args)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _detect_env_dims()

    if args.eval_only:
        ckpt_path = _preferred_checkpoint_path()
        if not ckpt_path.exists():
            print(f"ERROR: no checkpoint found at {BEST_CKPT_PATH} or {CKPT_PATH}. Train first.",
                  flush=True)
            sys.exit(1)
        model, obs_norm, delta_norm = load_checkpoint(ckpt_path, device)
        evaluate_and_record(model, obs_norm, delta_norm, device)
    else:
        model, obs_norm, delta_norm, history = train()
        ckpt_path = _preferred_checkpoint_path()
        print(f"Reloading {ckpt_path.name} for final evaluation", flush=True)
        model, obs_norm, delta_norm = load_checkpoint(ckpt_path, device)
        evaluate_and_record(model, obs_norm, delta_norm, device)

    print(f"\n=== Run finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    log_file.close()


if __name__ == "__main__":
    main()
