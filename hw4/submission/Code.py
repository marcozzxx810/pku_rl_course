#!/usr/bin/env python3
"""
Task 2: VDN (Value Decomposition Networks) for PettingZoo MPE simple_tag_v3.

Trains 3 adversary (predator) agents cooperatively to catch the good (prey)
agent using VDN (Sunehag et al. 2017).  Only the 3 adversary agents are
trained with VDN. The good agent / prey uses a fixed seeded random policy, so
this experiment evaluates cooperative predator learning against a fixed
opponent rather than simultaneous predator-prey self-play.

Architecture: shared Q-network with dueling head across the 3 symmetric
adversaries (paper Table 1 "S" variant — avoids the lazy-agent problem).
Deliberate simplifications vs. the paper: feed-forward MLP (no LSTM),
single-step TD target (no lambda-returns), which is standard for near-fully-
observed MPE tasks.

Setup:
    uv venv
    source .venv/bin/activate
    uv pip install torch numpy matplotlib pettingzoo[mpe] supersuit pygame \
                   imageio imageio-ffmpeg tqdm

Usage:
    python submission/task2/Code.py --mode train --episodes 2000
    python submission/task2/Code.py --mode eval   --checkpoint submission/task2/checkpoints/best_model.pt
    python submission/task2/Code.py --mode record --checkpoint submission/task2/checkpoints/best_model.pt --video-seconds 30
"""

# =============================================================================
# Hyperparameters
# =============================================================================
SEED = 42
NUM_ADVERSARIES = 3
NUM_GOOD = 1
NUM_OBSTACLES = 2
OBS_DIM = None        # detected at runtime; actual value is 16 (PDF has dims swapped)
N_ACTIONS = 5         # no-op, left, right, down, up
TORCH_NUM_THREADS = 1 # small MLP updates are faster on CPU without thread-pool overhead

HIDDEN = 128
DUELING = True        # dueling Q-head (paper Section 4.1)

GAMMA = 0.99          # EPyMARL default; better long-horizon credit assignment
DOUBLE_Q = True       # Double DQN — reduces Q-value overestimation (EPyMARL default)
STANDARDISE_REWARDS = False  # raw rewards (0/10) are numerically stable; batch-norm drifts as buffer fills
USE_AGENT_ID = True   # append one-hot agent index to obs; breaks symmetry in param-sharing
LR = 1e-4
BATCH_SIZE = 128
BUFFER_SIZE = 100_000
TARGET_UPDATE_INTERVAL = 500  # hard copy online→target every N gradient steps
LEARN_EVERY = 32             # update every N env steps; 100k eps still gives ~312k updates
WARMUP_STEPS = 5_000  # steps before learning starts
GRAD_CLIP = 10.0
REWARD_SCALE = 10.0   # scale sparse +10 collision rewards before storing in replay
DISTANCE_SHAPING = 0.1  # training-only dense pursuit signal, matching simple_tag's optional shaping scale
PREY_POLICY = "random"  # fixed seeded random prey; scripted flee is too evasive for this setup

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000

MAX_CYCLES = 100      # steps per episode during training/eval
MAX_CYCLES_RECORD = MAX_CYCLES

EPISODES_DEFAULT = 20_000
EVAL_EVERY = 500      # episodes
EVAL_EPISODES = 30

VIDEO_FPS = 10
MIN_VIDEO_SECONDS = 30.0

# Smoke-test overrides
_SMOKE = dict(
    WARMUP_STEPS=64,
    BUFFER_SIZE=1_000,
    EPISODES_DEFAULT=25,
    EPS_DECAY_STEPS=200,
    EVAL_EVERY=10,
    EVAL_EPISODES=2,
    MAX_CYCLES_RECORD=50,
    MIN_VIDEO_SECONDS=0.0,
)

# =============================================================================
# Imports
# =============================================================================
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pettingzoo.mpe import simple_tag_v3
from tqdm import trange

torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(TORCH_NUM_THREADS)
except RuntimeError:
    # Inter-op threads can only be set before parallel work starts.
    pass

# =============================================================================
# Paths
# =============================================================================
SUBMISSION_DIR = Path(__file__).parent
CKPT_DIR = SUBMISSION_DIR / "checkpoints"
PLOTS_DIR = SUBMISSION_DIR / "plots"
VIDEOS_DIR = SUBMISSION_DIR / "videos"
LOGS_DIR = SUBMISSION_DIR / "logs"

BEST_CKPT = CKPT_DIR / "best_model.pt"
FINAL_CKPT = CKPT_DIR / "final_model.pt"
METRICS_CSV = LOGS_DIR / "metrics.csv"
CONFIG_JSON = LOGS_DIR / "config.json"
VIDEO_PATH = VIDEOS_DIR / "eval_demo.mp4"


# =============================================================================
# Utilities
# =============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s); st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_eps(step: int) -> float:
    frac = min(step / max(EPS_DECAY_STEPS, 1), 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


def _smooth(x: List[float], w: int) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _adversary_agents(env) -> List[str]:
    return [a for a in env.possible_agents if a.startswith("adversary")]


# =============================================================================
# Environment factory
# =============================================================================
def make_env(render_mode: Optional[str] = None, max_cycles: int = MAX_CYCLES):
    return simple_tag_v3.parallel_env(
        num_good=NUM_GOOD,
        num_adversaries=NUM_ADVERSARIES,
        num_obstacles=NUM_OBSTACLES,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
    )


# =============================================================================
# Q-Network (feed-forward MLP with optional dueling head)
# =============================================================================
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.advantage = nn.Linear(hidden, n_actions)
        else:
            self.out = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if self.dueling:
            v = self.value(h)
            a = self.advantage(h)
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.out(h)


# =============================================================================
# Replay Buffer
# =============================================================================
class ReplayBuffer:
    """Stores joint transitions: (obs[N,D], act[N], r, nobs[N,D], done)."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int):
        self.cap = int(capacity)
        self.n = int(n_agents)
        self.d = int(obs_dim)
        self.ptr = 0
        self.size = 0
        self._obs  = np.zeros((self.cap, self.n, self.d), dtype=np.float32)
        self._act  = np.zeros((self.cap, self.n),         dtype=np.int64)
        self._rew  = np.zeros((self.cap,),                dtype=np.float32)
        self._nobs = np.zeros((self.cap, self.n, self.d), dtype=np.float32)
        self._done = np.zeros((self.cap,),                dtype=np.float32)

    def add(self, obs, act, rew, nobs, done):
        self._obs [self.ptr] = obs
        self._act [self.ptr] = act
        self._rew [self.ptr] = float(rew)
        self._nobs[self.ptr] = nobs
        self._done[self.ptr] = float(done)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=bs)
        to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        return (
            to(self._obs [idx]),
            torch.as_tensor(self._act[idx], dtype=torch.long, device=device),
            to(self._rew [idx]),
            to(self._nobs[idx]),
            to(self._done[idx]),
        )


# =============================================================================
# VDN Agent
# =============================================================================
class VDNAgent:
    """
    Shared-parameter VDN agent for N symmetric cooperative agents.
    Q_tot(o, a) = sum_i Q(o_i, a_i)  [Sunehag et al. 2017, eq. 1]
    """

    def __init__(self, obs_dim: int, n_actions: int, n_agents: int,
                 hidden: int, dueling: bool, device: torch.device):
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.device    = device

        self.q        = QNetwork(obs_dim, n_actions, hidden, dueling).to(device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden, dueling).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q.parameters(), lr=LR)
        self._n_updates = 0  # gradient step counter for hard target update

    @torch.no_grad()
    def select_actions_greedy(self, obs_arr: np.ndarray) -> np.ndarray:
        """obs_arr: (N, D) -> actions: (N,) int"""
        x = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        q_vals = self.q(x)  # (N, A)
        return q_vals.argmax(dim=-1).cpu().numpy().astype(np.int64)

    def select_actions(self, obs_arr: np.ndarray, eps: float) -> np.ndarray:
        """Per-agent independent epsilon-greedy exploration."""
        random_acts = np.random.randint(0, self.n_actions, size=self.n_agents)
        explore = np.random.random(self.n_agents) < eps
        if explore.all():
            return random_acts
        greedy = self.select_actions_greedy(obs_arr)
        if not explore.any():
            return greedy
        return np.where(explore, random_acts, greedy)

    def update(self, batch) -> float:
        obs, act, rew, nobs, done = batch
        # obs:  (B, N, D)
        # act:  (B, N)
        # rew:  (B,)
        # nobs: (B, N, D)
        # done: (B,)
        B, N, D = obs.shape

        # Reward standardisation: scale-only (divide by std, keep zero-rewards at 0)
        # z-score (subtracting mean) would assign negative values to zero rewards,
        # causing Q-values to unlearn as the buffer composition changes over training.
        if STANDARDISE_REWARDS:
            rew_std = rew.std()
            if rew_std > 1e-6:
                rew = rew / rew_std

        # Flatten agent dimension for shared network
        obs_flat  = obs.view(B * N, D)
        nobs_flat = nobs.view(B * N, D)

        # Q(o_i, a_i) for current actions
        q_all = self.q(obs_flat).view(B, N, self.n_actions)  # (B, N, A)
        q_chosen = q_all.gather(dim=2, index=act.unsqueeze(-1)).squeeze(-1)  # (B, N)
        q_tot = q_chosen.sum(dim=1)  # (B,)

        with torch.no_grad():
            q_next_all = self.q_target(nobs_flat).view(B, N, self.n_actions)  # (B, N, A)
            if DOUBLE_Q:
                # Online net selects action; target net evaluates it (reduces overestimation)
                q_next_online = self.q(nobs_flat).view(B, N, self.n_actions)
                next_acts = q_next_online.argmax(dim=-1, keepdim=True)  # (B, N, 1)
                q_next_max = q_next_all.gather(2, next_acts).squeeze(-1)  # (B, N)
            else:
                q_next_max = q_next_all.max(dim=-1).values  # (B, N)
            q_tot_next = q_next_max.sum(dim=1)  # (B,)
            target = rew + GAMMA * (1.0 - done) * q_tot_next

        # Huber loss caps large early TD outliers while remaining near-MSE after reward scaling.
        loss = F.smooth_l1_loss(q_tot, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # Hard target update: copy online → target every TARGET_UPDATE_INTERVAL steps
        self._n_updates += 1
        if self._n_updates % TARGET_UPDATE_INTERVAL == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def state_dict_payload(self, episode: int, best_eval: float) -> dict:
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "n_agents": self.n_agents,
            "hidden": HIDDEN,
            "dueling": DUELING,
            "episode": episode,
            "best_eval": best_eval,
        }

    def load_state_dict_payload(self, payload: dict):
        self.q.load_state_dict(payload["q"])
        self.q_target.load_state_dict(payload["q_target"])
        self.q.eval()
        self.q_target.eval()


# =============================================================================
# Checkpoint helpers
# =============================================================================
def save_checkpoint(agent: VDNAgent, path: Path, episode: int, best_eval: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict_payload(episode, best_eval), str(path))
    print(f"  Checkpoint saved → {path}", flush=True)


def load_checkpoint(path: Path, device: torch.device) -> VDNAgent:
    payload = torch.load(str(path), map_location=device, weights_only=True)
    agent = VDNAgent(
        obs_dim=int(payload["obs_dim"]),
        n_actions=int(payload["n_actions"]),
        n_agents=int(payload["n_agents"]),
        hidden=int(payload.get("hidden", HIDDEN)),
        dueling=bool(payload.get("dueling", DUELING)),
        device=device,
    )
    agent.load_state_dict_payload(payload)
    print(f"Loaded checkpoint from {path}  (ep={payload['episode']}, best_eval={payload['best_eval']:.2f})", flush=True)
    return agent


# =============================================================================
# Env dimension detection
# =============================================================================
def detect_dims():
    """Detect OBS_DIM from the environment (base obs + optional one-hot agent ID)."""
    global OBS_DIM
    if OBS_DIM is not None:
        return
    env = make_env()
    env.reset()
    base_dim = env.observation_space("adversary_0").shape[0]
    env.close()
    OBS_DIM = base_dim + (NUM_ADVERSARIES if USE_AGENT_ID else 0)
    print(f"Detected OBS_DIM={OBS_DIM} (base={base_dim}, use_agent_id={USE_AGENT_ID})", flush=True)


# =============================================================================
# Obs / action helpers
# =============================================================================
ADVERSARY_KEYS = [f"adversary_{i}" for i in range(NUM_ADVERSARIES)]
_AGENT_ID_MATRIX = np.eye(NUM_ADVERSARIES, dtype=np.float32)  # (N, N) one-hot IDs


def obs_to_array(obs_dict: dict, keys: List[str]) -> np.ndarray:
    """Stack observations into (N, D) array, optionally appending one-hot agent ID."""
    base = np.stack([obs_dict[k].astype(np.float32) for k in keys], axis=0)  # (N, D)
    if USE_AGENT_ID:
        return np.concatenate([base, _AGENT_ID_MATRIX], axis=1)  # (N, D+N)
    return base


def actions_to_dict(act_arr: np.ndarray, keys: List[str]) -> dict:
    return {k: int(act_arr[i]) for i, k in enumerate(keys)}


def prey_distance_shaping(adv_obs: np.ndarray) -> float:
    """Dense training reward: penalize total adversary distance to the prey."""
    prey_rel_pos = adv_obs[:, 12:14]  # good agent relative position in adversary observations
    return -DISTANCE_SHAPING * float(np.linalg.norm(prey_rel_pos, axis=1).sum())


def prey_action(obs_good: np.ndarray, rng: np.random.Generator) -> int:
    """Fixed non-learning prey policy used in train/eval/record."""
    if PREY_POLICY == "scripted_flee":
        return scripted_flee(obs_good)
    if PREY_POLICY == "noop":
        return 0
    return int(rng.integers(N_ACTIONS))


# Fixed prey policy: flee from nearest adversary along the dominant axis.
def scripted_flee(obs_good: np.ndarray) -> int:
    """
    Heuristic flee policy for the good agent.

    Good-agent observation layout (14-dim, simple_tag_v3):
      obs[0:2]  — self velocity
      obs[2:4]  — self position
      obs[4:8]  — relative positions of 2 obstacles (2 × 2)
      obs[8:14] — relative positions of 3 adversaries (3 × 2)

    Discrete actions: 0=no-op, 1=left(-x), 2=right(+x), 3=down(-y), 4=up(+y)

    Strategy: move directly away from the nearest adversary along the dominant axis.
    """
    adv_pos = obs_good[8:14].reshape(3, 2)          # (3, 2) relative positions
    dists = np.linalg.norm(adv_pos, axis=1)          # (3,)
    nearest = adv_pos[np.argmin(dists)]              # (2,) dx, dy
    dx, dy = float(nearest[0]), float(nearest[1])

    if abs(dx) >= abs(dy):
        # flee along x-axis
        return 1 if dx > 0 else 2   # adversary to +x → go left; to -x → go right
    else:
        # flee along y-axis
        return 3 if dy > 0 else 4   # adversary to +y → go down; to -y → go up


# =============================================================================
# Evaluate
# =============================================================================
@torch.no_grad()
def evaluate(agent: VDNAgent, n_episodes: int, seed_offset: int = 1000,
             label: str = "Eval") -> List[float]:
    env = make_env()
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=SEED + seed_offset + i)
        prey_rng = np.random.default_rng(SEED + seed_offset + 50_000 + i)
        ep_ret = 0.0
        while env.agents:
            adv_obs = obs_to_array(obs, ADVERSARY_KEYS)
            adv_acts = agent.select_actions_greedy(adv_obs)
            actions = actions_to_dict(adv_acts, ADVERSARY_KEYS)
            for good_a in env.agents:
                if good_a not in actions:
                    actions[good_a] = prey_action(obs[good_a], prey_rng)
            obs, rewards, terms, truncs, _ = env.step(actions)
            ep_ret += rewards.get("adversary_0", 0.0)
        returns.append(ep_ret)
    env.close()
    arr = np.array(returns)
    print(f"[{label}] mean={arr.mean():.2f} ± {arr.std():.2f}  "
          f"(min={arr.min():.2f} max={arr.max():.2f})", flush=True)
    return returns


# =============================================================================
# Record video
# =============================================================================
def record_video(agent: VDNAgent, video_path: Path, fps: int,
                 min_seconds: float, max_cycles: int):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    min_frames = math.ceil(min_seconds * fps)
    env = make_env(render_mode="rgb_array", max_cycles=max_cycles)
    frames: List[np.ndarray] = []
    ep = 0
    print(f"[Record] target ≥ {min_seconds:.0f}s ({min_frames} frames @ {fps} fps) ...", flush=True)
    while len(frames) < min_frames:
        obs, _ = env.reset(seed=SEED + 500 + ep)
        prey_rng = np.random.default_rng(SEED + 75_000 + ep)
        ep_ret = 0.0
        while env.agents:
            adv_obs = obs_to_array(obs, ADVERSARY_KEYS)
            adv_acts = agent.select_actions_greedy(adv_obs)
            actions = actions_to_dict(adv_acts, ADVERSARY_KEYS)
            for good_a in env.agents:
                if good_a not in actions:
                    actions[good_a] = prey_action(obs[good_a], prey_rng)
            obs, rewards, terms, truncs, _ = env.step(actions)
            ep_ret += rewards.get("adversary_0", 0.0)
            frame = env.render()
            if frame is not None:
                frames.append(frame.astype(np.uint8))
        ep += 1
        print(f"  ep {ep}: return={ep_ret:.1f}, frames_so_far={len(frames)}", flush=True)
    env.close()

    if not frames:
        print("[Record] No frames captured — skipping video write (smoke mode?)", flush=True)
        return

    try:
        with imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=7) as w:
            for f in frames:
                w.append_data(f)
    except Exception:
        imageio.mimsave(str(video_path), frames, fps=fps)

    duration = len(frames) / fps
    print(f"[Record] saved {video_path}  ({len(frames)} frames, {duration:.1f}s)", flush=True)
    if min_seconds > 0 and duration < min_seconds:
        raise RuntimeError(f"Video is only {duration:.1f}s < {min_seconds:.1f}s required")


# =============================================================================
# Plotting
# =============================================================================
def _plot_series(y, out: Path, title: str, ylabel: str, color: str,
                 x=None, xlabel: str = "Episode"):
    if not y:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x_vals = x if x is not None else list(range(len(y)))
    ax.plot(x_vals, y, color=color, lw=1.2, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def save_plots(history: dict, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    ep_ret = history["episode_returns"]

    # 1. Raw reward
    _plot_series(ep_ret, plots_dir / "reward.jpg",
                 "VDN – Team Return per Episode", "Team Return", "steelblue")

    # 2. Moving-average reward
    if len(ep_ret) >= 5:
        ma = _smooth(ep_ret, 50)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ep_ret, color="steelblue", lw=0.8, alpha=0.4, label="Raw")
        ax.plot(range(49, 49 + len(ma)), ma, color="darkorange", lw=2, label="50-ep MA")
        ax.set_title("VDN – Team Return (moving average)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Team Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(plots_dir / "reward_ma.jpg"), dpi=150)
        plt.close(fig)

    # 3. Loss
    _plot_series(history["losses"], plots_dir / "loss.jpg",
                 "VDN – TD Loss per Episode", "Huber Loss", "tomato")

    # 4. Episode length
    _plot_series(history["ep_lengths"], plots_dir / "episode_length.jpg",
                 "Episode Length", "Steps", "seagreen")

    # 5. Epsilon
    _plot_series(history["epsilons"], plots_dir / "epsilon.jpg",
                 "Epsilon (exploration rate)", "ε", "purple")

    # 6. Eval curve
    if history["eval_episodes"] and history["eval_returns"]:
        _plot_series(history["eval_returns"], plots_dir / "eval.jpg",
                     "VDN – Greedy Eval Team Return", "Mean Return", "darkorange",
                     x=history["eval_episodes"])

    # 7. Combined grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    pairs = [
        (ep_ret,               "steelblue",  "Team Return"),
        (history["losses"],    "tomato",     "TD Loss"),
        (history["ep_lengths"],"seagreen",   "Ep Length"),
        (history["epsilons"],  "purple",     "Epsilon"),
    ]
    for ax, (data, color, title) in zip(axes[:4], pairs):
        if data:
            ax.plot(data, color=color, lw=0.9)
            ax.set_title(title)
            ax.grid(alpha=0.3)
    if history["eval_episodes"] and history["eval_returns"]:
        axes[4].plot(history["eval_episodes"], history["eval_returns"],
                     color="darkorange", lw=1.5, marker="o", ms=4)
        axes[4].set_title("Eval Return")
        axes[4].grid(alpha=0.3)
    if len(ep_ret) >= 5:
        ma2 = _smooth(ep_ret, 50)
        axes[5].plot(range(49, 49 + len(ma2)), ma2, color="navy", lw=1.5)
        axes[5].set_title("Return (50-ep MA)")
        axes[5].grid(alpha=0.3)
    fig.suptitle("VDN – simple_tag_v3 Training Diagnostics")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "combined.jpg"), dpi=150)
    plt.close(fig)
    print(f"Plots saved to {plots_dir}/", flush=True)


# =============================================================================
# Metrics CSV and config JSON
# =============================================================================
def init_metrics_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["episode", "return", "length", "epsilon", "mean_loss", "eval_return"]
        )


def append_metrics_csv(path: Path, episode: int, ret: float, length: int,
                        eps: float, loss: float, eval_ret: Optional[float]):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            episode, f"{ret:.4f}", length, f"{eps:.4f}",
            f"{loss:.6f}", "" if eval_ret is None else f"{eval_ret:.4f}",
        ])


def save_config_json(path: Path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "SEED": SEED, "NUM_ADVERSARIES": NUM_ADVERSARIES, "NUM_GOOD": NUM_GOOD,
        "NUM_OBSTACLES": NUM_OBSTACLES, "OBS_DIM": OBS_DIM, "N_ACTIONS": N_ACTIONS,
        "HIDDEN": HIDDEN, "DUELING": DUELING, "DOUBLE_Q": DOUBLE_Q,
        "STANDARDISE_REWARDS": STANDARDISE_REWARDS, "USE_AGENT_ID": USE_AGENT_ID,
        "GAMMA": GAMMA, "LR": LR, "TORCH_NUM_THREADS": TORCH_NUM_THREADS,
        "BATCH_SIZE": BATCH_SIZE, "BUFFER_SIZE": BUFFER_SIZE,
        "TARGET_UPDATE_INTERVAL": TARGET_UPDATE_INTERVAL,
        "LEARN_EVERY": LEARN_EVERY, "WARMUP_STEPS": WARMUP_STEPS, "GRAD_CLIP": GRAD_CLIP,
        "REWARD_SCALE": REWARD_SCALE, "DISTANCE_SHAPING": DISTANCE_SHAPING,
        "LOSS": "smooth_l1", "TRAIN_REWARD": "scaled_collision_plus_distance_shaping",
        "PREY_POLICY": PREY_POLICY,
        "EPS_START": EPS_START, "EPS_END": EPS_END, "EPS_DECAY_STEPS": EPS_DECAY_STEPS,
        "MAX_CYCLES": MAX_CYCLES, "EPISODES": vars(args).get("episodes", EPISODES_DEFAULT),
        "EVAL_EVERY": EVAL_EVERY, "EVAL_EPISODES": EVAL_EPISODES,
        "VIDEO_FPS": VIDEO_FPS, "MIN_VIDEO_SECONDS": MIN_VIDEO_SECONDS,
        "cli_args": vars(args),
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved → {path}", flush=True)


# =============================================================================
# Training
# =============================================================================
def train(args):
    global EPS_START, EPS_DECAY_STEPS
    detect_dims()
    base_seed = SEED if args.seed is None else args.seed
    set_seed(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | OBS_DIM={OBS_DIM} | N_ACTIONS={N_ACTIONS} | "
          f"N_ADV={NUM_ADVERSARIES} | dueling={DUELING}", flush=True)
    n_episodes = args.episodes

    if not args.smoke:
        EPS_DECAY_STEPS = int(0.4 * n_episodes * MAX_CYCLES)
        print(f"[epsilon schedule] EPS_DECAY_STEPS = {EPS_DECAY_STEPS} "
              f"(0.4 x {n_episodes} ep x {MAX_CYCLES} steps)", flush=True)

    for d in [CKPT_DIR, PLOTS_DIR, VIDEOS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    save_config_json(CONFIG_JSON, args)
    init_metrics_csv(METRICS_CSV)

    env = make_env()
    agent = VDNAgent(OBS_DIM, N_ACTIONS, NUM_ADVERSARIES, HIDDEN, DUELING, device)
    buf = ReplayBuffer(BUFFER_SIZE, NUM_ADVERSARIES, OBS_DIM)

    # Resume: load checkpoint weights and skip exploration phase
    if getattr(args, 'resume', False):
        ckpt_path = Path(args.checkpoint) if args.checkpoint else (BEST_CKPT if BEST_CKPT.exists() else FINAL_CKPT)
        agent.load_state_dict_payload(torch.load(str(ckpt_path), map_location=device, weights_only=True))
        EPS_START = EPS_END      # already explored — start greedy
        EPS_DECAY_STEPS = 1      # epsilon stays at EPS_END throughout
        print(f"Resumed weights from {ckpt_path}  (eps fixed at {EPS_END})", flush=True)

    history = {
        "episode_returns": [], "losses": [], "ep_lengths": [], "epsilons": [],
        "eval_episodes": [], "eval_returns": [],
    }

    best_eval = -float("inf")
    global_step = 0
    eps_milestones_hit = {0.1: False, 0.05: False}

    for ep in trange(1, n_episodes + 1, desc="Train", unit="ep", mininterval=5.0):
        obs, _ = env.reset(seed=base_seed + ep)
        prey_rng = np.random.default_rng(base_seed + 100_000 + ep)
        ep_ret, ep_len, ep_losses = 0.0, 0, []
        eps = linear_eps(global_step)

        while env.agents:
            adv_obs = obs_to_array(obs, ADVERSARY_KEYS)
            adv_acts = agent.select_actions(adv_obs, eps)
            actions = actions_to_dict(adv_acts, ADVERSARY_KEYS)
            for good_a in env.agents:
                if good_a not in actions:
                    actions[good_a] = prey_action(obs[good_a], prey_rng)

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            # Team reward: all adversaries get the same collective reward
            team_r_raw = rewards.get("adversary_0", 0.0)
            team_r_train = (team_r_raw / REWARD_SCALE) + prey_distance_shaping(adv_obs)
            done = (not env.agents) or all(
                terms.get(a, False) or truncs.get(a, False)
                for a in ADVERSARY_KEYS
            )

            # Use previous obs as placeholder for terminal next_obs
            if all(k in next_obs for k in ADVERSARY_KEYS):
                next_adv_obs = obs_to_array(next_obs, ADVERSARY_KEYS)
            else:
                next_adv_obs = adv_obs

            buf.add(adv_obs, adv_acts, team_r_train, next_adv_obs, float(done))

            if buf.size >= WARMUP_STEPS and global_step % LEARN_EVERY == 0:
                loss = agent.update(buf.sample(BATCH_SIZE, device))
                ep_losses.append(loss)

            obs = next_obs
            ep_ret += team_r_raw
            ep_len += 1
            global_step += 1
            eps_now = linear_eps(global_step)
            for threshold in (0.1, 0.05):
                if not eps_milestones_hit[threshold] and eps_now <= threshold + 1e-12:
                    eps_milestones_hit[threshold] = True
                    print(f"[epsilon milestone] reached {threshold} at step {global_step} "
                          f"(ep {ep})", flush=True)

        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        eps_ep = linear_eps(global_step)

        history["episode_returns"].append(ep_ret)
        history["losses"].append(mean_loss)
        history["ep_lengths"].append(ep_len)
        history["epsilons"].append(eps_ep)

        eval_ret = None

        if ep % EVAL_EVERY == 0 or ep == n_episodes:
            eval_rets = evaluate(agent, EVAL_EPISODES, seed_offset=2000, label=f"Ep{ep}")
            eval_ret = float(np.mean(eval_rets))
            history["eval_episodes"].append(ep)
            history["eval_returns"].append(eval_ret)
            if eval_ret > best_eval:
                best_eval = eval_ret
                save_checkpoint(agent, BEST_CKPT, ep, best_eval)
                print(f"  ** New best eval={best_eval:.2f}", flush=True)

        if ep == n_episodes:
            save_checkpoint(agent, FINAL_CKPT, ep, best_eval)

        append_metrics_csv(METRICS_CSV, ep, ep_ret, ep_len, eps_ep, mean_loss, eval_ret)

    env.close()

    save_plots(history, PLOTS_DIR)

    print("\n[Auto-record] Recording video with best checkpoint ...", flush=True)
    best_path = BEST_CKPT if BEST_CKPT.exists() else FINAL_CKPT
    best_agent = load_checkpoint(best_path, device)
    record_video(best_agent, VIDEO_PATH, VIDEO_FPS, MIN_VIDEO_SECONDS, MAX_CYCLES_RECORD)

    print(f"\nTraining done. best_eval={best_eval:.2f}  steps={global_step}", flush=True)


# =============================================================================
# Eval mode
# =============================================================================
def eval_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", flush=True)
        sys.exit(1)
    agent = load_checkpoint(ckpt, device)
    n = args.eval_episodes
    rets = evaluate(agent, n, seed_offset=3000, label="Eval")
    arr = np.array(rets)
    print(f"\nEval ({n} episodes): mean={arr.mean():.2f} ± {arr.std():.2f} "
          f"min={arr.min():.2f} max={arr.max():.2f}", flush=True)


# =============================================================================
# Record mode
# =============================================================================
def record_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", flush=True)
        sys.exit(1)
    agent = load_checkpoint(ckpt, device)
    mc = args.max_cycles if args.max_cycles else MAX_CYCLES_RECORD
    out = Path(args.video_out) if args.video_out else VIDEO_PATH
    record_video(agent, out, VIDEO_FPS, args.video_seconds, mc)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="VDN – MPE simple_tag_v3")
    parser.add_argument("--mode", choices=["train", "eval", "record"], default="train")
    parser.add_argument("--episodes", type=int, default=EPISODES_DEFAULT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, dest="eval_episodes")
    parser.add_argument("--video-seconds", type=float, default=MIN_VIDEO_SECONDS, dest="video_seconds")
    parser.add_argument("--video-out", type=str, default=None, dest="video_out")
    parser.add_argument("--max-cycles", type=int, default=None, dest="max_cycles")
    parser.add_argument("--resume", action="store_true", help="Load checkpoint weights and train with eps=EPS_END (no re-exploration)")
    parser.add_argument("--smoke", action="store_true", help="Short sanity-check run")
    args = parser.parse_args()

    if args.smoke:
        g = globals()
        for k, v in _SMOKE.items():
            g[k] = v
        if args.mode == "train":
            args.episodes = EPISODES_DEFAULT  # already overridden above
        print("=== SMOKE MODE ===", flush=True)

    for d in [CKPT_DIR, PLOTS_DIR, VIDEOS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = LOGS_DIR / "run.log"
    log_mode = "w" if args.mode == "train" else "a"
    log_file = open(log_path, log_mode, buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} | mode={args.mode} ===", flush=True)
    print(f"args: {vars(args)}", flush=True)

    try:
        if args.mode == "train":
            train(args)
        elif args.mode == "eval":
            if not args.checkpoint:
                args.checkpoint = str(BEST_CKPT if BEST_CKPT.exists() else FINAL_CKPT)
            eval_mode(args)
        elif args.mode == "record":
            if not args.checkpoint:
                args.checkpoint = str(BEST_CKPT if BEST_CKPT.exists() else FINAL_CKPT)
            record_mode(args)
    finally:
        print(f"=== Done {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        log_file.close()


if __name__ == "__main__":
    main()
