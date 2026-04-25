#!/usr/bin/env python3
"""
Task 2: MBPO for HalfCheetah-v5.

This implementation keeps the real-environment interaction budget comparable
to Task 1 while using paper-aligned MBPO settings for HalfCheetah where the
available A800-class GPU makes them practical:

    paper HalfCheetah: N=400, E=1000, M=400, B=7, G=40, k=1

Here E, M, B, G, and k are preserved. N is reduced to match Task 1's
45,000 real environment transitions rather than the paper's 400,000.

Dependencies: gymnasium[mujoco] torch numpy matplotlib imageio imageio-ffmpeg
"""

# === Hyperparameters ===
SEED = 42

# Environment
ENV_ID = "HalfCheetah-v5"
OBS_DIM = None
ACT_DIM = None
ACTION_LOW = None
ACTION_HIGH = None
MAX_EPISODE_STEPS = None

# Real environment budget, matched to Task 1:
# task1 = 5000 random steps + 40 iterations * 1000 steps = 45000 real steps.
TOTAL_REAL_ENV_STEPS = 45_000
INIT_RANDOM_STEPS = 5_000
POLICY_REAL_ENV_STEPS = TOTAL_REAL_ENV_STEPS - INIT_RANDOM_STEPS

# Dynamics model / MBPO epoch cadence. Paper HalfCheetah uses E=1000.
MODEL_TRAIN_EVERY = 1_000
MODEL_HIDDEN = 200
MODEL_LAYERS = 4
ENSEMBLE_SIZE = 7
ELITE_SIZE = 5
MODEL_LR = 1e-3
MODEL_WD = 1e-5
MODEL_BATCH = 256
MODEL_MAX_EPOCHS = 80
MODEL_PATIENCE = 5
MODEL_VAL_FRAC = 0.1
MODEL_LOGVAR_REG = 0.01
MODEL_GRAD_CLIP = 100.0

# Model rollout settings. Paper HalfCheetah uses M=400, k=1.
ROLLOUT_BATCH = 400
ROLLOUT_K_MIN = 1
ROLLOUT_K_MAX = 1
ROLLOUT_SCHEDULE_START = 20
ROLLOUT_SCHEDULE_END = 100

# SAC. Paper HalfCheetah uses G=40 updates per real environment step.
SAC_HIDDEN = 256
SAC_LR = 3e-4
SAC_BATCH = 256
SAC_GAMMA = 0.99
SAC_TAU = 5e-3
SAC_UPDATES_PER_STEP = 40
REAL_RATIO = 0.05
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

# Replay
D_ENV_CAPACITY = int(1.5 * TOTAL_REAL_ENV_STEPS)
D_MODEL_CAPACITY = 400_000

# Evaluation / video
CHECKPOINT_EVAL_INTERVAL = 5_000
CHECKPOINT_EVAL_EPISODES = 3
CHECKPOINT_EVAL_SEED_OFFSET = 3_000
N_EVAL_EPISODES = 10
EVAL_VIDEO_EPISODES = 2
VIDEO_FPS = 20
VIDEO_MAX_FRAMES = None
MIN_VIDEO_SECONDS = 30.0

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SUBMISSION_DIR = Path(__file__).parent
PLOTS_DIR = SUBMISSION_DIR / "plots"
CKPT_PATH = SUBMISSION_DIR / "agent.pth"
BEST_CKPT_PATH = SUBMISSION_DIR / "best_agent.pth"
LOG_PATH = SUBMISSION_DIR / "train.log"


# =============================================================================
# Smoke-test overrides
# =============================================================================
_SMOKE = dict(
    TOTAL_REAL_ENV_STEPS=600,
    INIT_RANDOM_STEPS=100,
    MODEL_TRAIN_EVERY=200,
    MODEL_MAX_EPOCHS=3,
    ENSEMBLE_SIZE=3,
    ELITE_SIZE=2,
    ROLLOUT_BATCH=50,
    SAC_UPDATES_PER_STEP=2,
    MAX_EPISODE_STEPS=50,
    CHECKPOINT_EVAL_INTERVAL=200,
    CHECKPOINT_EVAL_EPISODES=2,
    N_EVAL_EPISODES=2,
    EVAL_VIDEO_EPISODES=1,
    VIDEO_MAX_FRAMES=20,
    MIN_VIDEO_SECONDS=0.0,
)


def _refresh_derived_constants():
    global POLICY_REAL_ENV_STEPS, D_ENV_CAPACITY
    POLICY_REAL_ENV_STEPS = max(0, TOTAL_REAL_ENV_STEPS - INIT_RANDOM_STEPS)
    D_ENV_CAPACITY = max(TOTAL_REAL_ENV_STEPS + 1, int(1.5 * TOTAL_REAL_ENV_STEPS))


def _apply_smoke():
    g = globals()
    for k, v in _SMOKE.items():
        g[k] = v
    _refresh_derived_constants()
    print("=== SMOKE MODE: shortened run for pipeline verification ===", flush=True)


# =============================================================================
# Logging and determinism helpers
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state():
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state(), cuda_state


def _restore_rng_state(state):
    py_state, np_state, torch_state, cuda_state = state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


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
    if (
        OBS_DIM is not None
        and ACT_DIM is not None
        and ACTION_LOW is not None
        and ACTION_HIGH is not None
    ):
        return

    env = make_env()
    try:
        OBS_DIM = int(env.observation_space.shape[0])
        ACT_DIM = int(env.action_space.shape[0])
        ACTION_LOW = env.action_space.low.astype(np.float32).copy()
        ACTION_HIGH = env.action_space.high.astype(np.float32).copy()
    finally:
        env.close()


def _action_bounds_tensors(device: torch.device):
    _detect_env_dims()
    return (
        torch.tensor(ACTION_LOW, dtype=torch.float32, device=device),
        torch.tensor(ACTION_HIGH, dtype=torch.float32, device=device),
    )


def _clip_action(act: np.ndarray) -> np.ndarray:
    _detect_env_dims()
    return np.clip(act, ACTION_LOW, ACTION_HIGH)


def _preferred_checkpoint_path() -> Path:
    return BEST_CKPT_PATH if BEST_CKPT_PATH.exists() else CKPT_PATH


# =============================================================================
# Replay buffer
# =============================================================================
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self._obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self._act = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self._nobs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self._rew = np.zeros((self.max_size, 1), dtype=np.float32)
        self._done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, obs, act, nobs, rew, done):
        self._obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self._act[self.ptr] = np.asarray(act, dtype=np.float32)
        self._nobs[self.ptr] = np.asarray(nobs, dtype=np.float32)
        self._rew[self.ptr, 0] = float(rew)
        self._done[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, act, nobs, rew, done):
        n = len(obs)
        if n <= 0:
            return
        idxs = np.arange(self.ptr, self.ptr + n) % self.max_size
        self._obs[idxs] = np.asarray(obs, dtype=np.float32)
        self._act[idxs] = np.asarray(act, dtype=np.float32)
        self._nobs[idxs] = np.asarray(nobs, dtype=np.float32)
        self._rew[idxs] = np.asarray(rew, dtype=np.float32).reshape(n, 1)
        self._done[idxs] = np.asarray(done, dtype=np.float32).reshape(n, 1)
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample_np(self, batch_size: int):
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self._obs[idx],
            self._act[idx],
            self._nobs[idx],
            self._rew[idx],
            self._done[idx],
        )

    def sample(self, batch_size: int, device: torch.device):
        obs, act, nobs, rew, done = self.sample_np(batch_size)
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=device),
            torch.as_tensor(act, dtype=torch.float32, device=device),
            torch.as_tensor(nobs, dtype=torch.float32, device=device),
            torch.as_tensor(rew, dtype=torch.float32, device=device),
            torch.as_tensor(done, dtype=torch.float32, device=device),
        )

    def sample_states(self, n: int) -> np.ndarray:
        if self.size <= 0:
            raise RuntimeError("Cannot sample states from an empty replay buffer.")
        idx = np.random.randint(0, self.size, size=n)
        return self._obs[idx].copy()

    def get_all(self):
        s = self.size
        return (
            self._obs[:s].copy(),
            self._act[:s].copy(),
            self._nobs[:s].copy(),
            self._rew[:s].copy(),
            self._done[:s].copy(),
        )


class Normalizer:
    def __init__(self, dim: int, device: torch.device):
        self.device = device
        self.mean = torch.zeros(dim, dtype=torch.float32, device=device)
        self.std = torch.ones(dim, dtype=torch.float32, device=device)

    def fit(self, data: np.ndarray):
        data_t = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        self.mean = data_t.mean(dim=0)
        self.std = data_t.std(dim=0, unbiased=False).clamp_min(1e-6)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.mean = state["mean"].to(self.device)
        self.std = state["std"].to(self.device)


# =============================================================================
# Probabilistic dynamics ensemble
# =============================================================================
def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, activation) -> nn.Sequential:
    mods: List[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), activation()]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class ProbabilisticDynamicsEnsemble(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble_size: int,
        elite_size: int,
        hidden: int,
        layers: int,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim + 1
        self.ensemble_size = ensemble_size
        self.elite_size = min(elite_size, ensemble_size)
        self.members = nn.ModuleList(
            [
                _mlp(obs_dim + act_dim, 2 * self.out_dim, hidden, layers, nn.SiLU)
                for _ in range(ensemble_size)
            ]
        )
        self.max_logvar = nn.Parameter(torch.ones(ensemble_size, self.out_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(ensemble_size, self.out_dim) * -10.0)
        self.elites: List[int] = list(range(self.elite_size))

    def _bound_logvar(self, raw_logvar: torch.Tensor, idx: int) -> torch.Tensor:
        max_lv = self.max_logvar[idx].view(1, -1)
        min_lv = self.min_logvar[idx].view(1, -1)
        logvar = max_lv - F.softplus(max_lv - raw_logvar)
        logvar = min_lv + F.softplus(logvar - min_lv)
        return logvar

    def forward_member_x(self, idx: int, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.members[idx](x_norm)
        mean, raw_logvar = out.chunk(2, dim=-1)
        return mean, self._bound_logvar(raw_logvar, idx)

    def forward_all_x(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means, logvars = [], []
        for idx in range(self.ensemble_size):
            mean, logvar = self.forward_member_x(idx, x_norm)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        normalizer: Normalizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = normalizer.norm(torch.cat([obs, act], dim=-1))
        means, logvars = self.forward_all_x(x)
        elite_ids = torch.tensor(self.elites, dtype=torch.long, device=obs.device)
        choices = elite_ids[torch.randint(0, len(self.elites), (obs.shape[0],), device=obs.device)]
        batch_idx = torch.arange(obs.shape[0], device=obs.device)
        mean = means[choices, batch_idx]
        logvar = logvars[choices, batch_idx]
        sample = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        sample = torch.nan_to_num(sample, nan=0.0, posinf=1e6, neginf=-1e6)
        delta = sample[:, : self.obs_dim].clamp(-1e6, 1e6)
        reward = sample[:, self.obs_dim : self.obs_dim + 1].clamp(-1e6, 1e6)
        return delta, reward


def train_ensemble(
    model: ProbabilisticDynamicsEnsemble,
    normalizer: Normalizer,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    obs, act, nobs, rew, _ = buffer.get_all()
    if len(obs) < 10:
        return [], [float("inf")] * model.ensemble_size

    x_np = np.concatenate([obs, act], axis=-1).astype(np.float32)
    y_np = np.concatenate([nobs - obs, rew], axis=-1).astype(np.float32)
    normalizer.fit(x_np)

    n = len(x_np)
    perm = np.random.permutation(n)
    n_val = max(1, int(n * MODEL_VAL_FRAC))
    val_idx_np = perm[:n_val]
    train_idx_np = perm[n_val:]
    if len(train_idx_np) == 0:
        train_idx_np = perm

    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    x_norm = normalizer.norm(x).detach()
    val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=device)
    x_val = x_norm[val_idx]
    y_val = y[val_idx]

    bootstrap_indices = []
    train_len = len(train_idx_np)
    for _ in range(model.ensemble_size):
        boot = np.random.choice(train_idx_np, size=train_len, replace=True)
        bootstrap_indices.append(torch.as_tensor(boot, dtype=torch.long, device=device))

    train_history: List[float] = []
    best_score = float("inf")
    best_state = None
    wait = 0

    for epoch in range(MODEL_MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        updates = 0
        for member_idx in range(model.ensemble_size):
            member_idxes = bootstrap_indices[member_idx]
            member_perm = torch.randperm(len(member_idxes), device=device)
            for start in range(0, len(member_idxes), MODEL_BATCH):
                batch = member_idxes[member_perm[start : start + MODEL_BATCH]]
                mean, logvar = model.forward_member_x(member_idx, x_norm[batch])
                inv_var = torch.exp(-logvar)
                nll = 0.5 * (((mean - y[batch]).square() * inv_var) + logvar).sum(dim=-1).mean()
                logvar_reg = MODEL_LOGVAR_REG * (
                    model.max_logvar[member_idx].sum() - model.min_logvar[member_idx].sum()
                )
                loss = nll + logvar_reg
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MODEL_GRAD_CLIP)
                optimizer.step()
                epoch_loss += float(loss.item())
                updates += 1

        mean_train_loss = epoch_loss / max(updates, 1)
        train_history.append(mean_train_loss)

        val_mses = evaluate_model_holdout(model, x_val, y_val)
        val_score = float(np.mean(val_mses))
        if val_score < best_score - 1e-6:
            best_score = val_score
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= MODEL_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_mses = evaluate_model_holdout(model, x_val, y_val)
    elite_count = min(ELITE_SIZE, model.ensemble_size)
    model.elites = np.argsort(final_val_mses)[:elite_count].astype(int).tolist()
    return train_history, final_val_mses


@torch.no_grad()
def evaluate_model_holdout(
    model: ProbabilisticDynamicsEnsemble,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> List[float]:
    model.eval()
    out = []
    for member_idx in range(model.ensemble_size):
        mean, _ = model.forward_member_x(member_idx, x_val)
        mse = (mean - y_val).square().mean().item()
        out.append(mse)
    return out


# =============================================================================
# SAC
# =============================================================================
class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.body = _mlp(obs_dim, hidden, hidden, 2, nn.ReLU)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - action.square() + 1e-6).sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mu)
        return action, log_prob, deterministic


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, 1, hidden, 2, nn.ReLU)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, device: torch.device):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.device = device
        self.actor = GaussianActor(obs_dim, act_dim, hidden).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=SAC_LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=SAC_LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=SAC_LR)
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=SAC_LR)
        self.target_entropy = -float(act_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, det = self.actor.sample(obs_t)
        out = det if deterministic else action
        return _clip_action(out.squeeze(0).cpu().numpy())

    def _sample_training_batch(
        self,
        real_buffer: ReplayBuffer,
        model_buffer: ReplayBuffer,
        batch_size: int,
        real_ratio: float,
    ):
        if model_buffer.size <= 0:
            return real_buffer.sample(batch_size, self.device)

        real_bs = int(round(batch_size * real_ratio))
        real_bs = min(max(real_bs, 1), batch_size - 1)
        model_bs = batch_size - real_bs
        rb = real_buffer.sample(real_bs, self.device)
        mb = model_buffer.sample(model_bs, self.device)
        return tuple(torch.cat([r, m], dim=0) for r, m in zip(rb, mb))

    def update(
        self,
        real_buffer: ReplayBuffer,
        model_buffer: ReplayBuffer,
        batch_size: int = SAC_BATCH,
        real_ratio: float = REAL_RATIO,
    ) -> Dict[str, float]:
        obs, act, nobs, rew, done = self._sample_training_batch(
            real_buffer, model_buffer, batch_size, real_ratio
        )

        with torch.no_grad():
            next_act, next_logp, _ = self.actor.sample(nobs)
            tq1 = self.q1_target(nobs, next_act)
            tq2 = self.q2_target(nobs, next_act)
            target_q = torch.min(tq1, tq2) - self.alpha.detach() * next_logp
            target = rew + SAC_GAMMA * (1.0 - done) * target_q

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        new_act, logp, _ = self.actor.sample(obs)
        q_new = torch.min(self.q1(obs, new_act), self.q2(obs, new_act))
        actor_loss = (self.alpha.detach() * logp - q_new).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "critic_loss": float((q1_loss + q2_loss).detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "q_value": float(q_new.detach().mean().cpu().item()),
        }

    @staticmethod
    def _soft_update(src: nn.Module, tgt: nn.Module):
        with torch.no_grad():
            for p, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.mul_(1.0 - SAC_TAU)
                tp.data.add_(SAC_TAU * p.data)


@torch.no_grad()
def generate_model_rollouts(
    agent: SACAgent,
    ensemble: ProbabilisticDynamicsEnsemble,
    normalizer: Normalizer,
    env_buffer: ReplayBuffer,
    model_buffer: ReplayBuffer,
    device: torch.device,
    rollout_length: int,
) -> int:
    if env_buffer.size <= 0 or rollout_length <= 0:
        return 0

    ensemble.eval()
    states = torch.as_tensor(
        env_buffer.sample_states(ROLLOUT_BATCH), dtype=torch.float32, device=device
    )
    total_added = 0
    for _ in range(rollout_length):
        actions, _, _ = agent.actor.sample(states)
        deltas, rewards = ensemble.sample(states, actions, normalizer)
        next_states = torch.nan_to_num(states + deltas, nan=0.0, posinf=1e6, neginf=-1e6)
        next_states = next_states.clamp(-1e6, 1e6)
        dones = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=device)
        model_buffer.add_batch(
            states.detach().cpu().numpy(),
            actions.detach().cpu().numpy(),
            next_states.detach().cpu().numpy(),
            rewards.detach().cpu().numpy(),
            dones.detach().cpu().numpy(),
        )
        total_added += states.shape[0]
        states = next_states
    return total_added


def rollout_length_for_epoch(epoch_idx: int) -> int:
    if ROLLOUT_K_MIN == ROLLOUT_K_MAX:
        return int(ROLLOUT_K_MIN)
    denom = max(1, ROLLOUT_SCHEDULE_END - ROLLOUT_SCHEDULE_START)
    frac = (epoch_idx - ROLLOUT_SCHEDULE_START) / denom
    val = ROLLOUT_K_MIN + np.clip(frac, 0.0, 1.0) * (ROLLOUT_K_MAX - ROLLOUT_K_MIN)
    return int(round(np.clip(val, ROLLOUT_K_MIN, ROLLOUT_K_MAX)))


# =============================================================================
# Checkpoint I/O
# =============================================================================
def _hyperparams_snapshot() -> Dict[str, object]:
    keys = [
        "SEED",
        "ENV_ID",
        "MAX_EPISODE_STEPS",
        "TOTAL_REAL_ENV_STEPS",
        "INIT_RANDOM_STEPS",
        "MODEL_TRAIN_EVERY",
        "ENSEMBLE_SIZE",
        "ELITE_SIZE",
        "ROLLOUT_BATCH",
        "ROLLOUT_K_MIN",
        "ROLLOUT_K_MAX",
        "SAC_UPDATES_PER_STEP",
        "REAL_RATIO",
        "SAC_BATCH",
        "SAC_GAMMA",
        "SAC_TAU",
    ]
    return {k: globals()[k] for k in keys}


def save_checkpoint(agent: SACAgent, path: Path, real_env_steps: int, best_eval: float):
    _detect_env_dims()
    payload = {
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_target": agent.q1_target.state_dict(),
        "q2_target": agent.q2_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "action_low": torch.tensor(ACTION_LOW, dtype=torch.float32),
        "action_high": torch.tensor(ACTION_HIGH, dtype=torch.float32),
        "sac_hidden": agent.hidden,
        "target_entropy": agent.target_entropy,
        "real_env_steps": int(real_env_steps),
        "best_eval_return": float(best_eval),
        "hyperparams": _hyperparams_snapshot(),
    }
    torch.save(payload, str(path))
    print(f"Checkpoint saved to {path}", flush=True)


def load_checkpoint(path: Path, device: torch.device) -> SACAgent:
    global OBS_DIM, ACT_DIM, ACTION_LOW, ACTION_HIGH
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    OBS_DIM = int(ckpt["obs_dim"])
    ACT_DIM = int(ckpt["act_dim"])
    ACTION_LOW = ckpt["action_low"].detach().cpu().numpy().astype(np.float32)
    ACTION_HIGH = ckpt["action_high"].detach().cpu().numpy().astype(np.float32)
    hidden = int(ckpt.get("sac_hidden", SAC_HIDDEN))
    agent = SACAgent(OBS_DIM, ACT_DIM, hidden, device)
    agent.actor.load_state_dict(ckpt["actor"])
    if "q1" in ckpt:
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        agent.q1_target.load_state_dict(ckpt["q1_target"])
        agent.q2_target.load_state_dict(ckpt["q2_target"])
    if "log_alpha" in ckpt:
        agent.log_alpha.data.copy_(ckpt["log_alpha"].to(device))
    agent.actor.eval()
    agent.q1.eval()
    agent.q2.eval()
    print(f"Loaded checkpoint from {path}", flush=True)
    return agent


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
    ax.plot(
        returns,
        marker="o",
        markersize=3,
        linewidth=1.2,
        alpha=0.75,
        color="steelblue",
        label="Raw episode return",
    )
    ma = _smooth(returns, smooth)
    if len(returns) >= smooth:
        ax.plot(
            range(smooth - 1, len(returns)),
            ma,
            color="darkorange",
            lw=2,
            label=f"{smooth}-episode moving average",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("MBPO - HalfCheetah-v5 Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}", flush=True)


def _plot_series(x, y, out: Path, title: str, xlabel: str, ylabel: str, color: str):
    if not y:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    if x is None:
        ax.plot(y, color=color, lw=1.2)
    else:
        ax.plot(x, y, color=color, lw=1.2, marker="o", markersize=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def save_all_curves(history: Dict, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_rewards(history["episode_returns"], plots_dir / "reward_curve.jpg")
    _plot_series(
        history["episode_end_steps"],
        history["episode_returns"],
        plots_dir / "reward_vs_env_steps.jpg",
        "MBPO Training Reward vs Real Environment Steps",
        "Real environment steps",
        "Episode return",
        "steelblue",
    )
    _plot_series(
        history["eval_steps"],
        history["eval_mean_returns"],
        plots_dir / "eval_return_vs_env_steps.jpg",
        "MBPO Deterministic Eval Return vs Real Environment Steps",
        "Real environment steps",
        "Mean eval return",
        "darkorange",
    )
    _plot_series(
        None,
        history["model_losses"],
        plots_dir / "model_loss.jpg",
        "MBPO Dynamics Ensemble Training Loss",
        "Model training epoch",
        "Gaussian NLL",
        "tomato",
    )
    _plot_series(
        history["sac_metric_steps"],
        history["critic_losses"],
        plots_dir / "critic_loss.jpg",
        "SAC Critic Loss",
        "Real environment steps",
        "Loss",
        "firebrick",
    )
    _plot_series(
        history["sac_metric_steps"],
        history["actor_losses"],
        plots_dir / "actor_loss.jpg",
        "SAC Actor Loss",
        "Real environment steps",
        "Loss",
        "seagreen",
    )
    _plot_series(
        history["sac_metric_steps"],
        history["q_values"],
        plots_dir / "q_value.jpg",
        "SAC Q-Value Estimate",
        "Real environment steps",
        "Mean Q",
        "purple",
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    axes[0].plot(history["episode_end_steps"], history["episode_returns"], color="steelblue", lw=1)
    axes[0].set_title("Training Return")
    axes[0].set_xlabel("Real env steps")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["model_losses"], color="tomato", lw=1)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Model epoch")
    axes[1].grid(alpha=0.3)
    axes[2].plot(history["sac_metric_steps"], history["critic_losses"], color="firebrick", lw=1)
    axes[2].set_title("Critic Loss")
    axes[2].set_xlabel("Real env steps")
    axes[2].grid(alpha=0.3)
    axes[3].plot(history["sac_metric_steps"], history["actor_losses"], color="seagreen", lw=1)
    axes[3].set_title("Actor Loss")
    axes[3].set_xlabel("Real env steps")
    axes[3].grid(alpha=0.3)
    fig.suptitle("MBPO Diagnostics - HalfCheetah-v5")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "diagnostics_combined.jpg"), dpi=150)
    plt.close(fig)
    print(f"All diagnostic curves saved to {plots_dir}/", flush=True)


# =============================================================================
# Evaluation and video
# =============================================================================
@torch.no_grad()
def evaluate_policy(
    agent: SACAgent,
    n_episodes: int,
    seed_offset: int,
    label: str,
    deterministic: bool = True,
) -> List[float]:
    print(f"\n[{label}] Running {n_episodes} deterministic episode(s)...", flush=True)
    saved_rng = _capture_rng_state() if deterministic else None
    rewards: List[float] = []
    env = make_env()
    try:
        for i in range(n_episodes):
            ep_seed = SEED + seed_offset + i
            if deterministic:
                set_seed(ep_seed)
            obs, _ = env.reset(seed=ep_seed)
            done = False
            ep_ret = 0.0
            while not done:
                act = agent.select_action(obs, deterministic=deterministic)
                obs, rew, term, trunc, _ = env.step(act)
                done = bool(term or trunc)
                ep_ret += float(rew)
            rewards.append(ep_ret)
            print(f"  {label} ep {i + 1}: return={ep_ret:.2f}", flush=True)
    finally:
        env.close()
        if saved_rng is not None:
            _restore_rng_state(saved_rng)

    if rewards:
        arr = np.array(rewards, dtype=np.float64)
        print(
            f"{label}: mean={arr.mean():.2f} +/- {arr.std():.2f}, "
            f"min={arr.min():.2f}, max={arr.max():.2f}",
            flush=True,
        )
    return rewards


def evaluate_and_record(agent: SACAgent):
    rewards = evaluate_policy(
        agent,
        N_EVAL_EPISODES,
        seed_offset=1_000,
        label="Eval",
        deterministic=True,
    )

    min_frames = int(math.ceil(MIN_VIDEO_SECONDS * VIDEO_FPS))
    print(
        f"\n[Video] Recording at least {EVAL_VIDEO_EPISODES} episode(s) "
        f"and {MIN_VIDEO_SECONDS:.1f}s...",
        flush=True,
    )
    env = make_env(render_mode="rgb_array")
    frames: List[np.ndarray] = []
    ep = 0
    try:
        while ep < EVAL_VIDEO_EPISODES or len(frames) < min_frames:
            ep_seed = SEED + 200 + ep
            set_seed(ep_seed)
            obs, _ = env.reset(seed=ep_seed)
            done = False
            ep_ret = 0.0
            ep_frames = 0
            while not done:
                act = agent.select_action(obs, deterministic=True)
                obs, rew, term, trunc, _ = env.step(act)
                done = bool(term or trunc)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                ep_ret += float(rew)
                ep_frames += 1
                if VIDEO_MAX_FRAMES is not None and ep_frames >= VIDEO_MAX_FRAMES:
                    break
            print(f"  Video ep {ep + 1}: return={ep_ret:.2f}, frames={ep_frames}", flush=True)
            ep += 1
            if ep_frames == 0:
                break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No video frames were rendered.")

    out_path = SUBMISSION_DIR / "Video.mp4"
    frames_u8 = [f.astype(np.uint8) for f in frames]
    try:
        with imageio.get_writer(str(out_path), fps=VIDEO_FPS, codec="libx264", quality=7) as writer:
            for f in frames_u8:
                writer.append_data(f)
    except Exception:
        imageio.mimsave(str(out_path), frames_u8, fps=VIDEO_FPS)

    duration = len(frames) / VIDEO_FPS
    print(f"Video: {out_path} ({len(frames)} frames, {duration:.1f}s)", flush=True)
    if MIN_VIDEO_SECONDS > 0 and duration < MIN_VIDEO_SECONDS:
        raise RuntimeError(
            f"Video duration is {duration:.1f}s; expected at least {MIN_VIDEO_SECONDS:.1f}s"
        )
    return rewards, duration


# =============================================================================
# Training
# =============================================================================
def _empty_history() -> Dict[str, List[float]]:
    return dict(
        episode_returns=[],
        episode_end_steps=[],
        eval_steps=[],
        eval_mean_returns=[],
        model_losses=[],
        model_val_mses=[],
        sac_metric_steps=[],
        critic_losses=[],
        actor_losses=[],
        q_values=[],
        alphas=[],
    )


def collect_random(
    env: gym.Env,
    buffer: ReplayBuffer,
    n_steps: int,
    history: Dict[str, List[float]],
) -> Tuple[np.ndarray, float, int]:
    obs, _ = env.reset(seed=SEED)
    try:
        env.action_space.seed(SEED)
    except Exception:
        pass
    ep_ret = 0.0
    ep_len = 0
    completed = 0
    for step in range(1, n_steps + 1):
        act = env.action_space.sample()
        nobs, rew, term, trunc, _ = env.step(act)
        sac_done = float(term)
        buffer.add(obs, act, nobs, rew, sac_done)
        ep_ret += float(rew)
        ep_len += 1
        obs = nobs
        if term or trunc:
            completed += 1
            history["episode_returns"].append(ep_ret)
            history["episode_end_steps"].append(step)
            ep_ret = 0.0
            ep_len = 0
            obs, _ = env.reset()
    print(
        f"  Random buffer={buffer.size} transitions | completed_eps={completed} | "
        f"open_ep_len={ep_len}",
        flush=True,
    )
    return obs, ep_ret, ep_len


def train():
    set_seed(SEED)
    _refresh_derived_constants()
    _detect_env_dims()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | OBS={OBS_DIM} ACT={ACT_DIM}", flush=True)
    print(
        "Paper HalfCheetah reference: N=400, E=1000, M=400, B=7, G=40, k=1",
        flush=True,
    )
    print(
        f"Comparable real budget: total={TOTAL_REAL_ENV_STEPS}, "
        f"random={INIT_RANDOM_STEPS}, policy={POLICY_REAL_ENV_STEPS}",
        flush=True,
    )
    print(
        f"MBPO config: E={MODEL_TRAIN_EVERY}, M={ROLLOUT_BATCH}, "
        f"B={ENSEMBLE_SIZE}, elites={ELITE_SIZE}, G={SAC_UPDATES_PER_STEP}, "
        f"k={ROLLOUT_K_MIN}->{ROLLOUT_K_MAX}",
        flush=True,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    history = _empty_history()
    real_buffer = ReplayBuffer(OBS_DIM, ACT_DIM, D_ENV_CAPACITY)
    model_buffer = ReplayBuffer(OBS_DIM, ACT_DIM, D_MODEL_CAPACITY)
    agent = SACAgent(OBS_DIM, ACT_DIM, SAC_HIDDEN, device)
    ensemble = ProbabilisticDynamicsEnsemble(
        OBS_DIM, ACT_DIM, ENSEMBLE_SIZE, ELITE_SIZE, MODEL_HIDDEN, MODEL_LAYERS
    ).to(device)
    normalizer = Normalizer(OBS_DIM + ACT_DIM, device)
    model_optimizer = optim.Adam(ensemble.parameters(), lr=MODEL_LR, weight_decay=MODEL_WD)

    best_eval_return = -float("inf")
    synthetic_transitions = 0
    sac_updates = 0
    t_start = time.time()

    env = make_env()
    try:
        print(f"\n[Phase 0] Collecting {INIT_RANDOM_STEPS} random transitions...", flush=True)
        obs, ep_ret, ep_len = collect_random(env, real_buffer, INIT_RANDOM_STEPS, history)

        print(f"\n[Initial model train] D_env={real_buffer.size}", flush=True)
        train_losses, val_mses = train_ensemble(
            ensemble, normalizer, model_optimizer, real_buffer, device
        )
        history["model_losses"].extend(train_losses)
        history["model_val_mses"].append(float(np.mean(val_mses)))
        print(
            f"  Model: epochs={len(train_losses)} "
            f"last_loss={train_losses[-1] if train_losses else float('nan'):.4f} "
            f"val_mse={np.mean(val_mses):.6f} elites={ensemble.elites}",
            flush=True,
        )
        save_checkpoint(agent, CKPT_PATH, INIT_RANDOM_STEPS, best_eval_return)

        def run_checkpoint_eval(tag: str, real_steps: int):
            nonlocal best_eval_return
            eval_rets = evaluate_policy(
                agent,
                CHECKPOINT_EVAL_EPISODES,
                CHECKPOINT_EVAL_SEED_OFFSET,
                tag,
                deterministic=True,
            )
            if not eval_rets:
                return
            eval_mean = float(np.mean(eval_rets))
            history["eval_steps"].append(real_steps)
            history["eval_mean_returns"].append(eval_mean)
            if eval_mean > best_eval_return:
                best_eval_return = eval_mean
                print(
                    f"  ** New best eval_mean_return={best_eval_return:.2f}; "
                    "saving best checkpoint",
                    flush=True,
                )
                save_checkpoint(agent, BEST_CKPT_PATH, real_steps, best_eval_return)
            else:
                print(f"  Best eval_mean_return remains {best_eval_return:.2f}", flush=True)
            save_checkpoint(agent, CKPT_PATH, real_steps, best_eval_return)

        print(f"\n[MBPO training] {POLICY_REAL_ENV_STEPS} policy real env steps...", flush=True)
        for policy_step in range(1, POLICY_REAL_ENV_STEPS + 1):
            real_steps = INIT_RANDOM_STEPS + policy_step
            action = agent.select_action(obs, deterministic=False)
            nobs, rew, term, trunc, _ = env.step(action)
            sac_done = float(term)
            real_buffer.add(obs, action, nobs, rew, sac_done)
            ep_ret += float(rew)
            ep_len += 1
            obs = nobs

            if term or trunc:
                history["episode_returns"].append(ep_ret)
                history["episode_end_steps"].append(real_steps)
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_len = 0

            epoch_idx = policy_step // max(MODEL_TRAIN_EVERY, 1)
            k = rollout_length_for_epoch(epoch_idx)
            synthetic_transitions += generate_model_rollouts(
                agent, ensemble, normalizer, real_buffer, model_buffer, device, k
            )

            step_metrics = []
            for _ in range(SAC_UPDATES_PER_STEP):
                metrics = agent.update(real_buffer, model_buffer, SAC_BATCH, REAL_RATIO)
                step_metrics.append(metrics)
                sac_updates += 1
            if step_metrics:
                history["sac_metric_steps"].append(real_steps)
                history["critic_losses"].append(float(np.mean([m["critic_loss"] for m in step_metrics])))
                history["actor_losses"].append(float(np.mean([m["actor_loss"] for m in step_metrics])))
                history["q_values"].append(float(np.mean([m["q_value"] for m in step_metrics])))
                history["alphas"].append(float(np.mean([m["alpha"] for m in step_metrics])))

            if policy_step % MODEL_TRAIN_EVERY == 0:
                print(
                    f"\n[Model train @ real_step={real_steps}] "
                    f"D_env={real_buffer.size} D_model={model_buffer.size}",
                    flush=True,
                )
                train_losses, val_mses = train_ensemble(
                    ensemble, normalizer, model_optimizer, real_buffer, device
                )
                history["model_losses"].extend(train_losses)
                history["model_val_mses"].append(float(np.mean(val_mses)))
                print(
                    f"  Model: epochs={len(train_losses)} "
                    f"last_loss={train_losses[-1] if train_losses else float('nan'):.4f} "
                    f"val_mse={np.mean(val_mses):.6f} elites={ensemble.elites}",
                    flush=True,
                )

            if policy_step % CHECKPOINT_EVAL_INTERVAL == 0:
                elapsed = time.time() - t_start
                recent_returns = history["episode_returns"][-5:]
                recent_avg = float(np.mean(recent_returns)) if recent_returns else 0.0
                print(
                    f"\n[Progress] real_steps={real_steps}/{TOTAL_REAL_ENV_STEPS} "
                    f"D_env={real_buffer.size} D_model={model_buffer.size} "
                    f"synthetic={synthetic_transitions} sac_updates={sac_updates} "
                    f"recent_return={recent_avg:.1f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                run_checkpoint_eval(f"Checkpoint eval step {real_steps}", real_steps)

        final_real_steps = TOTAL_REAL_ENV_STEPS
        if ep_len > 0:
            print(
                f"  Open final training episode: len={ep_len}, partial_return={ep_ret:.2f}",
                flush=True,
            )

        print(f"\n[Final model train] D_env={real_buffer.size}", flush=True)
        train_losses, val_mses = train_ensemble(
            ensemble, normalizer, model_optimizer, real_buffer, device
        )
        history["model_losses"].extend(train_losses)
        history["model_val_mses"].append(float(np.mean(val_mses)))
        print(
            f"  Final model: epochs={len(train_losses)} "
            f"val_mse={np.mean(val_mses):.6f} elites={ensemble.elites}",
            flush=True,
        )

        run_checkpoint_eval("Final checkpoint eval", final_real_steps)
        save_checkpoint(agent, CKPT_PATH, final_real_steps, best_eval_return)
    finally:
        env.close()

    print("\n[Plotting] ...", flush=True)
    plot_rewards(history["episode_returns"], SUBMISSION_DIR / "Plot.jpg")
    save_all_curves(history, PLOTS_DIR)

    print(
        f"Training complete: real_steps={TOTAL_REAL_ENV_STEPS}, "
        f"synthetic_transitions={synthetic_transitions}, sac_updates={sac_updates}, "
        f"best_eval={best_eval_return:.2f}",
        flush=True,
    )
    return agent, history


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Short sanity-check run")
    parser.add_argument("--eval-only", action="store_true", help="Load checkpoint and record video")
    args = parser.parse_args()

    if args.smoke:
        _apply_smoke()
    else:
        _refresh_derived_constants()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log_mode = "a" if args.eval_only else "w"
    log_file = open(LOG_PATH, log_mode, buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"=== Run started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"train.log mode: {log_mode}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    try:
        set_seed(SEED)
        _detect_env_dims()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.eval_only:
            ckpt_path = _preferred_checkpoint_path()
            if not ckpt_path.exists():
                print(
                    f"ERROR: no checkpoint found at {BEST_CKPT_PATH} or {CKPT_PATH}. Train first.",
                    flush=True,
                )
                sys.exit(1)
            agent = load_checkpoint(ckpt_path, device)
            evaluate_and_record(agent)
        else:
            train()
            ckpt_path = _preferred_checkpoint_path()
            print(f"Reloading {ckpt_path.name} for final evaluation", flush=True)
            agent = load_checkpoint(ckpt_path, device)
            evaluate_and_record(agent)
    finally:
        print(f"\n=== Run finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        log_file.close()


if __name__ == "__main__":
    main()
