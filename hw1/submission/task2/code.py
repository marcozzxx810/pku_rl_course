"""
Task 2: DDQN (Double Deep Q-Network) for LunarLander-v3
=========================================================
Modification from DQN: decouples action selection and evaluation to reduce
Q-value overestimation:

  DQN   target: r + γ * max_a Q_target(s', a)
  DDQN  target: r + γ * Q_target(s', argmax_a Q_online(s', a))

All other components (replay buffer, target network, ε-greedy, etc.) are
identical to Task 1.
"""

import os
import random
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
ENV_NAME       = "LunarLander-v3"
EPISODES       = 1000
BATCH_SIZE     = 64
BUFFER_SIZE    = 100_000
GAMMA          = 0.99
LR             = 1e-4
EPS_START      = 1.0
EPS_MIN        = 0.01
EPS_DECAY      = 0.995
TARGET_UPDATE  = 1000
HIDDEN         = 128
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Q-Network ────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Replay Buffer ────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32, device=DEVICE),
            torch.tensor(actions,               dtype=torch.long,    device=DEVICE),
            torch.tensor(rewards,               dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(dones,                 dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ─── DDQN Agent ───────────────────────────────────────────────────────────────
class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.epsilon = EPS_START
        self.steps = 0

        self.online = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)
        self.buffer = ReplayBuffer()

    def choose_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(state_t).argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Current Q-values for taken actions
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── DDQN target ──────────────────────────────────────────────────────
        # Online network selects the best action in next state
        with torch.no_grad():
            best_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            # Target network evaluates that action
            next_q = self.target(next_states).gather(1, best_actions).squeeze(1)
            targets = rewards + GAMMA * next_q * (1 - dones)
        # ─────────────────────────────────────────────────────────────────────

        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    def get_initial_q(self, state: np.ndarray) -> float:
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return float(self.online(state_t).max().item())


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim)

    rewards_history   = []
    initial_q_history = []
    avg_max_q_history = []

    print(f"Training DDQN on {ENV_NAME} for {EPISODES} episodes  [device={DEVICE}]")

    for ep in range(EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        initial_q_history.append(agent.get_initial_q(state))

        total_reward = 0.0
        step_max_qs  = []
        done = truncated = False

        while not (done or truncated):
            step_max_qs.append(agent.get_initial_q(state))
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done or truncated)
            agent.update()
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        avg_max_q_history.append(float(np.mean(step_max_qs)) if step_max_qs else 0.0)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-50:])
            print(f"  Episode {ep+1:4d}/{EPISODES}  avg_reward(50)={avg_r:8.2f}  eps={agent.epsilon:.3f}")

    env.close()
    return agent, rewards_history, initial_q_history, avg_max_q_history


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_rewards(rewards, filename):
    smooth = np.convolve(rewards, np.ones(50) / 50, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Episode reward')
    plt.plot(range(49, len(rewards)), smooth, label='Smoothed (50-ep)')
    plt.title('DDQN Training Reward — LunarLander-v3')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_q_values(initial_q, avg_max_q, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(initial_q,  alpha=0.5, label='Initial-state max Q')
    plt.plot(avg_max_q, alpha=0.5, label='Avg max Q per episode', color='green')
    plt.title('Q-value Curves — DDQN LunarLander-v3')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ─── Video Recording ──────────────────────────────────────────────────────────
def record_video(agent, video_path, min_seconds=35):
    """Record multiple episodes until total duration >= min_seconds, then concatenate."""
    import glob, shutil
    from moviepy import VideoFileClip, concatenate_videoclips

    video_dir = os.path.join(OUT_DIR, "_video_tmp")
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True,
                      name_prefix="ddqn")

    total_steps = 0
    fps = env.metadata.get("render_fps", 50)
    min_steps = min_seconds * fps
    ep = 0

    while total_steps < min_steps:
        state, _ = env.reset(seed=ep)
        done = truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action = agent.choose_action(state, explore=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            total_steps += 1
        print(f"  Recording ep {ep+1}: reward={ep_reward:.1f}, total_steps={total_steps}")
        ep += 1

    env.close()

    # Concatenate all recorded clips into one video
    clips_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    clips = [VideoFileClip(p) for p in clips_paths]
    final = concatenate_videoclips(clips)
    final.write_videofile(video_path, logger=None)
    for c in clips:
        c.close()
    final.close()
    shutil.rmtree(video_dir, ignore_errors=True)
    print(f"Saved: {video_path}  ({final.duration:.1f}s)")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent, rewards, init_q, avg_q = train()

    plot_rewards(rewards,         os.path.join(OUT_DIR, "plot1.jpg"))
    plot_q_values(init_q, avg_q,  os.path.join(OUT_DIR, "plot2.jpg"))
    record_video(agent,           os.path.join(OUT_DIR, "video.mp4"))

    print("\nDone. Artifacts saved to:", OUT_DIR)
