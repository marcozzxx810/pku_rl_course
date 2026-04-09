# === Hyperparameters ===
TOTAL_TIMESTEPS = 2_000_000
ROLLOUT_STEPS   = 2048
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
K_EPOCHS        = 10
MINIBATCH_SIZE  = 64
LR              = 3e-4
VALUE_COEFF     = 0.5
ENTROPY_COEFF   = 0.01
MAX_GRAD_NORM   = 0.5
HIDDEN_DIM      = 256
SEED            = 42
EVAL_EPISODES   = 5   # episodes to record for the video

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# === Neural Networks ===

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std   = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        features = self.net(x)
        mean = torch.tanh(self.mean_head(features))
        std  = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, x):
        mean, std = self.forward(x)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# === Rollout Buffer ===

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.dones     = []
        self.values    = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, last_value, gamma, gae_lambda):
        rewards   = np.array(self.rewards,   dtype=np.float32)
        dones     = np.array(self.dones,     dtype=np.float32)
        values    = np.array(self.values,    dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_done  = dones[t]
            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            gae   = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def get_tensors(self, device):
        states    = torch.FloatTensor(np.array(self.states)).to(device)
        actions   = torch.FloatTensor(np.array(self.actions)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        return states, actions, log_probs


# === PPO Agent ===

class PPOAgent:
    def __init__(self, obs_dim, act_dim, device):
        self.device = device
        self.actor  = Actor(obs_dim, act_dim, HIDDEN_DIM).to(device)
        self.critic = Critic(obs_dim, HIDDEN_DIM).to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LR,
        )

    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist    = self.actor.get_dist(state_t)
        action  = dist.mean if deterministic else dist.sample()
        action  = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        value   = self.critic(state_t)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def update(self, buffer, last_value):
        advantages, returns = buffer.compute_gae(last_value, GAMMA, GAE_LAMBDA)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns    = torch.FloatTensor(returns).to(self.device)

        states, actions, old_log_probs = buffer.get_tensors(self.device)

        n = len(states)
        indices = np.arange(n)

        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, n, MINIBATCH_SIZE):
                mb_idx = indices[start:start + MINIBATCH_SIZE]
                mb_states    = states[mb_idx]
                mb_actions   = actions[mb_idx]
                mb_old_lp    = old_log_probs[mb_idx]
                mb_adv       = advantages[mb_idx]
                mb_returns   = returns[mb_idx]

                # Normalize advantages within minibatch
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                dist     = self.actor.get_dist(mb_states)
                new_lp   = dist.log_prob(mb_actions).sum(-1)
                entropy  = dist.entropy().sum(-1).mean()
                values   = self.critic(mb_states)

                ratio    = (new_lp - mb_old_lp).exp()
                surr1    = ratio * mb_adv
                surr2    = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (values - mb_returns).pow(2).mean()

                loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    MAX_GRAD_NORM,
                )
                self.optimizer.step()


# === Training ===

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = gym.make("BipedalWalker-v3")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent  = PPOAgent(obs_dim, act_dim, device)
    buffer = RolloutBuffer()

    episode_rewards = []
    ep_reward = 0.0
    state, _ = env.reset(seed=SEED)
    total_steps = 0

    while total_steps < TOTAL_TIMESTEPS:
        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, log_prob, reward, float(done), value)
            state = next_state
            ep_reward += reward
            total_steps += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                state, _ = env.reset()

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            last_value = agent.critic(state_t).item()

        agent.update(buffer, last_value)

        if episode_rewards:
            recent = np.mean(episode_rewards[-10:])
            print(f"Steps: {total_steps:>8d} | Episodes: {len(episode_rewards):>5d} | "
                  f"Avg reward (last 10): {recent:>8.2f}")

    env.close()
    print("Training complete.")

    weights_path = os.path.join(SUBMISSION_DIR, "agent.pth")
    torch.save({
        "actor":  agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    }, weights_path)
    print(f"Weights saved to {weights_path}")

    return agent, episode_rewards


# === Plotting ===

def plot_rewards(episode_rewards):
    path = os.path.join(SUBMISSION_DIR, "reward_curve.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.4, color="steelblue", label="Episode reward")
    if len(episode_rewards) >= 10:
        window = 10
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), smoothed,
                color="darkorange", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("PPO on BipedalWalker-v3 — Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to {path}")


# === Evaluation + Video ===

def evaluate_and_record(agent):
    device = agent.device
    video_dir = os.path.join(SUBMISSION_DIR, "video")
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix="eval",
    )

    total_frames = 0
    ep = 0
    while ep < EVAL_EPISODES or total_frames < 1500:
        state, _ = env.reset(seed=SEED + ep)
        done = False
        ep_frames = 0
        ep_reward = 0.0
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_frames += 1
        total_frames += ep_frames
        ep += 1
        print(f"Eval episode {ep}: reward={ep_reward:.2f}, frames={ep_frames}")

    env.close()
    print(f"Video(s) saved to {video_dir}  (total frames: {total_frames})")


# === Main ===

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved weights and record video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(SUBMISSION_DIR, "agent.pth")

    if args.eval_only:
        base_env = gym.make("BipedalWalker-v3")
        obs_dim = base_env.observation_space.shape[0]
        act_dim = base_env.action_space.shape[0]
        base_env.close()
        agent = PPOAgent(obs_dim, act_dim, device)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        print(f"Loaded weights from {weights_path}")
        evaluate_and_record(agent)
    else:
        agent, episode_rewards = train()
        plot_rewards(episode_rewards)
        evaluate_and_record(agent)
