import gymnasium as gym
import matplotlib.pyplot as plt


def run_random_baseline(env_name="HalfCheetah-v5", episodes=5, seed=42):
    env = gym.make(env_name, exclude_current_positions_from_observation=False)
    returns = []

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)
        print(f"Episode {episode + 1:03d} | Return {total_reward:.2f}")

    env.close()
    return returns


def plot_returns(returns, filename="random_baseline_reward.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(returns, marker="o")
    plt.title("Random Baseline on HalfCheetah-v5")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved plot to {filename}")


if __name__ == "__main__":
    rewards = run_random_baseline()
    plot_returns(rewards)
