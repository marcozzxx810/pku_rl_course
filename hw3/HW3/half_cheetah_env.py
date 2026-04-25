import gymnasium as gym


# Set render_mode="human" to visualize the environment.
env = gym.make(
    "HalfCheetah-v5",
    render_mode="human",
    exclude_current_positions_from_observation=False,
)
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
