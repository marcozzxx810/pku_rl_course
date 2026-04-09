import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    """
    A simple Tabular Q-Learning Agent.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize the Q-table with zeros
        # Q-table shape: [number of states, number of actions]
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state, explore=True):
        """
        Choose an action based on the Epsilon-Greedy strategy.
        """
        if explore and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning update rule.
        Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a(Q(S',a)) - Q(S,A)]
        """
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (not done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


def train(env, agent, episodes=500):
    """
    Training loop.
    """
    rewards_per_episode = []
    q_values_per_episode = []
    avg_max_q_values_per_episode = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        
        q_values_per_episode.append(np.max(agent.q_table[state, :]))
        
        done = False
        truncated = False
        total_reward = 0
        step_max_q_values = []
        
        while not (done or truncated):
            step_max_q_values.append(np.max(agent.q_table[state, :]))
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
        rewards_per_episode.append(total_reward)
        avg_max_q_values_per_episode.append(np.mean(step_max_q_values) if step_max_q_values else 0)
        
        if (ep + 1) % (episodes/10) == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode: {ep+1}/{episodes}, Average Reward (last 50): {avg_reward:.2f}")
            
    return rewards_per_episode, q_values_per_episode, avg_max_q_values_per_episode


def test(env, agent, episodes=1):
    """
    Testing loop to evaluate the trained agent without exploration.
    """
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.choose_action(state, explore=False)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
        print(f"Test Episode {ep+1}/{episodes} | Total Reward: {total_reward}")


def plot_rewards(rewards, filename='q_learning_reward.png'):
    """
    Helper function to plot the training curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"Reward curve saved as: {filename}")

def plot_q_values(q_values, filename='q_learning_q_values.png'):
    """
    Helper function to plot the Q-value curve (Initial State Max Q-value).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(q_values, color='orange')
    plt.title('Initial State Max Q-value over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Max Q-value')
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"Q-value curve saved as: {filename}")


def plot_avg_max_q_values(avg_max_q_values, filename='q_learning_avg_max_q_values.png'):
    """
    Helper function to plot the Average Max Q-value curve per episode.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(avg_max_q_values, color='green')
    plt.title('Average Max Q-value per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Max Q-value')
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"Average Max Q-value curve saved as: {filename}")


if __name__ == '__main__':
    # 1. Initialize the environment (CliffWalking)
    env_name = 'CliffWalking-v0'
    env = gym.make(env_name)
    
    # 2. Get environment properties
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # 3. Instantiate the Agent
    agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # 4. Train the Agent
    print(f"Starting training on {env_name}...")
    training_rewards, training_q_values, training_avg_max_q = train(env, agent, episodes=1000)
    env.close()
    
    # 5. Plot Results
    plot_rewards(training_rewards)
    plot_q_values(training_q_values)
    plot_avg_max_q_values(training_avg_max_q)
    
    # 6. Visualize the trained policy
    print("Starting visualization of the trained policy...")
    env_render = gym.make(env_name, render_mode='human')
    test(env_render, agent, episodes=5)
    env_render.close()
