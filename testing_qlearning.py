import matplotlib.pyplot as plt
from rl_framework import QLearningAgent
from env import BlackjackEnv
import math
import random

random.seed(0)

def execute_qlearning(agent, epsilon_schedule, num_episodes=100000):
    env = BlackjackEnv()
    win_counts = []
    loss_counts = []
    draw_counts = []

    outcome_counter = {'win': 0, 'loss': 0, 'draw': 0}

    for episode in range(1, num_episodes + 1):
        epsilon = epsilon_schedule(episode)
        trajectory = agent.episode_execution(env, epsilon, first_exploration=False)

        # reward at the end of the episode
        final_reward = trajectory[-1][2]
        if final_reward == 1:
            outcome_counter['win'] += 1
        elif final_reward == -1:
            outcome_counter['loss'] += 1
        else:
            outcome_counter['draw'] += 1

        agent.update(trajectory)

        # every 1000 eppisodes, log the counters and restart them
        if episode % 1000 == 0:
            win_counts.append(outcome_counter['win'])
            loss_counts.append(outcome_counter['loss'])
            draw_counts.append(outcome_counter['draw'])
            outcome_counter = {'win': 0, 'loss': 0, 'draw': 0}

    
    unique_pairs = set(agent.Q.keys()) # unique (state, action) pairs
    pair_selection_counts = dict(agent.N) # selection counts
    q_values = dict(agent.Q) # estimated Q-values

    return {
        'wins': win_counts,
        'losses': loss_counts,
        'draws': draw_counts,
        'unique_pairs': unique_pairs,
        'pair_counts': pair_selection_counts,
        'q_values': q_values
    }


def plot_episode_outcomes(results):
    for config_name, data in results.items():
        episodes = list(range(1000, 100001, 1000))
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, data['wins'], label='Wins')
        plt.plot(episodes, data['losses'], label='Losses')
        plt.plot(episodes, data['draws'], label='Draws')
        plt.xlabel('Episodes')
        plt.ylabel('Count (per 1000 episodes)')
        plt.title(f'Episode Outcomes for {config_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# run configurations
epsilon_schedules = {
    "Q-Learning (SARSAMAX) | epsilon = 0.1": lambda k: 0.1,
    "Q-Learning (SARSAMAX) | epsilon = 1/k": lambda k: 1.0 / k,
    "Q-Learning (SARSAMAX) | epsilon = exp(-k/1000)": lambda k: math.exp(-k / 1000),
    "Q-Learning (SARSAMAX) | epsilon = exp(-k/10000)": lambda k: math.exp(-k / 10000),
}

results = {}

for name, eps_fn in epsilon_schedules.items():
    print(f"Running configuration: {name}")
    agent = QLearningAgent()
    results[name] = execute_qlearning(agent, eps_fn)

plot_episode_outcomes(results)