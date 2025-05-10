import matplotlib.pyplot as plt
from rl_framework import MonteCarloAgent
from env import BlackjackEnv
import random

random.seed(0)


def execute_MC(agent, num_episodes=100000):
    env = BlackjackEnv()
    win_counts = []
    loss_counts = []
    draw_counts = []

    outcome_counter = {'win': 0, 'loss': 0, 'draw': 0}

    for episode in range(1, num_episodes + 1):
        trajectory = agent.episode_execution(env)

        # reward at the end of the episode
        final_reward = trajectory[-1][2]  # reward at terminal transition

        if final_reward == 1:
            outcome_counter['win'] += 1
        elif final_reward == -1:
            outcome_counter['loss'] += 1
        else:
            outcome_counter['draw'] += 1

        agent.update(trajectory)

        # every 1000 episodes, log win/loss/draw
        if episode % 1000 == 0:
            win_counts.append(outcome_counter['win'])
            loss_counts.append(outcome_counter['loss'])
            draw_counts.append(outcome_counter['draw'])
            outcome_counter = {'win': 0, 'loss': 0, 'draw': 0}

    # unique (state, action) pairs
    unique_pairs = set(agent.Q.keys())

    # selection counts
    pair_selection_counts = dict(agent.N)

    # estimated Q-values
    q_values = dict(agent.Q)

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


# run configs
configs = {
    "ES (epsilon = 1/k)": MonteCarloAgent(exploring_starts=True, epsilon='1/k'),
    "No ES, epsilon = 1/k": MonteCarloAgent(exploring_starts=False, epsilon='1/k'),
    "No ES, epsilon = exp(-k/1000)": MonteCarloAgent(exploring_starts=False, epsilon='exp1000'),
    "No ES, epsilon = exp(-k/10000)": MonteCarloAgent(exploring_starts=False, epsilon='exp10000')
}

results = {}

for name, agent in configs.items():
    print(f"Running configuration: {name}")
    results[name] = execute_MC(agent)

plot_episode_outcomes(results)
