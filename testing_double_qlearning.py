import matplotlib.pyplot as plt
from rl_framework import DoubleQLearningAgent
import pandas as pd
import math
import random

random.seed(0)

# Strategy table builder for Double Q-Learning
def build_strategy_table(q_values, use_ace=True):
    table = []
    for player_sum in range(20, 11, -1):
        row = []
        for dealer_card in range(2, 12):
            state = (player_sum, dealer_card, use_ace)
            hit_value = q_values.get((state, 0), float('-inf'))
            stand_value = q_values.get((state, 1), float('-inf'))
            best_action = 'H' if hit_value > stand_value else 'S'
            row.append(best_action)
        table.append(row)
    return pd.DataFrame(table, index=range(20, 11, -1), columns=range(2, 12))

def execute_double_qlearning(agent, epsilon_schedule, num_episodes=100000):
    from env import BlackjackEnv
    env = BlackjackEnv()
    win_counts = []
    loss_counts = []
    draw_counts = []
    outcome_counter = {'win': 0, 'loss': 0, 'draw': 0}

    for episode in range(1, num_episodes + 1):
        epsilon = epsilon_schedule(episode)
        trajectory = agent.episode_execution(env, epsilon, first_exploration=False)

        final_reward = trajectory[-1][2]

        if final_reward == 1:
            outcome_counter['win'] += 1
        elif final_reward == -1:
            outcome_counter['loss'] += 1
        else:
            outcome_counter['draw'] += 1

        agent.update(trajectory)

        if episode % 1000 == 0:
            win_counts.append(outcome_counter['win'])
            loss_counts.append(outcome_counter['loss'])
            draw_counts.append(outcome_counter['draw'])
            outcome_counter = {'win': 0, 'loss': 0, 'draw': 0}

    # Compute final Q-values and dealer advantage
    q_values = {key: (agent.Q1[key] + agent.Q2[key]) / 2 for key in set(agent.Q1) | set(agent.Q2)}
    unique_pairs = set(q_values.keys())
    pair_selection_counts = dict(agent.N)
    final_10k_wins = sum(win_counts[-10:])
    final_10k_losses = sum(loss_counts[-10:])
    dealer_advantage = (
        (final_10k_losses - final_10k_wins) / (final_10k_losses + final_10k_wins)
        if (final_10k_losses + final_10k_wins) != 0 else 0
    )

    return {
        'wins': win_counts,
        'losses': loss_counts,
        'draws': draw_counts,
        'unique_pairs': unique_pairs,
        'pair_counts': pair_selection_counts,
        'q_values': q_values,
        'dealer_advantage': dealer_advantage
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

def plot_eval_metrics_and_strategy(results):
    configs = list(results.keys())
    unique_counts = [len(results[c]['unique_pairs']) for c in configs]
    plt.figure(figsize=(10, 5))
    plt.bar(configs, unique_counts)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Unique (s,a) Pairs")
    plt.title("Unique State-Action Pairs per Configuration")
    plt.tight_layout()
    plt.show()

    for config, data in results.items():
        top_items = sorted(data['pair_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [str(k) for k, _ in top_items]
        values = [v for _, v in top_items]
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.xticks(rotation=90)
        plt.ylabel("Frequency")
        plt.title(f"Top 10 Most Frequent (s,a) Selections - {config}")
        plt.tight_layout()
        plt.show()

    advantages = [results[c]['dealer_advantage'] for c in configs]
    best_index = advantages.index(min(advantages))
    bar_colors = ['orange' if i != best_index else 'green' for i in range(len(configs))]

    plt.figure(figsize=(10, 5))
    plt.bar(configs, advantages, color=bar_colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Dealer Advantage")
    plt.title("Dealer Advantage per Configuration (Lowest = Best)")
    plt.tight_layout()
    plt.show()

    for config, data in results.items():
        q_values = data['q_values']
        print(f"\nStrategy Table for {config} (Usable Ace = True):")
        print(build_strategy_table(q_values, use_ace=True).to_string())
        print(f"\nStrategy Table for {config} (Usable Ace = False):")
        print(build_strategy_table(q_values, use_ace=False).to_string())

# Run configurations
epsilon_schedules = {
    "Double Q-Learning | epsilon = 0.1": lambda k: 0.1,
    "Double Q-Learning | epsilon = 1/k": lambda k: 1.0 / k,
    "Double Q-Learning | epsilon = exp(-k/1000)": lambda k: math.exp(-k / 1000),
    "Double Q-Learning | epsilon = exp(-k/10000)": lambda k: math.exp(-k / 10000),
}

results = {}

for name, eps_fn in epsilon_schedules.items():
    print(f"Running configuration: {name}")
    agent = DoubleQLearningAgent()
    results[name] = execute_double_qlearning(agent, eps_fn)

plot_episode_outcomes(results)
plot_eval_metrics_and_strategy(results)
