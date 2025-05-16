import matplotlib.pyplot as plt
from rl_framework import MonteCarloAgent
from env import BlackjackEnv
import random

import matplotlib.pyplot as plt

# training for one agent configuration + tracking metrics
def execute_mc(agent, configuration, num_episodes=100_000):

    env = BlackjackEnv()
    agent = MonteCarloAgent()
    win_blocks = []
    loss_blocks = []
    draw_blocks = []
    w = l = d = 0

    last_10k_wins = last_10k_losses = last_10k_draws = 0

    for episode in range(1, num_episodes + 1):
        trj = agent.episode_execution(env)
        agent.update(trj)
        rslt = trj[-1][2]  # final reward

        if rslt > 0:
            w += 1
        elif rslt < 0:
            l += 1
        else:
            d += 1

        if episode > num_episodes - 10000:
            if rslt > 0:
                last_10k_wins += 1
            elif rslt < 0:
                last_10k_losses += 1
            else:
                last_10k_draws += 1

        # storing result blocks and resetting counters every 1k episodes
        if episode % 1000 == 0:
            win_blocks.append(w)
            loss_blocks.append(l)
            draw_blocks.append(d)
            w = l = d = 0

    # plotting episode outcomes
    episodes = [i * 1000 for i in range(1, len(win_blocks) + 1)]
    plt.figure()
    plt.plot(episodes, win_blocks, label='Wins', color='green')
    plt.plot(episodes, loss_blocks, label='Losses', color='red')
    plt.plot(episodes, draw_blocks, label='Draws', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Count per 1,000 episodes')
    plt.title(f'Win/Loss/Draw Trend: {configuration}')
    plt.legend()
    plt.show()

    # unique (s,a) counts
    total_possible = 360
    unique_sa_pairs = sum(
        1 for ((ps, dc, ua), a), cnt in agent.N.items() if cnt > 0 and 12 <= ps <= 20 and 2 <= dc <= 11)
    print(f"Unique (s,a) visited: {unique_sa_pairs} / {total_possible}\n")

    # state-action counts
    sa_pair_counts = [((state, action), cnt) for (state, action), cnt in agent.N.items() if
                      cnt > 0 and 12 <= state[0] <= 20 and 2 <= state[1] <= 11]
    sa_pair_counts.sort(key=lambda x: x[1], reverse=True)
    top50 = sa_pair_counts[:50]
    labels = [f"{'Hit' if a == 0 else 'Stand'}-{ps},{dc},{'YES' if ua else 'NO'}" for ((ps, dc, ua), a), _ in top50]
    counts = [cnt for _, cnt in top50]

    # Plot top-50 bar chart
    plt.figure(figsize=(20, 6))
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), labels, rotation=90)
    plt.xlabel('State-Action pair (HIT/STAND - player_sum,dealer,usable_ace)')
    plt.ylabel('Visit Count')
    plt.title(f'Top 50 State-Action Visits: {configuration}')
    plt.tight_layout()
    plt.show()

    # strategy tables for policy visualisation
    cell_width = 3
    for usable_ace in [False, True]:
        title = "Ace not counted as 11" if not usable_ace else "Ace counted as 11"
        print(f"--- {title} ---")
        header_values = list(range(2, 11)) + [1]
        header_labels = [str(v) if v != 1 else 'A' for v in header_values]
        header_line = "   " + "".join(f"{h:>{cell_width}}" for h in header_labels)
        print(header_line)
        for ps in range(20, 11, -1):
            row_actions = []
            for dc in header_values:
                state = (ps, dc, usable_ace)
                Q_hit = agent.Q.get((state, 0), 0.0)
                Q_stand = agent.Q.get((state, 1), 0.0)
                cell = 'H' if Q_hit > Q_stand else 'S'
                row_actions.append(cell)
            row_line = "".join(f"{act:>{cell_width}}" for act in row_actions)
            print(f"{ps:>2} " + row_line)
        print()

    # calculating dealer advantage
    adv = (last_10k_losses - last_10k_wins) / (last_10k_losses + last_10k_wins)
    print(f"Last 10k episodes — wins: {last_10k_wins}, losses: {last_10k_losses}, draws: {last_10k_draws}")
    print(f"Dealer advantage: {adv:.3f}\n")

    return unique_sa_pairs, adv


def main():

    # defining 4 MC variants
    configs = {
        "ES (ε = 1/k)": lambda: MonteCarloAgent(exploring_starts=True, epsilon='1/k'),
        "No ES, ε = 1/k": lambda: MonteCarloAgent(exploring_starts=False, epsilon='1/k'),
        "No ES, ε = exp(-k/1000)": lambda: MonteCarloAgent(exploring_starts=False, epsilon='exp1000'),
        "No ES, ε = exp(-k/10000)": lambda: MonteCarloAgent(exploring_starts=False, epsilon='exp10000'),
    }

    unique_counts = {}
    advantages = {}

    # running all experiments
    for name, build_agent in configs.items():
        print(f"Running configuration: {name}")
        agent = build_agent()
        unique_sa_count, adv = execute_mc(agent, name)
        unique_counts[name] = unique_sa_count
        advantages[name] = adv

    # plotting unique state-action coverage
    plt.figure()
    plt.bar(unique_counts.keys(), unique_counts.values(), color=['green', 'orange', 'blue', 'purple'])
    plt.xlabel('ε Configuration')
    plt.ylabel('Unique state-action pairs visited')
    plt.title('Exploration Coverage Across ε Configurations')
    plt.ylim(290, 315)
    plt.yticks(range(290, 320, 10))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot dealer advantage
    plt.figure()
    plt.bar(advantages.keys(), advantages.values(), color=['red', 'brown', 'pink', 'black'])
    plt.xlabel('Configuration')
    plt.ylabel('Dealer advantage')
    plt.title('Dealer Advantage Across MC ε Configurations')
    plt.ylim(0.06, 0.14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
