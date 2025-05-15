import math
import matplotlib.pyplot as plt
from env import BlackjackEnv
from rl_framework import SARSAAgent


def execute_sarsa(configuration, episode_fn, num_episodes=100_000, evaluation_episodes=10_000):
    print(f"\n===== SARSA On-Policy Control: {configuration} =====\n")

    env = BlackjackEnv()
    agent = SARSAAgent()

    win_blocks  = []
    loss_blocks = []
    draw_blocks = []
    w = 0
    l = 0
    d = 0

    last_10k_wins  = 0
    last_10k_losses = 0
    last_10k_draws = 0

    # Training with the configured epsilon
    total_return = 0.0

    for episode in range(1, num_episodes + 1):
        epsilon = episode_fn(episode)

        trj = agent.episode_execution(env, epsilon, first_exploration=False)
        agent.update(trj)

        rslt = trj[-1][2]
        
        # Count wins, losses, and draws
        if rslt > 0:
            w += 1
        elif rslt < 0:
            l += 1
        else:
            d += 1

        # Tally last 10k
        if episode > num_episodes - 10000:
            if   rslt > 0: 
                last_10k_wins += 1
            elif rslt < 0: 
                last_10k_losses += 1
            else:       
                last_10k_draws += 1

        if episode % 1000 == 0:
            win_blocks.append(w)
            loss_blocks.append(l)
            draw_blocks.append(d)
            w = 0
            l = 0
            d = 0


    # Plotting the win/loss/draw trend
    episodes = [i * 1000 for i in range(1, len(win_blocks) + 1)]
    plt.figure()
    plt.plot(episodes, win_blocks,  label='Wins',   color='green')
    plt.plot(episodes, loss_blocks, label='Losses', color='red')
    plt.plot(episodes, draw_blocks, label='Draws',  color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Count per 1,000 episodes')
    plt.title(f'Win/Loss/Draw Trend: {configuration}')
    plt.legend()
    plt.show()


    # Print the results in blocks of 1000 episodes
    print("Episodes   Wins   Losses   Draws")
    #for i, (wb, lb, db) in enumerate(zip(win_blocks, loss_blocks, draw_blocks), start=1):
    #    print(f"{i*1000:7d}   {wb:4d}   {lb:7d}   {db:5d}")
    print()


    # Print the number of unique (s,a) pairs visited
    total_possible = 360
    unique_sa_pairs = sum(1 for ((ps, dc, ua), a), cnt in agent.N.items() if cnt > 0 and 12 <= ps <= 20 and 2 <= dc <= 11)
    print(f"Unique (s,a) visited: {unique_sa_pairs} / {total_possible}\n")


    # Print the state-action counts for the visited pairs
    sa_pair_counts = [
        (state, action, cnt)
        for (state, action), cnt in agent.N.items()
        if cnt > 0 and 12 <= state[0] <= 20 and 2 <= state[1] <= 11]
    
    sa_pair_counts.sort(key=lambda x: x[2], reverse=True)

    top50 = sa_pair_counts[:50]
    labels = [
        f"{'Hit' if a == 0 else 'Stand'}-{ps},{dc},{'YES' if ua else 'NO'}"
        for (ps, dc, ua), a, _ in top50]
    
    counts = [cnt for _, _, cnt in top50]

    # Plotting the top 50 state-action pairs
    plt.figure(figsize=(20, 6))
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), labels, rotation=90)
    plt.xlabel('State-Action pair (HIT/STAND - player_sum,dealer,usable_ace)')
    plt.ylabel('Visit Count')
    plt.title(f'Top 50 State-Action Visits: {configuration}')
    plt.tight_layout()
    plt.show()

    print("State-action counts (sorted by frequency):")
    #for state, action, cnt in sa_pair_counts:
    #   print(f"State={state}, action={action}: {cnt}")
    print()


    # Print the estimated Q-values for each unique (s,a) pair
    Q_list = [
        (state, action, agent.Q[(state, action)])
        for (state, action), cnt in agent.N.items()
        if cnt > 0 and 12 <= state[0] <= 20 and 2 <= state[1] <= 11]
    
    Q_list.sort(key=lambda x: x[2], reverse=True)
    print("Estimated Q-values (sorted by value):")
    #for (ps, dc, ua), action, q_val in Q_list:
    #    act_str = "HIT" if action == 0 else "STAND"
    #    print(f"  State=(sum={ps}, dealer={dc}, usable_ace={ua}), action={act_str} → Q={q_val:.3f}")
    print()


    # Evaluating the Epsilon Greedy policy
    wins = 0
    draws = 0
    losses = 0

    for epi in range(evaluation_episodes):
        trj = agent.episode_execution(env, epsilon=0.0, first_exploration=False)
        result = trj[-1][2]

        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

    print(f"Epsilon Greedy evaluation over {evaluation_episodes} episodes ---> wins: {wins}, losses: {losses}, draws: {draws}")
    

    print()
    # Blackjack strategy tables
    cell_width = 3
    for usable_ace in [False, True]:
        title = "Ace not counted as 11" if not usable_ace else "Ace counted as 11"
        print(f"--- {title} ---")

        # Header row: dealer upcards 2…10, A
        header_values = list(range(2, 11)) + [1]
        header_labels = [str(v) if v != 1 else 'A' for v in header_values]
        header_line = "   " + "".join(f"{h:>{cell_width}}" for h in header_labels)
        print(header_line)

        for ps in range(20, 11, -1):
            row_actions = []
            for dc in header_values:
                state   = (ps, dc, usable_ace)         # now dc==1 for Ace
                Q_hit   = agent.Q.get((state, 0), 0.0)
                Q_stand = agent.Q.get((state, 1), 0.0)
                cell    = 'H' if Q_hit > Q_stand else 'S'
                row_actions.append(cell)

            row_line = "".join(f"{act:>{cell_width}}" for act in row_actions)
            print(f"{ps:>2} " + row_line)

        print()

    # Calculate the dealer advantage based on the last 10k episodes
    adv = (last_10k_losses - last_10k_wins) / (last_10k_losses + last_10k_wins)
    print(f"Last 10k episodes — wins: {last_10k_wins}, losses: {last_10k_losses}, draws: {last_10k_draws}")
    print(f"Dealer advantage: {adv:.3f}\n")

    return unique_sa_pairs, adv


   



def main():
    configurations = {
        "constant ε=0.1": lambda k: 0.1,
        "ε = 1/k": lambda k: 1.0 / k,
        "ε = exp(-k/1000)":  lambda k: math.exp(-k/1000),
        "ε = exp(-k/10000)": lambda k: math.exp(-k/10000),
    }

    unique_counts = {}
    advantages    = {}

    for name, func in configurations.items():
        uc, adv = execute_sarsa(name, func)
        unique_counts[name] = uc
        advantages[name] = adv

    names  = list(unique_counts.keys())
    counts = list(unique_counts.values())

    # Plotting the unique state-action pairs visited for each configuration
    plt.figure()
    plt.bar(names, counts, color=['green', 'orange', 'blue', 'purple'])
    plt.xlabel('ε Configuration')
    plt.ylabel('Unique state-action pairs visited')
    plt.title('Exploration Coverage Across ε Configurations')
    plt.ylim(290, 341)
    plt.yticks(range(290, 340, 10))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plotting the dealer advantage for each configuration
    advs = [advantages[n] for n in names]
    plt.figure()
    plt.bar(names, advs, color=['red', 'brown', 'pink', 'black'])
    plt.xlabel('ε Configuration')
    plt.ylabel('Dealer advantage')
    plt.title('Dealer Advantage Across SARSA ε Configurations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

