import math
from env import BlackjackEnv
from rl_framework import SARSAAgent


def execute_sarsa(configuration, episode_fn, num_episodes=100_000, evaluation_episodes=10_000):
    print(f"\n===== SARSA On-Policy Control: {configuration} =====\n")

    env = BlackjackEnv()
    agent = SARSAAgent()


    # Training with the configured epsilon
    total_return = 0.0

    for episode in range(1, num_episodes + 1):
        epsilon = episode_fn(episode)

        trj = agent.episode_execution(env, epsilon, first_exploration=False)

        agent.update(trj)
        total_return += trj[-1][2]

    avg_return = total_return / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_return:.3f}")



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



    # Displaying the Learned Policy
    print("\nLearned policy (0=HIT, 1=STAND):")
    for player_sum in range(12, 21):
        row = []
        for dealer_card in range(2, 12):
            for usable_ace in [False, True]:
                state = (player_sum, dealer_card, usable_ace)
                # Setting epsilon to 0 by default to interpret the learned policy using purely greedy decisions (No Exploration)
                action = agent.select_action(state, epsilon=0.0)

                row.append(str(action))
        print(f"Sum={player_sum}: " + " ".join(row))



def main():
    configurations = {
        "constant ε=0.1": lambda k: 0.1,
        "ε = 1/√k":      lambda k: 1.0 / math.sqrt(k), # Assuming that 1/k' means 1/(square root(k))
        "ε = exp(-k/1000)":  lambda k: math.exp(-k/1000),
        "ε = exp(-k/10000)": lambda k: math.exp(-k/10000),
    }

    for name, func in configurations.items():
        execute_sarsa(name, func)

if __name__ == "__main__":
    main()

