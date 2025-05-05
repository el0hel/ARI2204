""" this is where we can test the blackjack game and RL algorithms implemented """

from env import BlackjackEnv


def main():
    env = BlackjackEnv()
    obs, game_over = env.reset()

    print(f"Initial state: {obs}")
    while not game_over:
        action = int(input("Enter action (0=HIT, 1=STAND): "))
        obs, reward, done = env.step(action)
        print(f"Next state: {obs}, Reward: {reward}, Game over: {game_over}")
        if done:
            print("Game over.")
            env.print_hands()


if __name__ == "__main__":
    main()
