from env import BlackjackEnv
import random
from collections import defaultdict
import math

random.seed(0)
"""
    Shared RL framework base class for Monte Carlo, SARSA, Q-Learning
    Implements:
      - Storage for state-action counts N(s,a) and value estimates Q(s,a)
      - Epsilon-greedy action selection (exploration policy)
      - Episode runner (with optional exploring starts)

    Subclasses must implement `update(trajectory)` to apply their specific learning rule.
    """


class BaseAgent:
    def __init__(self, num_of_actions=2, gamma=1.0):
        self.num_of_actions = num_of_actions
        self.gamma = gamma
        self.N = defaultdict(int)  # N(s,a): visit counts
        self.Q = defaultdict(float)  # Q(s,a): estimated action values
        self.episode = 0  # Counter for updates

    def select_action(self, state, epsilon):
        player_sum, _, _ = state
        if player_sum < 12:
            return 0  # HIT
        if player_sum == 21:
            return 1  # STAND

        # If the random probability is < epsilon, we explore and pick a random action
        if random.random() < epsilon:
            return random.randrange(self.num_of_actions)

        # Otherwise, we exploit and pick the action with the highest Q-value
        else:
            q_values = []
            for a in range(self.num_of_actions):
                q_values.append(self.Q[(state, a)])

            max_q = max(q_values)

            best_actions = []

            for idx, value in enumerate(q_values):
                if value == max_q:
                    best_actions.append(idx)

            # Making a random choice if their are ties
            return random.choice(best_actions)

    def episode_execution(self, env: BlackjackEnv, epsilon, first_exploration=False):
        # Resetting the enviornment
        state, _ = env.reset()
        done = False
        trajectory = []

        # Choosing the first action; in what manner is based upon whether we are first exploring or not
        if first_exploration:
            action = random.randrange(self.num_of_actions)
        else:
            action = self.select_action(state, epsilon)

        # The while loop will iterate until the round ends, where:
        #  first we apply the action, 
        #  then choose the next action,
        # And append to trajectory
        while not done:
            next_state, reward, done = env.step(action)

            if not done:
                next_action = self.select_action(next_state, epsilon)
            else:
                next_action = None

            trajectory.append((state, action, reward, next_state, next_action))
            state, action = next_state, next_action

        # Incrementing the episode counter
        self.episode = self.episode + 1

        return trajectory

    # The update function will be overridden by each subclass (algorithm) 
    def update(self, trajectory):
        raise NotImplementedError


############################################################################################################################
#                                                      The Algorithms                                                      #
class SARSAAgent(BaseAgent):
    def __init__(self, num_of_actions=2, gamma=1.0):
        super().__init__(num_of_actions, gamma)

    def update(self, trajectory):
        for state, action, reward, next_state, next_action in trajectory:
            # Incrementing visit counts
            self.N[(state, action)] += 1

            # The step size
            alpha = 1 / self.N[(state, action)]

            # The Current value
            q_current = self.Q[(state, action)]

            # The Next value
            if next_action is not None:
                q_next = self.Q[(next_state, next_action)]
            else:
                q_next = 0
            # The update rule
            self.Q[(state, action)] = q_current + alpha * (reward + self.gamma * q_next - q_current)


class MonteCarloAgent(BaseAgent):
    # iniitalise MC agent with its specific params
    def __init__(self, num_of_actions=2, gamma=1.0, exploring_starts=False, epsilon='1/k'):
        super().__init__(num_of_actions, gamma)
        self.exploring_starts = exploring_starts  # flag for whether to use Exploring Starts mechanism
        self.epsilon = epsilon
        self.state_action_counts = defaultdict(int)

    # returning episode value based on k (episode count)
    def get_epsilon(self):
        k = self.episode + 1  # episode count starting from 1

        # all epsilon decay strategies
        if self.epsilon == '1/k':
            return 1 / k
        elif self.epsilon == 'exp1000':
            return math.exp(-k / 1000)
        elif self.epsilon == 'exp10000':
            return math.exp(-k / 10000)

    # selecting action according to epsilon greedy policy
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_of_actions)
        else:
            q_values = [self.Q[(state, a)] for a in range(self.num_of_actions)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    # executing one episode
    def episode_execution(self, env: BlackjackEnv, epsilon=None, first_exploration=False):
        # resetting environment
        state, _ = env.reset()
        done = False
        trajectory = []

        # getting epsilon for the current episode
        epsilon = self.get_epsilon()

        # only using epsilon-greedy in [12 ... 20], and using purely greedy (ε = 0) elsewhere
        # handling the first action of the episode for Exploring Starts
        if self.exploring_starts and 12 <= state[0] <= 20:
            # always random action in the first state if Exploring Starts is enabled and player sum is bet. 12 and 20
            action = random.randrange(self.num_of_actions)
        else:
            # normal epsilon-greedy action selection otherwise
            action = self.select_action(state, epsilon)

        # handling the rest of the episode
        while not done:
            # stepping the environment w selected action
            next_state, reward, done = env.step(action)

            # if player sum is bet. 12 and 20, use epsilon-greedy
            if not done:
                if 12 <= next_state[0] <= 20:
                    next_action = self.select_action(next_state, epsilon)
                else:
                    next_action = self.select_action(next_state, 0.0)  # setting epsilon to 0 for purely greedy
                    # actions if player sum is not bet. 12 and 20
            else:
                next_action = None

            # append the state, action ,reward, next state and next action to trajectory
            trajectory.append((state, action, reward, next_state, next_action))
            state, action = next_state, next_action  # move to next state and action
            self.state_action_counts[(state, action)] += 1

        self.episode += 1
        return trajectory

    # updating q-values using monte carlo update rule
    def update(self, trajectory):
        G_t = 0  # initialising return

        # traversing trajectory in reverse order, to calculate returns
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward, next_state, next_action = trajectory[t]
            G_t = reward + self.gamma * G_t

            # incrementing visit count  for (s, a)
            self.N[(state, action)] += 1

            # computing step size
            alpha = 1 / self.N[(state, action)]

            # getting current Q-value for (s, a)
            q_current = self.Q[(state, action)]

            # Monte Carlo update rule
            self.Q[(state, action)] = q_current + alpha * (G_t - q_current)


class QLearningAgent(BaseAgent):
    def __init__(self, num_of_actions=2, gamma=1.0):
        super().__init__(num_of_actions, gamma)

    def update(self, trajectory):
        for state, action, reward, next_state, _ in trajectory:
            # incrementing visit count  for (s, a)
            self.N[(state, action)] += 1

            # Computing step size as given
            alpha = 1 / (self.N[(state, action)] + 1)

            # get the maximum Q-value of the best possible actions in the next state
            max_q_next = max(self.Q[(next_state, a)] for a in range(self.num_of_actions))

            # Bellman optimality equation
            td_target = reward + self.gamma * max_q_next

            # getting the difference between the target q and the current value of q 
            td_error = td_target - self.Q[(state, action)]

            # updating current q value using the weights calculated
            self.Q[(state, action)] += alpha * td_error


class DoubleQLearningAgent(BaseAgent):
    def __init__(self, num_of_actions=2, gamma=1.0):
        super().__init__(num_of_actions, gamma)
        # the two state-action value functions
        self.Q1 = defaultdict(float)
        self.Q2 = defaultdict(float)

    def update(self, trajectory):
        for state, action, reward, next_state, _ in trajectory:
            # update the count for (state, action)
            self.N[(state, action)] += 1
            alpha = 1 / (self.N[(state, action)] + 1)  # step size = 1 / (N(s, a) + 1)

            # Randomly decide whether to update Q1 or Q2
            if random.random() < 0.5:
                # same steps as the Q-Learning agent
                max_a = max(range(self.num_of_actions), key=lambda a: self.Q1[(next_state, a)])
                q_next = self.Q2[(next_state, max_a)]
                td_target = reward + self.gamma * q_next
                td_error = td_target - self.Q1[(state, action)]
                self.Q1[(state, action)] += alpha * td_error
            else:
                # same steps as the Q-Learning agent
                max_a = max(range(self.num_of_actions), key=lambda a: self.Q2[(next_state, a)])
                q_next = self.Q1[(next_state, max_a)]
                td_target = reward + self.gamma * q_next
                td_error = td_target - self.Q2[(state, action)]
                self.Q2[(state, action)] += alpha * td_error

    def select_action(self, state, epsilon):
        player_sum, _, _ = state
        if player_sum < 12:
            return 0  # HIT
        if player_sum == 21:
            return 1  # STAND

        if random.random() < epsilon:
            return random.randint(0, self.num_of_actions - 1)
        else:
            # Use average Q-value for decision
            q_vals = [self.Q1[(state, a)] + self.Q2[(state, a)] for a in range(self.num_of_actions)]
            max_q = max(q_vals)
            best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
            return random.choice(best_actions)
