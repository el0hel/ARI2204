from env import BlackjackEnv
import random
from collections import defaultdict

"""
    Shared RL framework base class for Monte Carlo, SARSA, Q-Learning
    Implements:
      - Storage for state-action counts N(s,a) and value estimates Q(s,a)
      - Epsilon-greedy action selection (exploration policy)
      - Episode runner (with optional exploring starts)

    Subclasses must implement `update(trajectory)` to apply their specific learning rule.
    """

class BaseAgent:
    def __init__(self,num_of_actions = 2, gamma = 1.0):
        self.num_of_actions = num_of_actions
        self.gamma = gamma
        self.N = defaultdict(int)    # N(s,a): visit counts
        self.Q = defaultdict(float)  # Q(s,a): estimated action values
        self.episode = 0             # Counter for updates

    
    def select_action(self, state, epsilon):
        player_sum, _, _ = state
        if player_sum < 12:
            return 0   # HIT
        if player_sum == 21:
            return 1   # STAND
        
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
    

    def episode_execution(self, env: BlackjackEnv, epsilon, first_exploration = False):
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
        self.episode = self.episode+1
        
        return trajectory

    # The update function will be overridden by each subclass (algorithm) 
    def update(self, trajectory):
        raise NotImplementedError


############################################################################################################################
#                                                      The Algorithms                                                      #
class SARSAAgent(BaseAgent):
    def __init__(self,num_of_actions = 2, gamma = 1.0):
        super().__init__(num_of_actions, gamma)
    
    def update(self, trajectory):
        for state, action, reward, next_state, next_action in trajectory:
            # Incrementing visit counts
            self.N[(state, action)] +=1
            
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

