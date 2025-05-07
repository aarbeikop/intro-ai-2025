import numpy as np
from tqdm import tqdm
from utils import get_epsilon_greedy_action
from rl_agent import RLAgent

class SARSA(RLAgent):
    def __init__(self, rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon, epsilon_decay=False) -> None:
        super().__init__(rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon)
        self.epsilon_decay = epsilon_decay
    

    def update_q_value(self, s, a, s_prime, a_prime, r, alpha, gamma):
        """
        Update the Q-values (inplace) in self.q_values.

        Parameters:
            s (int):        current state.
            a (int):        action to take in state s.
            s_prime (int):  next state, i.e., state reached from taking action a in state s.
            a_prime (int):  action to take in state s_prime.
            r (float):      reward from taking action a in state s.
            alpha (float):  learning rate
            gamma (float):  discount rate

        Returns:
            None
        """
        # TODO: implement Q-value update
        raise NotImplementedError


    def train(self, env):
        """"""
        print(f"Training {self.name}...")
        # training loop
        progress_bar = tqdm(np.arange(self.n_episodes+1))
        for episode in progress_bar:
            if episode == self.n_episodes:
                progress_bar.set_description(f'Simulating greedy policy')
            else:
                progress_bar.set_description(f'Episode: {episode+1}')

            # reset environment and get initial state s
            s, _ = env.reset()
            is_terminal = False
            episode_reward = 0

            # TODO: set epsilon for the current episode (hint: also condition on self.epsilon_decay)
            # in the last episode only, take only greedy actions

            # TODO: SARSA algorithm
            raise NotImplementedError

            # take steps in the environment until a terminal state is reached
            while not is_terminal:
                # TODO: SARSA algorithm (make sure to update is_terminal with the return from env.step(a))
                raise NotImplementedError

                # TODO: update rewards for the current episode
                # store some data
                episode_reward += r
            
            # TODO: nothing to do below this comment (leave code below unchanged)
            # store reward and render episode
            self.total_reward_per_episode[episode] = episode_reward
            self.render(env, episode, progress_bar, episode_reward)

        print(f"Training {self.name} done.")