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
        # select action a′ in s′

        # update if s′ was reached with a from s yielding r :
        #   ^q(s, a) ← ^q(s, a) + α · (r + γ ^q(s′, a′) − ^q(s, a))

        # TODO: implement Q-value update
        td_target = r + gamma * self.q_values[s_prime][a_prime]
        td_error = td_target - self.q_values[s][a]
        self.q_values[s][a] += alpha * td_error

    def train(self, env):
        """"""
        print(f"Training {self.name}...")

        np.random.seed(42)
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
            is_last_episode = episode == self.n_episodes

            if self.epsilon_decay:
                epsilon = 0.0 if is_last_episode else self.epsilon / self.n_episodes
            else:
                epsilon = 0.0 if is_last_episode else self.epsilon

            #NOTE: Ignore epsilon decay, use epsilon-greedy for sampling (SARSA default)
            action = get_epsilon_greedy_action(env.np_random, s, self.q_values, epsilon)

            # TODO: SARSA algorithm
            # take steps in the environment until a terminal state is reached
            while not is_terminal:

                # TODO: SARSA algorithm (make sure to update is_terminal with the return from env.step(a))
                next_state, r, terminated, truncated, _ = env.step(action)
                is_terminal = terminated or truncated

                if not is_terminal:
                    next_action = get_epsilon_greedy_action(env.np_random, next_state, self.q_values, epsilon)

                    self.update_q_value(
                        s, action, next_state, next_action, r, self.alpha, self.gamma
                    )

                # TODO: update rewards for the current episode
                episode_reward += r

                s = next_state
                action = next_action if not is_terminal else None
                
            # TODO: nothing to do below this comment (leave code below unchanged)
            # store reward and render episode
            self.total_reward_per_episode[episode] = episode_reward
            self.render(env, episode, progress_bar, episode_reward)

        print(f"Training {self.name} done.")