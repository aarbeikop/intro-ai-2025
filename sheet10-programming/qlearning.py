import numpy as np
from tqdm import tqdm
from utils import get_epsilon_greedy_action, get_legal_actions
from rl_agent import RLAgent

class QLearning(RLAgent):
    def __init__(self, rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon) -> None:
        super().__init__(rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon)

    def update_q_value(self, s, a, s_prime, r, alpha, gamma):
        """
        Update the Q-values (inplace) in self.q_values.

        Parameters:
            s (int):        current state.
            a (int):        action to take in state s.
            s_prime (int):  next state, i.e., state reached from taking action a in state s.
            r (float):      reward from taking action a in state s.
            alpha (float):  learning rate
            gamma (float):  discount rate

        Returns:
            None
        """
        #   q(s, a) ← ˆq(s, a)+α·(r +γ max(a′) ˆq(s′, a′)−ˆq(s, a))
        #   (treats ˆq(s, ·) = 0 for terminal states s)

        legal_actions = get_legal_actions(s)

        max_q = np.max(self.q_values[s_prime][legal_actions])

        td_target = r + gamma * max_q
        td_error = td_target - self.q_values[s][a]
        self.q_values[s][a] += alpha * td_error


        # TODO: implement Q-value update
        #raise NotImplementedError


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

            # TODO: set epsilon for the current episode
            # in the last episode only, take only greedy actions
            greedy_only = episode == self.n_episodes
            epsilon = 0.0 if greedy_only else self.epsilon

            # TODO: Q-Learning algorithm
            
            # take steps in the environment until a terminal state is reached
            while not is_terminal:
                # TODO: Q-Learning algorithm (make sure to update is_terminal with the return from env.step(a))

                action = get_epsilon_greedy_action(env.np_random, s, self.q_values, epsilon)

                next_state, r, terminated, truncated, _ = env.step(action)
                is_terminal = terminated or truncated

                # TODO: Q-Learning algorithm
                self.update_q_value(s, action, next_state, r, self.alpha, self.gamma)

                s = next_state
                # TODO: update rewards for the current episode
                # store some data
                episode_reward += r
                #print(r)
                #print(f"Q: {self.q_values.shape}")
            #print(f"episode {episode}, reward: {episode_reward}")
            #print(f"Q: {self.q_values.shape}")
            # TODO: nothing to do below this comment (leave code below unchanged)
            # store reward and render episode
            self.total_reward_per_episode[episode] = episode_reward
            self.render(env, episode, progress_bar, episode_reward)

        print(f"Training {self.name} done.")