import numpy as np
from tqdm import tqdm
import copy

from lib.env import BanditMaze
from lib.agents import Q_agent
from lib.utils import my_argmax


def QL(episodes=50, alpha=0.1):
    experiences = []

    Q_history = []
    step_lengths = []
    cumulative_reward_list = []

    bm = BanditMaze()
    agent = Q_agent(bm.state_size, bm.action_size, alpha=alpha)
    agent.reset()
    for episode in tqdm(range(episodes), desc="episodes"):
        bm.reset()
        state = bm.observation

        # epsilon decay
        epsilon = 0.9 * (0.9 ** episode) + 0.1

        cumulative_reward = 0
        step_idx = 0
        while True:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(bm.action_size)
            else:
                action = my_argmax(agent.q_learning[state, :])

            reward = bm.step(action)
            cumulative_reward += reward
            state_next = bm.observation
            done = bm.done
            experiences.append([state, action, state_next, reward, done])
            state = state_next
            agent.update_Q(experiences[-1])
            if bm.done or cumulative_reward < -100:
                break
            step_idx += 1

        step_lengths.append(step_idx)
        cumulative_reward_list.append(cumulative_reward)
        copyed = copy.deepcopy(agent.Q_vector_estimated)
        Q_history.append(copyed)

    return step_lengths, cumulative_reward_list, Q_history
