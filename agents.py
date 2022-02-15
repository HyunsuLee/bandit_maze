# agents.py

import numpy as np


class Q_agent():
    def __init__(self, state_size, action_size, alpha):
        self.state_size = state_size
        self.action_size = action_size
        self.q_learning = np.zeros([state_size, action_size])
        self.alpha = alpha  # learning rate for q learner
        # self.gamma = gamma # we don't need discount rate

    def reset(self):
        self.q_learning = np.zeros([self.state_size, self.action_size])

    def update_Q(self, current_exp):
        state = current_exp[0]
        action = current_exp[1]
        state_next = current_exp[2]
        reward = current_exp[3]
        done = current_exp[4]
        if done:
            td_error = (reward - self.q_learning[state, action])
        else:
            td_error = (reward + self.q_learning[state_next, :].max() -
                        self.q_learning[state, action])
        self.q_learning[state, action] += self.alpha * td_error
        return td_error

    @property
    def Q_vector_estimated(self):
        return self.q_learning
        # 이 경우 v_learning은 float을 반환한다. immutable한 객체이다.
        # 그러나 list같은 mutable한 객채를 반환할 경우, 그것은 같은 메모리를 가리킨다.
        # 그래서 deepcopy를 써야 한다.

    def Q_estimates(self, next_state, reward):
        '''
        estimate Q value depends on estimated value of next state
        '''
        V = self.v_learning[next_state]
        Qvalue = reward + self.gamma * V
        return Qvalue
