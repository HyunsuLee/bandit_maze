# env.py

import numpy as np
import bandits as bd


class BanditMaze():
    def __init__(self):
        self.state_size = 2

        # This is 1D maze with 3 cells starting at the mid state.
        self.action_size = 2
        self.ACTION_LT = 0
        self.ACTION_RT = 1
        self.action_set = [self.ACTION_LT, self.ACTION_RT]

        # enviroment properties
        self.out_pos = 0
        self.bandit_pos = 1
        self.start_pos = 1
        self.done = None
        self.agent_pos = None

    def reset(self):
        self.done = False
        self.agent_pos = self.start_pos

    @property
    def all_positions(self):
        '''
        return position index of goal and agent in list 
        '''
        all_positions = [self.out_pos] + [self.agent_pos]
        return all_positions

    def step(self, action):
        '''
        take action and move postion and return reward
        '''
        if self.agent_pos == self.start_pos:
            if action == self.ACTION_LT:
                self.done = True
                return -20
            elif action == self.ACTION_RT:
                PB = bd.PowerBandit(1)
                return PB.pull()
            else:
                raise IndexError("check action space")

    @property
    def observation(self):
        agent_pos_index = self.agent_pos
        return agent_pos_index
