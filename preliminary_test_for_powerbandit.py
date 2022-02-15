# test.py
import numpy as np
import bandits as bd
import env as env
import agents as agents

#GB = bd.GaussianBandit(2)
# GB.reset()
# print(GB.pull(0))
# print(GB.pull(1))


#BB = bd.BinomialBandit(2,1)
# BB.reset()
# print(BB.action_values)
# print(BB.bin.distribution.p)
# print(BB.optimal)
# print(BB.pull(0))
# print(BB.pull(1))

PB = bd.PowerBandit(1)

n_trial = 1000
result = []
for trial in range(n_trial):
    result.append(PB.pull())
result = np.array(result)
returned = n_trial + result.sum()
print("when you have " + str(1000) +
      " budget, bandit return you " + str(returned))

bm = env.BanditMaze()
bm.reset()
state = bm.observation
action = 1
reward = bm.step(1)
state_next = bm.observation
done = bm.done


experiences = []
experiences.append([state, action, state_next, reward, done])

current_exp = experiences[-1]

agent = agents.Q_agent(bm.state_size, bm.action_size, alpha=0.01)

agent.update_Q(current_exp)
print(experiences)


for step_idx in range(10):
    state = bm.observation
    action = 1
    reward = bm.step(1)
    state_next = bm.observation
    done = bm.done

    experiences.append([state, action, state_next, reward, done])
    current_exp = experiences[-1]
    agent.update_Q(current_exp)
    print(agent.q_learning)

print(experiences)
