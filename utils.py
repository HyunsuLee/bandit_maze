
import numpy as np


def my_argmax(Qarray):
    max_index = np.where(Qarray == Qarray.max())[0]
    if len(max_index) == 1:
        return np.argmax(Qarray)
    else:
        return np.random.randint(len(Qarray))


