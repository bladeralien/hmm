# -*- coding:utf-8 -*-


import numpy as np


class MarkovChain(object):

    def __init__(self, states, initial, transition):
        self.states = states
        self.initial = initial
        self.transition = transition

    def state_at(self, t):
        if t < 0:
            raise ValueError("t must be >= 0")
        if t == 0:
            return self.initial
        else:
            return np.dot(np.linalg.matrix_power(self.transition, t), self.initial)


if __name__ == "__main__":
    markov_chain = MarkovChain([0, 1], [0.5, 0.5], [[0.8, 0.4], [0.2, 0.6]])
    for t in range(100):
        print(markov_chain.state_at(t))
