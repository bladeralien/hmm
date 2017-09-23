# -*- coding:utf-8 -*-


import numpy as np


class HiddenMarkovModel(object):

    # TODO Scaling

    def __init__(self, states, initial, transition, observations, emission):
        self.states = states
        self.initial = initial
        self.transition = transition
        self.observations = observations
        self.emission = emission

    def state_at(self, t):
        if t < 0:
            raise ValueError("t must be >= 0")
        if t == 0:
            return self.initial
        else:
            return np.dot(np.linalg.matrix_power(self.transition, t), self.initial)

    def status_at(self, t):
        if t < 0:
            raise ValueError("t must be >= 0")
        if t == 0:
            return np.dot(self.emission, self.initial)
        else:
            return np.dot(self.emission, np.dot(np.linalg.matrix_power(self.transition, t), self.initial))

    # The Forward Calculation
    # Pr(X(t...1)^s(t)=i|H) or Pr(X(t...1)|H)
    # TODO vectorize
    def alpha(self, observations, t, state=None):
        T = len(observations)
        if t < 0 or t >= T:
            raise ValueError("t must be >= 0 and < T")
        if state is not None:
            if t == 0:
                return self.emission[observations[t]][state] * self.initial[state]
            else:
                # TODO rename
                temp = 0
                for j in self.states:
                    temp += self.transition[state][j] * self.alpha(observations, t - 1, j)
                return self.emission[observations[t]][state] * temp
        else:
            # TODO rename
            temp = 0
            for i in self.states:
                temp += self.alpha(observations, t, i)
            return temp

    # The Backward Calculation
    # Pr(X(t+1...T)|H^s(t)=i)
    # TODO vectorize
    def beta(self, observations, t, state):
        T = len(observations)
        if t < 0 or t >= T:
            raise ValueError("t must be >= 0 and < T")
        if t == T - 1:
            return 1
        else:
            # TODO rename
            temp = 0
            for i in self.states:
                temp += self.emission[observations[t + 1]][i] * self.transition[i][state] * self.beta(observations, t + 1, i)
            return temp

    # Pr(s(t)=i|H^X)
    # TODO vectorize
    def gamma(self, observations, t, state):
        T = len(observations)
        if t < 0 or t >= T:
            raise ValueError("t must be >= 0 and < T")
        temp = 0
        for i in self.states:
            temp += self.alpha(observations, t, i) * self.beta(observations, t, i)
        return self.alpha(observations, t, state) * self.beta(observations, t, state) / temp


    # δ(t, i) is the highest probability of being in state i at time t, over
    # all state sequences that account for the first t observed symbols.
    def delta(self, observations, t, state):
        T = len(observations)
        if t < 0 or t >= T:
            raise ValueError("t must be >= 0 and < T")
        if t == 0:
            return self.alpha(observations, t, state)
        else:
            # TODO rename
            temp = 0
            for j in self.states:
                p = self.transition[state][j] * self.alpha(observations, t - 1, j)
                if p > temp:
                    temp = p
            return self.emission[observations[t]][state] * temp

    # The score for the state sequence as a whole is the best score for any final state.
    def P_hat(self, observations):
        T = len(observations)
        temp = 0
        for i in self.states:
            p = self.delta(observations, T - 1, i)
            if p > temp:
                temp = p
        return temp

    def s_hat(self, observations, t):
        T = len(observations)
        if t < 0 or t >= T:
            raise ValueError("t must be >= 0 and < T")
        if t == T - 1:
            temp = 0
            temp_state = None
            for i in self.states:
                p = self.delta(observations, T - 1, i)
                if p > temp:
                    temp = p
                    temp_state = i
            return temp_state
        else: return self.psi(observations, t, self.s_hat(observations, t + 1))

    def psi(self, observations, t, state):
        T = len(observations)
        # There is no base case as such because the initial state has no predecessor.
        if t < 1 or t >= T:
            raise ValueError("t must be >= 1 and < T")
        else:
            temp = 0
            temp_state = None
            for j in self.states:
                p = self.transition[state][j] * self.delta(observations, t - 1, j)
                if p > temp:
                    temp = p
                    temp_state = j
            return temp_state



if __name__ == "__main__":
    hmm = HiddenMarkovModel([0, 1], [0.5, 0.5], [[0.8, 0.4], [0.2, 0.6]], [0, 1], [[0.9, 0.1], [0.1, 0.9]])
    for i in range(100):
        print(hmm.status_at(i))
    # TODO rename
    temp = 0
    for state in range(2):
        for t0 in range(2):
            for t1 in range(2):
                for t2 in range(2):
                    alpha_value = hmm.alpha([t0, t1, t2], 2, state)
                    temp += alpha_value
                    print(alpha_value)
    print(temp)
    print(hmm.beta([0, 0, 0], 2, 1))
    print(hmm.beta([0, 0, 0], 1, 1))
    print(hmm.beta([0, 0, 0], 0, 1))
