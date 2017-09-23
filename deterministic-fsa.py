# -*- coding:utf-8 -*-


class DeterministicFSA(object):
    def __init__(self, states, vocabulary, initial, final, transition):
        self.states = states
        self.vocabulary = vocabulary
        self.initial = initial
        self.final = final
        self.transition = transition
