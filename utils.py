# utils.py
# Coded by George H. Chen (georgehc@mit.edu) -- updated 12/3/2012 12:36am
from __future__ import division
import numpy as np

#-----------------------------------------------------------------------------
# A general purpose Distribution class for finite discrete random variables
#

class Distribution(dict):
    """
    The Distribution class extend the Python dictionary such that
    each key's value should correspond to the probability of the key.

    For example, here's how you can create a random variable X that takes on
    value 'spam' with probability .7 and 'eggs' with probability .3:

    X = Distribution()
    X['spam'] = .7
    X['eggs'] = .3

    Methods
    -------
    renormalize():
      scales all the probabilities so that they sum to 1
    get_mode():
      returns an item with the highest probability, breaking ties arbitrarily
    sample():
      draws a sample from the Distribution
    """
    def __missing__(self, key):
        # if the key is missing, return probability 0
        return 0

    def renormalize(self):
        normalization_constant = sum(self.itervalues())
        assert normalization_constant > 0, "Probabilities shouldn't all be 0"
        for key in self.iterkeys():
            self[key] /= normalization_constant

    def get_mode(self):
        maximum = -1
        arg_max = None

        for key in self.iterkeys():
            if self[key] > maximum:
                arg_max = key
                maximum = self[key]

        return arg_max

    def sample(self):
        keys  = []
        probs = []
        for key, prob in self.iteritems():
            if prob > 0:
                keys.append(key)
                probs.append(prob)

        rand_idx = np.where(np.random.multinomial(1, probs))[0][0]
        return keys[rand_idx]


#-----------------------------------------------------------------------------
# General purpose functions for finite discrete random variables
#

def KL_divergence(dist1, dist2):
    # returns KL divergence D(dist1 || dist2)
    divergence = 0.
    for x in dist1:
        if dist1[x] > 0:
            if x not in dist2 or dist2[x] == 0:
                divergence = np.inf
                return divergence
            else:
                divergence += dist1[x]*(np.log2(dist1[x]) - np.log2(dist2[x]))
    return divergence

