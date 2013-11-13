__author__ = "Ian Goodfellow"

import numpy as np

from theano import config
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.sandbox.lisa_rl.bandit.environment import Environment
from pylearn2.utils import sharedX


class GaussianBandit(Environment):
    """
    An n-armed bandit whose rewards are drawn from a different Gaussian
    distribution for each arm.
    The mean and standard deviation of the reward for each arm is drawn
    at initialization time from N(0, <corresponding std arg>).
    (For the standard deviation we use the absolute value of the Gaussian
    sample)
    """

    def __init__(self, num_arms, mean_std = 1.0, std_std = 1.0):
        self.rng = np.random.RandomState([2013, 11, 12])
        self.means = sharedX(self.rng.randn(num_arms) * mean_std)
        self.stds = sharedX(np.abs(self.rng.randn(num_arms) * std_std))
        self.theano_rng = MRG_RandomStreams(self.rng.randint(2 ** 16))

    def get_action_func(self):
        """
        Returns a theano function that takes an action and returns a reward.
        """

        action = T.iscalar()
        reward_mean = self.means[action]
        reward_std = self.stds[action]
        reward = self.theano_rng.normal(avg=reward_mean, std=reward_std,
                dtype=config.floatX, size=reward_mean.shape)
        rval = function([action], reward)
        return rval
