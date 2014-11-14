__author__ = "Ian Goodfellow"

import numpy as np

from theano import function
from theano import tensor as T


from pylearn2.compat import OrderedDict
from pylearn2.sandbox.lisa_rl.bandit.agent import Agent
from pylearn2.utils import sharedX

class AverageAgent(Agent):
    """
    A simple n-armed bandit playing agent that always plays the
    arm with the highest estimated reward. The estimated reward is just
    based on the average of all observations from that arm. If an arm
    has not been tried, the estimated reward is given by init_reward_estimate.

    .. todo::

        WRITEME : parameter list
    """

    def __init__(self, init_reward_estimate, num_arms):
        self.__dict__.update(locals())
        del self.self
        self.estimated_rewards = sharedX(np.zeros((num_arms,)) \
                + self.init_reward_estimate)
        self.observation_counts = sharedX(np.zeros((num_arms,)))


    def get_decide_func(self):
        """
        Returns a theano function that decides what action to take.
        Since this is a bandit playing agent, there is no input.
        """

        # Cast is for compatibility with default bit depth of T.iscalar
        # (wtf, theano?)
        return function([], T.cast(T.argmax(self.estimated_rewards), 'int32'))

    def get_learn_func(self):
        """
        Returns a theano function that takes an action and a reward,
        and updates the agent based on this experience.
        """

        a = T.iscalar()
        r = T.scalar()

        old_estimated_reward = self.estimated_rewards[a]
        old_observation_count = self.observation_counts[a]
        observation_count = old_observation_count + 1.

        delta = r - old_estimated_reward
        new_estimated_reward = old_estimated_reward + delta / observation_count

        new_estimated_rewards = T.set_subtensor(self.estimated_rewards[a],
            new_estimated_reward)
        new_observation_counts = T.set_subtensor(self.observation_counts[a], observation_count)

        updates = OrderedDict([
            (self.estimated_rewards, new_estimated_rewards),
            (self.observation_counts, new_observation_counts)
            ])

        rval = function([a, r], updates=updates)

        return rval
