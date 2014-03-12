__author__ = "Ian Goodfellow"

from pylearn2.sandbox.lisa_rl.bandit.environment import Environment

class ClassifierBandit(Environment):
    """
    An n-armed contextual bandit based on a classification problem.

    Each of the n-arms corresponds to a different class. If the agent
    selects the correct class for the given context, the environment
    gives reward 1. Otherwise, the environment gives reward 0.

    .. todo::

        WRITEME : parameter list
    """

    def __init__(self, dataset, batch_size):
        self.__dict__.update(locals())
        del self.self

    def get_context_func(self):
        """
        Returns a callable that takes no arguments and returns a minibatch
        of contexts. Minibatch should be in VectorSpace(n).
        """

        def rval():
            X, y = self.dataset.get_batch_design(self.batch_size, include_labels=True)
            self.y_cache = y
            return X

        return rval

    def get_action_func(self):
        """
        Returns a callable that takes no arguments and returns a minibatch of
        rewards.
        Assumes that this function has been called after a call to context_func
        that gave the contexts used to choose the actions.
        """

        def rval(a):
            return (a * self.y_cache).sum(axis=1)

        return rval

    def get_learn_func(self):
        """
        Returns a callable that takes a minibatch of contexts, a minibatch of
        actions, and a minibatch of rewards, and updates the model according
        to them.
        """

        raise NotImplementedError()
