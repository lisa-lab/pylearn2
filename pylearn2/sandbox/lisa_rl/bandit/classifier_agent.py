__author__ = "Ian Goodfellow"

import logging
import time

from theano import function
import theano.tensor as T

from pylearn2.sandbox.lisa_rl.bandit.agent import Agent
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_theano_rng


logger = logging.getLogger(__name__)


class ClassifierAgent(Agent):
    """
    A contextual bandit agent that knows a priori that the task is
    classification. Specifically, the actions the agent can take
    are to output k different one-hot vectors. If it outputs the
    correct code for the current input it gets a reward of 1,
    otherwise it gets a reward of 0.

    This hard-coded prior knowledge means that the expected reward
    of taking action a is just the probability of a being the
    correct class.

    This makes the contextual bandit problem as close as possible
    to the classification task, so that any loss in performance
    comes only from needing to explore to discover the correct
    action for for each input.

    .. todo::

        WRITEME : parameter list

    Parameters
    ----------
    stochastic: bool
        If True, samples actions from P(y | x) otherwise, uses argmax_y P(y |x)
    """

    def __init__(self, mlp, learning_rule, init_learning_rate, cost,
            update_callbacks, stochastic=False, epsilon=None, neg_target=False,
            ignore_wrong=False, epsilon_stochastic=None):
        self.__dict__.update(locals())
        del self.self

        self.learning_rate = sharedX(init_learning_rate)

    def get_decide_func(self):
        """
        Returns a theano function that takes a minibatch
        (num_examples, num_features) of contexts and returns
        a minibatch (num_examples, num_classes) of one-hot codes
        for actions.
        """

        X = T.matrix()
        y_hat = self.mlp.fprop(X)

        theano_rng = make_theano_rng(None, 2013+11+20, which_method="multinomial")
        if self.stochastic:
            a = theano_rng.multinomial(pvals=y_hat, dtype='float32')
        else:
            mx = T.max(y_hat, axis=1).dimshuffle(0, 'x')
            a = T.eq(y_hat, mx)

        if self.epsilon is not None:
            a = theano_rng.multinomial(pvals = (1. - self.epsilon) * a +
                    self.epsilon * T.ones_like(y_hat) / y_hat.shape[1],
                    dtype = 'float32')

        if self.epsilon_stochastic is not None:
            a = theano_rng.multinomial(pvals = (1. - self.epsilon_stochastic) * a +
                    self.epsilon_stochastic * y_hat,
                    dtype = 'float32')

        logger.info("Compiling classifier agent learning function")
        t1 = time.time()
        f = function([X], a)
        t2 = time.time()

        logger.info("...done, took {0}".format(t2 - t1))

        return f

    def get_learn_func(self):
        """
        Returns a theano function that does a learning update when passed
        a context, the action that the agent chose, and the reward it got.

        This agent expects the action to be a matrix of one-hot class
        selections and the reward to be a vector of 0 / 1 rewards per example.
        """

        contexts = T.matrix()
        actions = T.matrix()
        rewards = T.vector()

        assert sum([self.neg_target, self.ignore_wrong]) <= 1
        if self.neg_target:
            signed_rewards = 2. * rewards - 1.
            fake_targets = actions * signed_rewards.dimshuffle(0, 'x')
        elif self.ignore_wrong:
            fake_targets = actions * rewards.dimshuffle(0, 'x')
        else:
            correct_actions = actions * rewards.dimshuffle(0, 'x')
            roads_not_taken = (T.ones_like(actions) - actions) / (T.cast(actions.shape[1], 'float32') - 1.)
            #from theano.printing import Print
            #roads_not_taken = Print('roads_not_taken')(roads_not_taken)
            fake_targets = correct_actions + roads_not_taken * (1 - rewards).dimshuffle(0, 'x')

        lr_scalers = self.mlp.get_lr_scalers()

        grads, updates = self.cost.get_gradients(self.mlp, (contexts, fake_targets))

        updates.update(self.learning_rule.get_updates(
                self.learning_rate, grads, lr_scalers))

        self.mlp.modify_updates(updates)

        learn_func = function([contexts, actions, rewards], updates=updates)

        def rval(contexts, actions, rewards):
            learn_func(contexts, actions, rewards)
            for callback in self.update_callbacks:
                callback(self)

        return rval

    def get_weights(self):
        return self.mlp.get_weights()

    def get_weights_format(self):
        return self.mlp.get_weights_format()

    def get_weights_topo(self):
        return self.mlp.get_weights_topo()

    def get_weights_view_shape(self):
        return self.mlp.get_weights_view_shape()

    def get_params(self):
        return self.mlp.get_params()

    def set_batch_size(self, batch_size):
        self.mlp.set_batch_size(batch_size)

    def get_input_space(self):
        return self.mlp.get_input_space()

    def get_output_space(self):
        return self.mlp.get_output_space()

    def fprop(self, state_below):
        return self.mlp.fprop(state_below)
