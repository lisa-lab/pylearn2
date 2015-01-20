"""
Functionality for training with dropout.
"""
__authors__ = 'Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from functools import wraps

from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost


class Dropout(DefaultDataSpecsMixin, Cost):
    """
    Implements the dropout training technique described in
    "Improving neural networks by preventing co-adaptation of feature
    detectors"
    Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever,
    Ruslan R. Salakhutdinov
    arXiv 2012

    This paper suggests including each unit with probability p during training,
    then multiplying the outgoing weights by p at the end of training.
    We instead include each unit with probability p and divide its
    state by p during training. Note that this means the initial weights should
    be multiplied by p relative to Hinton's.
    The SGD learning rate on the weights should also be scaled by p^2 (use
    W_lr_scale rather than adjusting the global learning rate, because the
    learning rate on the biases should not be adjusted).

    During training, each input to each layer is randomly included or excluded
    for each example. The probability of inclusion is independent for each
    input and each example. Each layer uses "default_input_include_prob"
    unless that layer's name appears as a key in input_include_probs, in which
    case the input inclusion probability is given by the corresponding value.

    Each feature is also multiplied by a scale factor. The scale factor for
    each layer's input scale is determined by the same scheme as the input
    probabilities.

    Parameters
    ----------
    default_input_include_prob : float
        The probability of including a layer's input, unless that layer appears
        in `input_include_probs`
    input_include_probs : dict
        A dictionary mapping string layer names to float include probability
        values. Overrides `default_input_include_prob` for individual layers.
    default_input_scale : float
        During training, each layer's input is multiplied by this amount to
        compensate for fewer of the input units being present. Can be
        overridden by `input_scales`.
    input_scales : dict
        A dictionary mapping string layer names to float values to scale that
        layer's input by. Overrides `default_input_scale` for individual
        layers.
    per_example : bool
        If True, chooses separate units to drop for each example. If False,
        applies the same dropout mask to the entire minibatch.
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
                 default_input_scale=2., input_scales=None, per_example=True):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        todo::

        Parameters
        ----------
        model : TODO
        data : TODO
        kwargs : TODO
        """

        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        Y_hat = model.dropout_fprop(
            X,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )
        return model.cost(Y, Y_hat)

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return True
