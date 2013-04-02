__authors__ = 'Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from pylearn2.costs.cost import Cost

class Dropout(Cost):
    """
    Implements the dropout training technique described in
    "Improving neural networks by preventing co-adaptation of feature detectors"
    Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov
    arXiv 2012
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None):
        """
        During training, each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y, ** kwargs):
        Y_hat = model.dropout_fprop(X, default_input_include_prob=self.default_input_include_prob,
                input_include_probs=self.input_include_probs, default_input_scale=self.default_input_scale,
                input_scales=self.input_scales
                )
        return model.cost(Y, Y_hat)
