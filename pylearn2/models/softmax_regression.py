"""
Softmax regression

Note: softmax regression is implemented as a special case of the MLP.
     It is an MLP with no hidden layers, and with the output layer
     always set to be softmax.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

from pylearn2.models import mlp

class SoftmaxRegression(mlp.MLP):
    """
    A softmax regression model. TODO: add reference.
    This is just a convenience class making a logistic regression model
    out of an MLP.
    """

    def __init__(self,
            n_classes,
            batch_size=None,
            input_space=None,
            irange=None,
            istdev=None,
            W_lr_scale=None,
            b_lr_scale=None,
            max_row_norm=None,
            max_col_norm=None,
            sparse_init=None,
            dropout_include_probs = None,
            dropout_scales = None,
            dropout_input_include_prob = None,
            dropout_input_scale = None,
            init_bias_target_marginals = None,
            nvis=None,
            seed=None):
        """
            layers: a list of MLP_Layers. The final layer will specify the
                    MLP's output space.
            batch_size: optional. if not None, then should be a positive
                        integer. Mostly useful if one of your layers
                        involves a theano op like convolution that requires
                        a hard-coded batch size.
            input_space: a Space specifying the kind of input the MLP acts
                        on. If None, input space is specified by nvis.

            See pylearn2.models.MLP for notes on dropout.
        """

        super(SoftmaxRegression, self).__init__(
                layers=[mlp.Softmax(n_classes=n_classes, layer_name='y',
                    irange=irange, istdev=istdev, sparse_init=sparse_init,
                    W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale,
                    max_row_norm=max_row_norm, max_col_norm=max_col_norm,
                    init_bias_target_marginals=init_bias_target_marginals)],
                batch_size=batch_size,
                input_space=input_space,
                dropout_input_include_prob=dropout_input_include_prob,
                dropout_input_scale=dropout_input_scale,
                nvis=nvis,
                seed=seed)

