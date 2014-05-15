"""
TrainExtension subclass for calculating ROC AUC scores on monitoring
dataset(s), reported via monitor channels.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
import warnings
try:
    import sklearn.metrics
except ImportError:
    warnings.warn("Cannot import from sklearn.")

import theano
from theano import gof, config
from theano import tensor as T

from pylearn2.train_extensions import TrainExtension


class RocAucScoreOp(gof.Op):
    """
    Theano Op wrapping sklearn.metrics.roc_auc_score.

    Parameters
    ----------
    use_c_code : WRITEME
    """
    def make_node(self, y_true, y_score):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        y_true : tensor_like
            Target class labels.
        y_score : tensor_like
            Predicted class labels or probabilities for positive class.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.TensorType('float64', []).make_variable(name='roc_auc')]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        node : Apply instance
            Symbolic inputs and outputs.
        inputs : list
            Sequence of inputs.
        output_storage : list
            List of mutable 1-element lists.
        """
        y_true, y_score = inputs
        try:
            roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        except ValueError:
            roc_auc = np.nan
        output_storage[0][0] = theano._asarray(roc_auc, dtype='float64')


def roc_auc_score(y_true, y_score):
    """
    Calculate ROC AUC score.

    Parameters
    ----------
    y_true: tensor_like
        Target class values.
    y_score: tensor_like
        Predicted class labels or probabilities for positive class.
    """
    return RocAucScoreOp()(y_true, y_score)


class RocAucChannel(TrainExtension):
    """
    Adds a ROC AUC channel to the monitor for each monitoring dataset.

    This monitor will return nan unless both classes are represented in
    y_true. For this reason, it is recommended to set monitoring_batches
    to 1, especially when using unbalanced datasets.
    """
    def setup(self, model, dataset, algorithm):
        """
        Add ROC AUC channels for monitoring dataset(s) to model.monitor.

        Parameters
        ----------
        model : object
            The model being trained.
        dataset : object
            Training dataset.
        algorithm : object
            Training algorithm.
        """
        m_space, m_source = model.get_monitoring_data_specs()
        state, target = m_space.make_theano_batch()
        y = T.argmax(target, axis=1)
        y_hat = model.fprop(state)[:, 1]
        roc_auc = roc_auc_score(y, y_hat)
        roc_auc = T.cast(roc_auc, config.floatX)
        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{}_y_roc_auc'.format(dataset_name)
            else:
                channel_name = 'y_roc_auc'
            model.monitor.add_channel(name=channel_name,
                                      ipt=(state, target),
                                      val=roc_auc,
                                      data_specs=(m_space, m_source),
                                      dataset=dataset)
