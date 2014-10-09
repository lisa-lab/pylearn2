"""
TrainExtension subclass for calculating ROC AUC scores on monitoring
dataset(s), reported via monitor channels.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

import theano
from theano import gof, config
from theano import tensor as T

from pylearn2.train_extensions import TrainExtension


class RocAucScoreOp(gof.Op):
    """
    Theano Op wrapping sklearn.metrics.roc_auc_score.

    Parameters
    ----------
    name : str, optional (default 'roc_auc')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='roc_auc', use_c_code=theano.config.cxx):
        super(RocAucScoreOp, self).__init__(use_c_code)
        self.name = name

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
        output = [T.scalar(name=self.name, dtype=config.floatX)]
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
        if roc_auc_score is None:
            raise RuntimeError("Could not import from sklearn.")
        y_true, y_score = inputs
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            roc_auc = np.nan
        output_storage[0][0] = theano._asarray(roc_auc, dtype=config.floatX)


class RocAucChannel(TrainExtension):
    """
    Adds a ROC AUC channel to the monitor for each monitoring dataset.

    This monitor will return nan unless both classes are represented in
    y_true. For this reason, it is recommended to set monitoring_batches
    to 1, especially when using unbalanced datasets.

    Parameters
    ----------
    channel_name_suffix : str, optional (default 'roc_auc')
        Channel name suffix.
    positive_class_index : int, optional (default 1)
        Index of positive class in predicted values.
    negative_class_index : int or None, optional (default None)
        Index of negative class in predicted values for calculation of
        one vs. one performance. If None, uses all examples not in the
        positive class (one vs. the rest).
    """
    def __init__(self, channel_name_suffix='roc_auc', positive_class_index=1,
                 negative_class_index=None):
        self.channel_name_suffix = channel_name_suffix
        self.positive_class_index = positive_class_index
        self.negative_class_index = negative_class_index

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
        y_hat = model.fprop(state)[:, self.positive_class_index]

        # one vs. the rest
        if self.negative_class_index is None:
            y = T.eq(y, self.positive_class_index)

        # one vs. one
        else:
            pos = T.eq(y, self.positive_class_index)
            neg = T.eq(y, self.negative_class_index)
            keep = T.add(pos, neg).nonzero()
            y = T.eq(y[keep], self.positive_class_index)
            y_hat = y_hat[keep]

        roc_auc = RocAucScoreOp(self.channel_name_suffix)(y, y_hat)
        roc_auc = T.cast(roc_auc, config.floatX)
        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{0}_{1}'.format(dataset_name,
                                                self.channel_name_suffix)
            else:
                channel_name = self.channel_name_suffix
            model.monitor.add_channel(name=channel_name,
                                      ipt=(state, target),
                                      val=roc_auc,
                                      data_specs=(m_space, m_source),
                                      dataset=dataset)
