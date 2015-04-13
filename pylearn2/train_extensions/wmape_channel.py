"""
TrainExtension subclass for calculating Weighted Mean Average Percentage 
Error scores for regression models on monitoring dataset(s), reported via 
monitor channels.
"""

__author__ = "Junbo chen"
__copyright__ = "Copyright 2015, Alibaba Group"
__license__ = "3-clause BSD"
__maintainer__ = "Junbo Chen"

import numpy as np

import theano
from theano import gof, config
from theano import tensor as T

from pylearn2.train_extensions import TrainExtension


class WMapeOp(gof.Op):
    """
    Theano Op calculate wmape score.

    Parameters
    ----------
    name : str, optional (default 'wmape')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='wmape', use_c_code=theano.config.cxx):
        super(WMapeOp, self).__init__(use_c_code)
        self.name = name

    def make_node(self, y_true, y_score):
        """
        Calculate WMAPE score.

        Parameters
        ----------
        y_true : tensor_like
            Target regression values.
        y_score : tensor_like
            Predicted regression values.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.scalar(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate WMAPE score.

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
        wmape = np.sum(abs(y_true - y_score)) / np.sum(abs(y_true))      
        output_storage[0][0] = theano._asarray(wmape, dtype=config.floatX)


class WMapeChannel(TrainExtension):
    """
    Adds a WMape channel to the monitor for each monitoring dataset.

    This monitor will return nan unless sum of y_true is not 0.
    For this reason, it is recommended to set monitoring_batches
    to 1, especially when using unbalanced datasets.

    Parameters
    ----------
    channel_name_suffix : str, optional (default 'wmape')
        Channel name suffix.
    """
    def __init__(self, channel_name_suffix='wmape'):
        self.channel_name_suffix = channel_name_suffix

    def setup(self, model, dataset, algorithm):
        """
        Add WMAPE channels for monitoring dataset(s) to model.monitor.

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

        y = target[:, 0]
        y_hat = model.fprop(state)[:, 0]

        wmape = WMapeOp(self.channel_name_suffix)(y, y_hat)
        wmape = T.cast(wmape, config.floatX)
        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{0}_{1}'.format(dataset_name,
                                                self.channel_name_suffix)
            else:
                channel_name = self.channel_name_suffix
            model.monitor.add_channel(name=channel_name,
                                      ipt=(state, target),
                                      val=wmape,
                                      data_specs=(m_space, m_source),
                                      dataset=dataset)
