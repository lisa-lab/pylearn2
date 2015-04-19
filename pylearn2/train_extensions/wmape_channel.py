"""
TrainExtension subclass for calculating Weighted Mean Average Percentage
Error scores for regression models on monitoring dataset(s), reported via
monitor channels.
:math:`WMAPE = \frac{\sum_i|R_i - P_i|}{\sum_i |R_i|}`
"""

__author__ = "Junbo chen"
__copyright__ = "Copyright 2015, Alibaba Group"
__license__ = "3-clause BSD"
__maintainer__ = "Junbo Chen"

from theano import config
from theano import tensor as T

from pylearn2.train_extensions import TrainExtension


class WMapeNumeratorChannel(TrainExtension):
    """
    Adds a WMape Numerator channel to the monitor for each monitoring dataset.
    It calculates the numerator of the WMAPE formula:
    :math:`WMAPE = \frac{\sum_i|R_i - P_i|}{\sum_i |R_i|}`

    Parameters
    ----------
    channel_name_suffix : str, optional (default 'wmape_num')
        Channel name suffix.

    Notes
    ----------
    * See also: WMapeDenominatorChannel defines the denominator of the
        WMAPE formula.
    """
    def __init__(self, channel_name_suffix='wmape_num'):
        self.channel_name_suffix = channel_name_suffix

    def setup(self, model, dataset, algorithm):
        """
        Add WMAPE Numerator channels for monitoring dataset(s) to
        model.monitor.

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

        wmape_numerator = abs(y - y_hat).sum()
        wmape_numerator = T.cast(wmape_numerator, config.floatX)
        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{0}_{1}'.format(dataset_name,
                                                self.channel_name_suffix)
            else:
                channel_name = self.channel_name_suffix
            model.monitor.add_channel(name=channel_name,
                                      ipt=(state, target),
                                      val=wmape_numerator,
                                      data_specs=(m_space, m_source),
                                      dataset=dataset)


class WMapeDenominatorChannel(TrainExtension):
    """
    Adds a WMape Denominator channel to the monitor for each monitoring
    dataset. It calculates the denominator of the WMAPE formula:
    :math:`WMAPE = \frac{\sum_i|R_i - P_i|}{\sum_i |R_i|}`

    Parameters
    ----------
    channel_name_suffix : str, optional (default 'wmape_den')
        Channel name suffix.

    Notes
    ----------
    * See also: WMapeNumeratorChannel defines the numerator of the
        WMAPE formula.
    """
    def __init__(self, channel_name_suffix='wmape_den'):
        self.channel_name_suffix = channel_name_suffix

    def setup(self, model, dataset, algorithm):
        """
        Add WMAPE Denominator channels for monitoring dataset(s) to
        model.monitor.

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

        wmape_denominator = abs(y).sum()
        wmape_denominator = T.cast(wmape_denominator, config.floatX)
        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{0}_{1}'.format(dataset_name,
                                                self.channel_name_suffix)
            else:
                channel_name = self.channel_name_suffix
            model.monitor.add_channel(name=channel_name,
                                      ipt=(state, target),
                                      val=wmape_denominator,
                                      data_specs=(m_space, m_source),
                                      dataset=dataset)
