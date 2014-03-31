from pylearn2.train_extensions import TrainExtension
from pylearn2.training_algorithms.bgd import BGD
from theano import tensor as T
from theano import gof, config
import theano
import sklearn.metrics
import numpy as np


class ROCAUCScoreOp(gof.Op):
    # See function roc_auc_score for docstring
    def make_node(self, y_true, y_score):
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.TensorType('float64', []).make_variable(name='roc_auc')]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        y_true, y_score = inputs
        try:
            roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        except ValueError:
            roc_auc = np.nan
        output_storage[0][0] = theano._asarray(roc_auc, dtype='float64')


def roc_auc_score(y_true, y_score):
    """Calculate ROC AUC score.

    Parameters
    ----------
    y_true: tensor_like
        Target values.
    y_score: tensor_like
        Predicted values or probabilities for positive class.
    """
    return ROCAUCScoreOp()(y_true, y_score)


class ROCAUCChannel(TrainExtension):
    """Adds a ROC AUC channel to the monitor for each monitoring dataset.

    Notes
    -----
    This monitor will return nan unless both classes are represented in y_true.

    Currently only supports BGD, and requires monitoring_batches and
    batches_per_iter to be set to 1 to avoid class population issues.
    """
    def setup(self, model, dataset, algorithm):
        
        # sanity checks
        try:
            assert isinstance(algorithm, BGD)
        except AssertionError:
            # TODO: workaround for batch_size requirement of SGD
            raise NotImplementedError("ROC AUC is only supported when using " +
                                      "batch gradient descent (BGD).")
        try:
            assert algorithm.batch_size is None
            assert algorithm.monitoring_batches == 1
            assert algorithm.batches_per_iter == 1
        except AssertionError:
            # TODO: stratified dataset iterator to supply training batches
            raise ValueError("batch_size should be None and " +
                             "monitoring_batches and batches_per_iter " +
                             "should both be set to 1 to avoid class " +
                             "population issues.")

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
