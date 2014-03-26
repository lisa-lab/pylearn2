from pylearn2.train_extensions import TrainExtension
from theano import tensor as T
from theano import gof, config
import theano
import sklearn.metrics

class ROCAUCScoreOp(gof.Op):
    # See function roc_auc_score for docstring
    def make_node(self, y_true, y_score):
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.TensorType('float64', []).make_variable(name='roc_auc')]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        y_true, y_score = inputs
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        output_storage[0][0] = theano._asarray(roc_auc, dtype='float64')

def roc_auc_score(y_true, y_score):
    """Calculate ROC AUC score.

    Parameters
    ----------
    y_true: tensor_like
        Target values.
    y_score: tensor_like
        Predicted values or probabilities for positive class.

    Notes
    -----
    This method will fail unless both classes are represented in y_true.
    """
    return ROCAUCScoreOp()(y_true, y_score)

class ROCAUCChannel(TrainExtension):
    """Adds a ROC AUC channel to the monitor for each monitoring dataset."""
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor
        m_space, m_source = model.get_monitoring_data_specs()
        state, target = m_space.make_theano_batch()
        y = T.argmax(target, axis=1)
        y_hat = model.fprop(state)[:,1]
        roc_auc = roc_auc_score(y, y_hat)
        roc_auc = T.cast(roc_auc, config.floatX)
        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{}_y_roc_auc'.format(dataset_name)
            else:
                channel_name = 'y_roc_auc'
            monitor.add_channel(name=channel_name,
                                ipt=(state, target),
                                val=roc_auc,
                                data_specs=(m_space, m_source),
                                dataset=dataset)
