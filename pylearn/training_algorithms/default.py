from framework.utils import serial #rm
import numpy as N#rm

class DefaultTrainingAlgorithm:
    def __init__(self, batch_size, batches_per_iter , monitoring_batches = - 1, monitoring_dataset = None):
        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.monitoring_dataset, self.monitoring_batches = monitoring_dataset, monitoring_batches
    #

    def train(self, model, dataset):

        if len(model.error_record) == 0 and self.monitoring_dataset:
            self.monitor(model)
        #

        for i in xrange(self.batches_per_iter):
            model.learn(dataset, self.batch_size)
        #

        if self.monitoring_dataset:
            self.monitor(model)
        #

        return True
    #

    def monitor(self, model):
        if True:
            s = self.monitoring_dataset.get_stream_position()

            self.monitoring_dataset.restart_stream()

            model.record_monitoring_error(self.monitoring_dataset,batches=self.monitoring_batches,batch_size=self.batch_size)
            print model.error_record[-1]

            self.monitoring_dataset.set_stream_position(s)
        #
    #
#
