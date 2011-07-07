
class DefaultTrainingAlgorithm(object):
    def __init__(self, batch_size = None , batches_per_iter = 1000 , monitoring_batches = - 1, monitoring_dataset = None):
        """
        if batch_size is None, reverts to the force_batch_size field of the model
        """

        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.monitoring_dataset, self.monitoring_batches = monitoring_dataset, monitoring_batches
    #

    def setup(self, model):
        self.model = model

    def train(self, dataset):
        model = self.model

        if self.batch_size is None:
            batch_size = model.force_batch_size
        else:
            batch_size = self.batch_size
            if hasattr(model,'force_batch_size'):
                assert model.force_batch_size <= 0 or batch_size == model.force_batch_size

        if len(model.error_record) == 0 and self.monitoring_dataset:
            self.monitor(model)
        #

        for i in xrange(self.batches_per_iter):
            model.learn(dataset, batch_size)
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
