
class DefaultTrainingAlgorithm:
    def __init__(self, batch_size, batches_per_iter , monitoring_batches = - 1, monitoring_dataset = None):
        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.monitoring_dataset, self.monitoring_batches = monitoring_dataset, monitoring_batches
    #

    def train(self, model, dataset):

        for i in xrange(self.batches_per_iter):
            model.learn(dataset, self.batch_size)
        #

        if self.monitoring_dataset:
            self.monitoring_dataset.reset_RNG()
            model.record_monitoring_error(self.monitoring_dataset,batches=self.monitoring_batches,batch_size=self.batch_size)
            print model.error_record[-1]
        #


        return True
    #
#
