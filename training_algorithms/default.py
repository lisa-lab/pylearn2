
class DefaultTrainingAlgorithm:
    def __init__(self, batch_size, batches_per_iter):
        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
    #

    def train(self, model, dataset):

        for i in xrange(self.batches_per_iter):
            print i
            model.learn(dataset, self.batch_size)
        #

        return True
    #
#
