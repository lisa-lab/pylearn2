from framework.cost import MeanSquaredError
from framework.optimizer import SGDOptimizer
from theano import function, tensor

class Demo:
    """An example training algorithm. This ports the training from example_da.py
        to the train.py script setup """

    def __init__(self, base_lr = 0.01, anneal_start = 100, batch_size = 10,
                num_epochs = 5):


        self.base_lr = base_lr
        self.anneal_start = anneal_start
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self, model, dataset):
        minibatch = tensor.matrix()

        cost = MeanSquaredError(model)(minibatch, model.reconstruct(minibatch))
        trainer = SGDOptimizer(model, self.base_lr, self.anneal_start)
        updates = trainer.cost_updates(cost)

        train_fn = function([minibatch], cost, updates = updates)

        data = dataset.get_design_matrix()

        for epoch in xrange(self.num_epochs):
            for offset in xrange(0, data.shape[0], self.batch_size):
                minibatch_err = train_fn(data[offset:(offset+self.batch_size)])
                print "epoch %d, batch %d-%d: %f" % (epoch, offset, offset + self.batch_size -1, minibatch_err)
