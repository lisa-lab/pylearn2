"""An example of how to use the library so far."""
# Standard library imports
from itertools import izip

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from cost import MeanSquaredError
from corruption import GaussianCorruptor
from autoencoder import DenoisingAutoencoder, StackedDA
from optimizer import SGDOptimizer

if __name__ == "__main__":
    # Simulate some fake data.
    data = numpy.random.normal(size=(1000, 15))

    conf = {
        'corruption_level': 0.1,
        'n_hid': 20,
        'n_vis': data.shape[1],
        'lr_anneal_start': 100,
        'base_lr': 0.01,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': None,
        #'lr_hb': 0.10,
        #'lr_vb': 0.10,
        'irange': 0.001,
    }

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Allocate a denoising autoencoder with binomial noise corruption.
    corruptor = GaussianCorruptor(conf)
    da = DenoisingAutoencoder(conf, corruptor)

    # Allocate an optimizer, which tells us how to update our model.
    # TODO: build the cost another way
    cost = MeanSquaredError(conf, da)([minibatch])
    trainer = SGDOptimizer(conf, da, cost)

    # Finally, build a Theano function out of all this.
    train_fn = trainer.function([minibatch])

    # Suppose we want minibatches of size 10
    batchsize = 10

    # Here's a manual training loop. I hope to have some classes that
    # automate this a litle bit.
    for epoch in xrange(5):
        for offset in xrange(0, data.shape[0], batchsize):
            minibatch_err = train_fn(data[offset:(offset + batchsize)])
            print "epoch %d, batch %d-%d: %f" % \
                    (epoch, offset, offset + batchsize - 1, minibatch_err)

    # Suppose you then want to use the representation for something.
    transform = theano.function([minibatch], da([minibatch])[0])

    print "Transformed data:"
    print numpy.histogram(transform(data))

    # We'll now create a stacked denoising autoencoder. First, we change
    # the number of hidden units to be a list. This tells the StackedDA
    # class how many layers to make.
    sda_conf = conf.copy()
    sda_conf['n_hid'] = [20, 20, 10]
    sda = StackedDA(sda_conf, corruptor)

    # To pretrain it, we'll use a different SGDOptimizer for each layer.
    optimizers = []
    thislayer_input = [minibatch]
    for layer in sda.layers():
        cost = MeanSquaredError(sda_conf, layer)([thislayer_input[0]])
        opt = SGDOptimizer(sda_conf, layer, cost)
        optimizers.append(opt)
        # Retrieve a Theano function for training this layer.
        thislayer_train_fn = opt.function([minibatch])

        # Train as before.
        for epoch in xrange(10):
            for offset in xrange(0, data.shape[0], batchsize):
                minibatch_err = thislayer_train_fn(
                    data[offset:(offset + batchsize)]
                )
                print "epoch %d, batch %d-%d: %f" % \
                        (epoch, offset, offset + batchsize - 1, minibatch_err)

        # Now, get a symbolic input for the next layer.
        thislayer_input = layer(thislayer_input)
