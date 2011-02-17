"""An example of how to use the library so far."""
# Standard library imports
from itertools import izip

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from costs import MeanSquaredError
from corruption import GaussianCorruptor
from autoencoder import DenoisingAutoencoder, DATrainer, StackedDA

if __name__ == "__main__":
    conf = {
        'corruption_level': 0.1,
        'n_hid': 20,
        'n_vis': 15,
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
    minibatch = tensor.dmatrix()

    # Allocate a denoising autoencoder with binomial noise corruption.
    corruptor = GaussianCorruptor.alloc(conf)
    da = DenoisingAutoencoder.alloc(corruptor, conf)

    # Allocate a trainer, which tells us how to update our model.
    cost_fn = MeanSquaredError.alloc(conf, da)
    trainer = DATrainer.alloc(da, cost_fn, minibatch, conf)

    # Finally, build a Theano function out of all this.
    train_fn = trainer.function(minibatch)

    # Simulate some fake data.
    data = numpy.random.normal(size=(1000, 15))

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
    sda = StackedDA.alloc(corruptor, sda_conf)

    # To pretrain it, we'll create a DATrainer for each layer.
    trainers = []
    inp = [minibatch]
    for d in sda.layers():
        cost_fn = MeanSquaredError.alloc(conf, d)
        trainers.append(DATrainer.alloc(d, cost_fn, inp[0], sda_conf))
        # Each time, we'll be passing the data through the layer to
        # obtain a symbolic representation for the input to the next
        # layer.
        inp = d(inp)

    # We'll do roughly the same thing as above, but inside a loop over layers.
    thislayer_input = [minibatch]
    for trainer, layer in izip(trainers, sda.layers()):
        # Retrieve a Theano function for training this layer.
        thislayer_train_fn = trainer.function(minibatch)

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
