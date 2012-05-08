"""An example of how to use the library so far."""
# Standard library imports
import sys

# Third-party imports
import numpy
import theano
from theano import tensor

try:
    import pylearn2
except ImportError:
    print >>sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

# Local imports
from pylearn2.cost import SquaredError
from pylearn2.autoencoder import ContractiveAutoencoder, build_stacked_ae
from pylearn2.optimizer import SGDOptimizer

if __name__ == "__main__":
    # Simulate some fake data.
    rng = numpy.random.RandomState(seed=42)
    data = rng.normal(size=(1000, 15))

    conf = {
        'corruption_level': 0.1,
        'nhid': 20,
        'nvis': data.shape[1],
        'anneal_start': 100,
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
    minibatch = theano.printing.Print('min')(minibatch)

    # Allocate a denoising autoencoder with binomial noise corruption.
    cae = ContractiveAutoencoder(conf['nvis'], conf['nhid'],
                                 conf['act_enc'], conf['act_dec'])

    # Allocate an optimizer, which tells us how to update our model.
    cost = SquaredError(cae)(minibatch, cae.reconstruct(minibatch)).mean()
    cost += cae.contraction_penalty(minibatch).mean()
    trainer = SGDOptimizer(cae, conf['base_lr'], conf['anneal_start'])
    updates = trainer.cost_updates(cost)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], cost, updates=updates)

    # Suppose we want minibatches of size 10
    batchsize = 10

    # Here's a manual training loop. I hope to have some classes that
    # automate this a litle bit.
    for epoch in xrange(5):
        for offset in xrange(0, data.shape[0], batchsize):
            minibatch_err = train_fn(data[offset:(offset + batchsize)])
            print ("epoch %d, batch %d-%d: %f" %
                   (epoch, offset, offset + batchsize - 1, minibatch_err))

    # Suppose you then want to use the representation for something.
    transform = theano.function([minibatch], cae(minibatch))

    print "Transformed data:"
    print numpy.histogram(transform(data))

    # We'll now create a stacked denoising autoencoder. First, we change
    # the number of hidden units to be a list. This tells the build_stacked_AE
    # method how many layers to make.
    stack_conf = conf.copy()
    stack_conf['nhids'] = [20, 20, 10]
    #choose which layer is a regular da and which one is a cae
    stack_conf['contracting']=[True,False,True]
    stack_conf['anneal_start'] = None # Don't anneal these learning rates
    scae = build_stacked_ae(nvis=stack_conf['nvis'],
                            nhids=stack_conf['nhids'],
                            act_enc=stack_conf['act_enc'],
                            act_dec=stack_conf['act_dec'],
                            contracting=stack_conf['contracting'])

    # To pretrain it, we'll use a different SGDOptimizer for each layer.
    optimizers = []
    thislayer_input = [minibatch]
    for layer in scae.layers():
        cost = SquaredError(layer)(thislayer_input[0],
                layer.reconstruct(thislayer_input[0])
                ).mean()
        if isinstance(layer,ContractiveAutoencoder):
            cost+=layer.contraction_penalty(thislayer_input[0]).mean()
        opt = SGDOptimizer( layer.get_param_values(),
                            stack_conf['base_lr'],
                            stack_conf['anneal_start']
                            )
        optimizers.append((opt, cost))
        # Retrieve a Theano function for training this layer.
        updates = opt.cost_updates(cost)
        thislayer_train_fn = theano.function([minibatch], cost, updates=updates)

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
