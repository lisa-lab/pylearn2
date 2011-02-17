"""An example of how to use the library so far."""
import numpy
import theano
from theano import tensor
from corruption import BinomialCorruptor
from autoencoder import DenoisingAutoencoder, DATrainer

if __name__ == "__main__":
    conf = {
        'corruption_level': 0.2,
        'n_hid': 20,
        'n_vis': 15,
        'lr_anneal_start': 100,
        'base_lr': 0.01,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': None,
        'lr_hb': 0.05,
        'lr_vb': 0.10
    }

    # A symbolic input representing your minibatch.
    minibatch = tensor.dmatrix()

    # Allocate a denoising autoencoder with binomial noise corruption.
    corruptor = BinomialCorruptor.alloc(conf)
    da = DenoisingAutoencoder.alloc(corruptor, conf)

    # Allocate a trainer, which tells us how to update our model.
    trainer = DATrainer.alloc(da, da.mse, minibatch, conf)

    # Finally, build a Theano function out of all this.
    # NOTE: this could be incorporated into a method of the trainer
    #       class, somehow. How would people feel about that?
    #       James: are there disadvantages?
    train = theano.function(
        [minibatch],              # The input you'll pass
        da.mse(minibatch),        # Whatever quantities you want returned
        updates=trainer.updates() # How Theano should update shared vars
    )

    # Simulate some fake data.
    data = numpy.random.normal(1000, 15)

    # Suppose we want minibatches of size 10
    batchsize = 10

    # Here's a manual training loop. I hope to have some classes that
    # automate this a litle bit.
    for epoch in xrange(5):
        for offset in xrange(0, data.shape[0], batchsize):
            minibatch_err = train(data[offset:(offset + batchsize)])
            print "epoch %d, batch %d-%d: %f" % \
                    (epoch, offset, offset + batchsize, minibatch_err)

    # Suppose you then want to use the representation for something.
    transform = theano.function([minibatch], da([minibatch])[0])
