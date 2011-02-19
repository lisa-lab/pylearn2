"""An example of experiment made with the new library."""
# Standard library imports
import time

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from framework import utils
from framework import cost
from framework import corruption
from framework.utils import BatchIterator
from framework.autoencoder import DenoisingAutoencoder, DATrainer

def basic_trainer(conf):
    """
    This function basically train a denoising autoencoder according
    to the parameters in conf, and save the learned model
    """
    # Load the dataset
    data = utils.load_data(conf)
    conf['n_vis'] = utils.get_constant(data[0].shape[1])

    # A symbolic input representing your minibatch.
    minibatch = tensor.dmatrix()

    # Allocate a denoising autoencoder with a given noise corruption.
    corruptor = corruption.get(conf['corruption_class']).alloc(conf)
    da = DenoisingAutoencoder.alloc(corruptor, conf)

    # Allocate a trainer, which tells us how to update our model.
    cost_fn = cost.get(conf['cost_class']).alloc(conf, da)
    trainer = DATrainer.alloc(da, cost_fn, minibatch, conf)
    train_fn = trainer.function(minibatch)

    # Here's a manual training loop.
    start_time = time.clock()
    batch_time = time.clock()
    batchiter = BatchIterator(conf, data)
    for epoch in xrange(conf['epochs']):
        c = []
        for minibatch in batchiter:
            c.append(train_fn(minibatch))
        train_time = time.clock() - batch_time
        batch_time = time.clock()
        print 'Training epoch %d, time spent (min) %f, cost ' \
            % (epoch, train_time / 60.), numpy.mean(c)

    end_time = time.clock()

    conf['training_time'] = (end_time - start_time) / 60.

    # Compute denoising error for valid and train datasets.
    error_fn = theano.function([minibatch], cost_fn([minibatch]))

    conf['error_valid'] = error_fn(data[1].value)
    conf['error_test'] = error_fn(data[2].value)

    # Save model parameters
    # TODO: Not implemented yet
    # trainer.save(exp['model_dir'], 'model.pkl')

def submit(conf):
    """
    This function create a submission file with a
    model trained according to conf parameters
    """
    # Load the model parameters
    corruptor = corruption.get(conf['corruption_class']).alloc(conf)
    da = DenoisingAutoencoder.alloc(corruptor, conf)
    # TODO: Not implemented yet
    # da.load(exp['model_dir'], 'model.pkl')

    # Create submission file
    minibatch = tensor.dmatrix()
    transform = theano.function([minibatch], da([minibatch])[0])
    utils.create_submission(conf, transform)


if __name__ == "__main__":
    conf = {# Network specific arguments
            'corruption_level': 0.3,
            'n_hid': 500,
            #'n_vis': 15, # Determined by the datasize
            'lr_anneal_start': 100,
            'base_lr': 0.01,
            'tied_weights': True,
            'act_enc': 'sigmoid',
            'act_dec': 'sigmoid',
            #'lr_hb': 0.10,
            #'lr_vb': 0.10,
            'irange': 0.001,
            'cost_class' : 'CrossEntropy',
            'corruption_class' : 'GaussianCorruptor',
            # Experiment specific arguments
            'dataset' : 'ule',
            'expname' : 'myfirstexp',
            'train_prop' : 1,
            'valid_prop' : 0,
            'test_prop' : 0,
            'normalize' : True,
            'normalize_on_the_fly' : False,
            'randomize_valid' : True,
            'randomize_test' : True,
            'batchsize' : 20,
            'epochs' : 50,
            'model_dir' : './outputs/',
            'submission_dir' : './outputs/'
            }


    basic_trainer(conf)
