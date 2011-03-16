"""An example of experiment made with the new library."""
# Standard library imports
import time
import sys
import os

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
try:
    import framework
except ImportError:
    print >> sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

from framework import utils
from framework.pca import PCA
from framework import cost
from framework import corruption
from framework.utils import BatchIterator
from framework.autoencoder import DenoisingAutoencoder
from framework.optimizer import SGDOptimizer


def train_da(conf, data):
    """
    This function basically train a denoising autoencoder according
    to the parameters in conf, and save the learned model
    """
    # Set visible units size
    conf['nvis'] = utils.get_constant(data[0].shape[1]).item()

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Allocate a denoising autoencoder with a given noise corruption.
    corruptor = corruption.get(conf['corruption_class'])(conf['corruption_level'])
    da = DenoisingAutoencoder(corruptor, conf['nvis'], conf['nhid'],
                              conf['act_enc'], conf['act_dec'])

    # Allocate an optimizer, which tells us how to update our model.
    mycost = cost.get(conf['cost_class'])(da)(minibatch,
                                              da.reconstruct(minibatch))
    trainer = SGDOptimizer(da, conf['base_lr'], conf['anneal_start'])
    updates = trainer.cost_updates(mycost)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], mycost, updates=updates, name='train_fn')

    # Here's a manual training loop.
    print '... training model'
    start_time = time.clock()
    iterator = BatchIterator(data, conf['proba'], conf['batchsize'])
    saving_counter = 0
    saving_rate = conf.get('saving_rate',0)
    alc_counter = 0
    alc_rate = conf.get('alc_rate', 0)
    best_alc_value = 0
    best_alc_epoch = -1
    last_alc_epoch = -1
    for epoch in xrange(conf['epochs']):
        c = []
        batch_time = time.clock()
        for minibatch_data in iterator:
            c.append(train_fn(minibatch_data))

        # Print training time + cost
        train_time = time.clock() - batch_time
        print '... training epoch %d, time spent (min) %f, cost' \
            % (epoch, train_time / 60.), numpy.mean(c)

        # Saving intermediate models
        if saving_rate != 0:
            saving_counter += 1
            if saving_counter % saving_rate == 0:
                da.save(os.path.join(conf['da_dir'], 'model-da-epoch-%02d.pkl' % epoch))
        
        # Keep track of the best alc computed sor far
        if alc_rate != 0:
            alc_counter += 1
            if alc_counter % alc_rate == 0:
                alc = utils.lookup_alc(data, da.function())
                last_alc_epoch = epoch
                print '... training epoch %d, computed alc' % epoch, alc
                if best_alc_value < alc:
                    best_alc_value = alc
                    best_alc_epoch = epoch

    # Compute the ALC at the end of the training if requested
    if conf.get('resulting_alc', False) and last_alc_epoch != epoch:
        alc = utils.lookup_alc(data, da.function())
        print '... training epoch %d, computed alc' % epoch, alc
        if best_alc_value < alc:
            best_alc_value = alc
            best_alc_epoch = epoch

    end_time = time.clock()
    conf['training_time'] = (end_time - start_time) / 60.
    print '... training ended after %f min' % conf['training_time']

    # Compute denoising error for valid and train datasets.
    error_fn = theano.function([minibatch], mycost, name='error_fn')

    conf['error_valid'] = error_fn(data[1].get_value()).item()
    conf['error_test'] = error_fn(data[2].get_value()).item()
    print '... final denoising error with valid is', conf['error_valid']
    print '... final denoising error with test  is', conf['error_test']
    
    # Record the best alc obtained
    if alc_rate != 0:
        print '... best alc at epoch %d, with a value of' \
            % best_alc_epoch, best_alc_value
        conf['best_alc_value'] = best_alc_value
        conf['best_alc_epoch'] = best_alc_epoch
        
    # Save model parameters
    da.save(os.path.join(conf['da_dir'], 'model-da-final.pkl'))
    print '... model has been saved into %s as model-da-final.pkl' % conf['da_dir']

    # Return the learned transformation function
    return da.function('da_transform_fn')


def train_pca(conf, data):
    """Simple wrapper to either load a PCA or train it and save its parameters"""
    pca_model_file = os.path.join(conf['pca_dir'], 'model-pca.pkl')
    if os.path.isfile(pca_model_file):
        # Load a pretrained model.
        print '... loading precomputed PCA transform'
        pca = PCA.load(pca_model_file)
    else:
        # Train the model.
        print '... computing PCA transform'
        pca = PCA(conf['num_components'], conf['min_variance'], conf['whiten'])
        pca.train(data.get_value())
        pca.save(pca_model_file)

    # Return the learned transformation function
    return pca

if __name__ == "__main__":
    conf = {# DA specific arguments
            'corruption_level': 0.1,
            'nhid': 200,
            #'n_vis': 15, # Determined by the datasize
            'anneal_start': 100,
            'base_lr': 0.001,
            'tied_weights': True,
            'act_enc': 'sigmoid',
            'act_dec': None,
            #'lr_hb': 0.10,
            #'lr_vb': 0.10,
            'irange': 0.001,
            'cost_class' : 'MeanSquaredError',
            'corruption_class' : 'BinomialCorruptor',
            # Experiment specific arguments
            'dataset' : 'avicenna',
            'expname' : 'dummy', # Used to create the submission file
            'batchsize' : 20,
            'epochs' : 5,
            'proba' : [1,2,2],
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'saving_rate': 2, # (Default = 0)
            'alc_rate' : 2, # (Default = 0)
            'resulting_alc' : True, # (Default = False)
            'da_dir' : './outputs/',
            'pca_dir' : './outputs/',
            'submit_dir' : './outputs/',
            # Arguments for PCA
            'num_components': 75,
            'min_variance': 0.0, # (Default = 0)
            'whiten': True # (Default = False)
            }

    data = utils.load_data(conf)

    # Blend data subsets.
    data_blended = utils.blend(data, conf['proba'])
    
    # Train (or load a pretrained) PCA transform.
    pca = train_pca(conf, data_blended)
    pca_fn = pca.function('pca_transform_fn')
    del data_blended
    data_after_pca = [utils.sharedX(pca_fn(set.get_value()), borrow=True)
                      for set in data]
    
    # Train a DA over the computed representation
    da_fn = train_da(conf, data_after_pca)
    del data_after_pca
    da = DenoisingAutoencoder.load(os.path.join(conf['da_dir'], 'model-da-epoch-01.pkl'))
    da_fn = da.function('da_transform_fn')
    
    # Stack both layers and create submission file
    input = tensor.matrix()
    transform = theano.function([input], da(pca(input)))
    utils.create_submission(conf, transform)
