"""An example of experiment made with the new library."""
# Standard library imports
import time
import sys

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
from framework import cost
from framework import corruption
from framework.utils import BatchIterator
from framework.autoencoder import DenoisingAutoencoder
from framework.optimizer import SGDOptimizer
from posttraitement.pca import PCA


def train_da(conf, data):
    """
    This function basically train a denoising autoencoder according
    to the parameters in conf, and save the learned model
    """
    # Set visible units size
    conf['n_vis'] = utils.get_constant(data[0].shape[1])

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Allocate a denoising autoencoder with a given noise corruption.
    corruptor = corruption.get(conf['corruption_class'])(conf)
    da = DenoisingAutoencoder(conf, corruptor)

    # Allocate an optimizer, which tells us how to update our model.
    cost_fn = cost.get(conf['cost_class'])(conf, da)([minibatch])
    trainer = SGDOptimizer(conf, da.params(), cost_fn)
    train_fn = trainer.function([minibatch], name='train_fn')

    # Here's a manual training loop.
    print '... training model'
    start_time = time.clock()
    batch_time = start_time
    batchiter = BatchIterator(conf, data)
    saving_counter = 0
    saving_rate = conf.get('saving_rate',0)
    alc_counter = 0
    alc_rate = conf.get('alc_rate', 0)
    best_alc_value = 0
    best_alc_epoch = -1
    last_alc_epoch = -1
    for epoch in xrange(conf['epochs']):
        c = []
        for minibatch_data in batchiter:
            c.append(train_fn(minibatch_data))

        # Print training time + cost
        train_time = time.clock() - batch_time
        batch_time += train_time
        print '... training epoch %d, time spent (min) %f, cost' \
            % (epoch, train_time / 60.), numpy.mean(c)

        # Saving intermediate models
        if saving_rate != 0:
            saving_counter += 1
            if saving_counter % saving_rate == 0:
                da.save(conf['models_dir'], 'model-da-epoch-%02d.pkl' % epoch)
        
        # Keep track of the best alc computed
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
    error_fn = theano.function([minibatch], cost_fn, name='error_fn')

    conf['error_valid'] = error_fn(data[1].value)
    conf['error_test'] = error_fn(data[2].value)
    print '... final denoising error with valid is', conf['error_valid']
    print '... final denoising error with test  is', conf['error_test']
    
    # Record the best alc obtained
    if alc_rate != 0:
        print '... best alc at epoch %d, with a value of' \
            % best_alc_epoch, best_alc_value
        conf['best_alc_value'] = best_alc_value
        conf['best_alc_epoch'] = best_alc_epoch
        
    # Save model parameters
    da.save(conf['models_dir'], 'model-da-final.pkl')
    print '... model has been saved into %s as model-da-final.pkl' % conf['models_dir']

    # Return the learned transformation function
    return da.function('da_transform_fn')


def train_pca(conf, dataset):
    """Simple wraper to train a PCA and save its parameters"""
    # Train the model
    print '... training PCA'
    pca = PCA(conf)
    pca.train(dataset.get_value())
    
    print '... saving PCA'
    pca.save(conf['models_dir'], 'model-pca.pkl')

    # Return the learned transformation function
    return pca.function('pca_transform_fn')


if __name__ == "__main__":
    conf = {# DA specific arguments
            'corruption_level': 0.5,
            'n_hid': 600,
            #'n_vis': 15, # Determined by the datasize
            'lr_anneal_start': 100,
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
            'expname' : 'dummy',
            'batch_size' : 20,
            'epochs' : 10,
            'train_prop' : 1,
            'valid_prop' : 2,
            'test_prop' : 2,
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'saving_rate': 2, # (Default = 0)
            'alc_rate' : 2, # (Default = 0)
            'resulting_alc' : True, # (Default = False)
            'models_dir' : './outputs/',
            'submit_dir' : './outputs/',
            # Arguments for PCA
            'num_components': 75,
            'min_variance': 0, # (Default = 0)
            'whiten': True # (Default = False)
            }

    data = utils.load_data(conf)
    
    # Train a PCA
    data_blended = utils.blend(conf, data)
    pca_fn = train_pca(conf, data_blended)
    del data_blended
    pca = PCA.load(conf['models_dir'], 'model-pca.pkl')
    pca_fn = pca.function('pca_transform_fn')
    
    data_after_pca = [utils.sharedX(pca_fn(set.get_value()))
                      for set in data]
    
    # Train a DA over the computed representation
    da_fn = train_da(conf, data_after_pca)
    del data_after_pca
    da = DenoisingAutoencoder.load(conf['models_dir'], 'model-da-final.pkl')
    da_fn = da.function('da_transform_fn')
    
    # Stack both layers and create submission file
    input = tensor.matrix()
    transform = theano.function([input], da(pca(input)))
    utils.create_submission(conf, transform)
