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

from auc import embed
import framework.cost
import framework.corruption

from framework import utils
from framework.pca import PCA, CovEigPCA
from framework.utils import BatchIterator
from framework.base import StackedBlocks
from framework.autoencoder import Autoencoder
from framework.autoencoder import DenoisingAutoencoder, ContractingAutoencoder
from framework.rbm import GaussianBinaryRBM, PersistentCDSampler
from framework.optimizer import SGDOptimizer


def create_pca(conf, layer, data, model=None):
    """
    Simple wrapper to either load a PCA or train it and save its parameters
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    if model is not None:
        # Load the model
        if not model.endswith('.pkl'):
            model += '.pkl'
        try:
            print '... loading PCA layer'
            filename = os.path.join(savedir, model)
            return PCA.load(filename)
        except Exception, e:
            print 'Warning: error while loading PCA.', e.args[0]
            print 'Switching back to training mode.'

    # Train the model
    print '... computing PCA layer'
    MyPCA = framework.pca.get(layer.get('pca_class', 'CovEigPCA'))
    pca = MyPCA.fromdict(layer)

    proba = utils.getboth(layer, conf, 'proba')
    blended = utils.blend(data, proba)
    pca.train(blended.get_value(borrow=True))

    filename = os.path.join(savedir, layer['name'] + '.pkl')
    pca.save(filename)
    return pca


def create_ae(conf, layer, data, model=None):
    """
    This function basically train an autoencoder according
    to the parameters in conf, and save the learned model
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    if model is not None:
        # Load the model instead of training
        print '... loading AE layer'
        if not model.endswith('.pkl'):
            model += '.pkl'
        try:
            filename = os.path.join(savedir, model)
            return Autoencoder.load(filename)
        except Exception, e:
            print 'Warning: error while loading %s:' % layer['autoenc_class'],
            print e.args[0],'\nSwitching back to training mode.'

    # Set visible units size
    layer['nvis'] = utils.get_constant(data[0].shape[1]).item()

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Retrieve the corruptor object (if needed)
    name = layer.get('corruption_class', 'DummyCorruptor')
    MyCorruptor = framework.corruption.get(name)
    corruptor = MyCorruptor(layer.get('corruption_level', 0))

    # Allocate an denoising or contracting autoencoder
    MyAutoencoder = framework.autoencoder.get(layer['autoenc_class'])
    ae = MyAutoencoder.fromdict(layer, corruptor=corruptor)

    # Allocate an optimizer, which tells us how to update our model.
    MyCost = framework.cost.get(layer['cost_class'])
    varcost = MyCost(ae)(minibatch, ae.reconstruct(minibatch))
    if isinstance(ae, ContractingAutoencoder):
        alpha = layer.get('contracting_penalty', 1.)
        varcost += alpha * ae.contraction_penalty(minibatch)
    trainer = SGDOptimizer(ae, layer['base_lr'], layer['anneal_start'])
    updates = trainer.cost_updates(varcost)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], varcost,
                               updates=updates,
                               name='train_fn')

    # Here's a manual training loop.
    print '... training AE layer'
    start_time = time.clock()
    proba = utils.getboth(layer, conf, 'proba')
    iterator = BatchIterator(data, proba, layer['batch_size'])
    saving_counter = 0
    saving_rate = utils.getboth(layer, conf, 'saving_rate', 0)
    for epoch in xrange(layer['epochs']):
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
                ae.save(os.path.join(savedir,
                                     layer['name'] + '-epoch-%02d.pkl' % epoch))

    end_time = time.clock()
    layer['training_time'] = (end_time - start_time) / 60.
    print '... training ended after %f min' % layer['training_time']

    # Compute denoising error for valid and train datasets.
    error_fn = theano.function([minibatch], varcost, name='error_fn')

    layer['error_valid'] = error_fn(data[1].get_value(borrow=True)).item()
    layer['error_test'] = error_fn(data[2].get_value(borrow=True)).item()
    print '... final denoising error with valid is', layer['error_valid']
    print '... final denoising error with test  is', layer['error_test']

    # Save model parameters
    filename = os.path.join(savedir, layer['name'] + '.pkl')
    ae.save(filename)
    print '... final model has been saved as %s' % filename

    # Return the autoencoder object
    return ae


if __name__ == "__main__":
    # First layer = PCA-75 whiten
    layer1 = {'name' : '1st-PCA',
              'num_components': 75,
              'min_variance': 0,
              'whiten': True,
              'pca_class' : 'CovEigPCA',
              # Training properties
              'proba' : [1, 0, 0],
              'savedir' : './outputs',
              }

    # Second layer = CAE-200
    layer2 = {'name' : '2nd-CAE',
              'nhid': 200,
              'tied_weights': True,
              'act_enc': 'sigmoid',
              'act_dec': None,
              'irange': 0.001,
              'cost_class' : 'MeanSquaredError',
              'autoenc_class': 'ContractingAutoencoder',
              'corruption_class' : 'BinomialCorruptor',
              #'corruption_level' : 0.3, # For DenoisingAutoencoder
              # Training properties
              'base_lr': 0.001,
              'anneal_start': 100,
              'batch_size' : 10,
              'epochs' : 10,
              'proba' : [1, 0, 0],
              }

    # Third layer = PCA-3 no whiten
    layer3 = {'name' : '3st-PCA',
              'num_components': 7,
              'min_variance': 0,
              'whiten': True,
              # Training properties
              'proba' : [0, 1, 0]
              }

    # Experiment specific arguments
    conf = {'dataset' : 'avicenna',
            'expname' : 'dummy', # Used to create the submission file
            'transfer' : True,
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'saving_rate': 0, # (Default = 0)
            'savedir' : './outputs',
            }

    # Load the dataset
    data = utils.load_data(conf)

    if conf['transfer']:
        # Data for the ALC proxy
        label = data[3]
        data = data[:3]

    # First layer : train or load a PCA
    pca1 = create_pca(conf, layer1, data, model=layer1['name'])

    data = [utils.sharedX(pca1.function()(set.get_value(borrow=True)),
                          borrow=True) for set in data]

    # Second layer : train or load a DAE or CAE
    ae = create_ae(conf, layer2, data, model=layer2['name'])

    data = [utils.sharedX(ae.function()(set.get_value(borrow=True)),
                          borrow=True) for set in data]

    # Third layer : train or load a PCA
    pca2 = create_pca(conf, layer3, data, model=layer3['name'])

    data = [utils.sharedX(pca2.function()(set.get_value(borrow=True)),
                          borrow=True) for set in data]

    # Compute the ALC for example with labels
    if conf['transfer']:
        data_train, label_train = utils.filter_labels(data[0], label)
        alc = embed.score(data_train, label_train)
        print '... resulting ALC on train is', alc
        conf['train_alc'] = alc

    # Stack both layers and create submission file
    block = StackedBlocks([pca1, ae, pca2])
    utils.create_submission(conf, block.function())
