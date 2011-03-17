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
from framework import utils
from framework import cost
from framework import corruption
from framework import autoencoder
#from posttraitement.pca import PCA
from framework.pca import PCA
from framework.base import StackedBlocks
from framework.utils import BatchIterator
from framework.autoencoder import DenoisingAutoencoder, ContractingAutoencoder
from framework.optimizer import SGDOptimizer


def create_da(conf, layer, data, model=None):
    """
    This function basically train an autoencoder according
    to the parameters in conf, and save the learned model
    """
    # Either DenoisingAutoencoder or ConstrastingAutoencoder
    MyAutoencoder = autoencoder.get(layer['autoenc_class'])
    
    savedir = layer.get('savedir', conf.get('savedir'))
    if model is not None:
        # Load the model instead of training
        print '... loading Autoencoder layer'
        if not model.endswith('.pkl'):
            model += '.pkl'
        filename = os.path.join(savedir, model)
        return MyAutoencoder.load(filename)
    
    # Set visible units size
    layer['nvis'] = utils.get_constant(data[0].shape[1]).item()

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Allocate an denoising or contracting autoencoder
    if MyAutoencoder == DenoisingAutoencoder:
        MyCorruptor = corruption.get(layer['corruption_class'])
        corruptor = MyCorruptor(conf['corruption_level'])
        # TODO : refactor this part of the code
        ae = MyAutoencoder(corruptor, layer['nvis'], layer['nhid'],
                           layer['act_enc'], layer['act_dec'])
    else:
        ae = MyAutoencoder.fromdict(layer)

    # Allocate an optimizer, which tells us how to update our model.
    MyCost = cost.get(layer['cost_class'])
    cost_fn = MyCost(ae)(minibatch, ae.reconstruct(minibatch))
    trainer = SGDOptimizer(ae, layer['base_lr'], layer['anneal_start'])
    updates = trainer.cost_updates(cost_fn)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], cost_fn,
                               updates=updates,
                               name='train_fn')

    # Here's a manual training loop.
    print '... training Autoencoder'
    start_time = time.clock()
    proba = layer.get('proba', conf.get('proba'))
    iterator = BatchIterator(data, proba, layer['batchsize'])
    saving_counter = 0
    saving_rate = layer.get('saving_rate', conf.get('saving_rate', 0))
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
    error_fn = theano.function([minibatch], cost_fn, name='error_fn')

    conf['error_valid'] = error_fn(data[1].get_value()).item()
    conf['error_test'] = error_fn(data[2].get_value()).item()
    print '... final denoising error with valid is', conf['error_valid']
    print '... final denoising error with test  is', conf['error_test']
    
    # Save model parameters
    filename = os.path.join(savedir, layer['name'] + '.pkl')
    ae.save(filename)
    print '... final model has been saved as %s' % filename

    # Return the learned transformation function
    return ae


def create_pca(conf, layer, data, model=None):
    """Simple wrapper to either load a PCA or train it and save its parameters"""
    savedir = layer.get('savedir', conf.get('savedir'))
    if model is not None:
        # Load the model
        print '... loading PCA layer'
        if not model.endswith('.pkl'):
            model += '.pkl'
        filename = os.path.join(savedir, model)
        return PCA.load(filename)
    else:
        # Train the model
        print '... computing PCA layer'
        pca = PCA.fromdict(layer)
        
        proba = layer.get('proba', conf.get('proba'))
        blended = utils.blend(data, proba)
        pca.train(blended.get_value())
        
        filename = os.path.join(savedir, layer['name'] + '.pkl')
        pca.save(filename)
        return pca

if __name__ == "__main__":
    # First layer = PCA-75 whiten
    layer1 = {'name' : '1st-PCA',
              'num_components': 75,
              'min_variance': 0,
              'whiten': True,
              # Training properties
              'proba' : [1,0,0],
              'savedir' : './outputs',
              }
    
    # Second layer = CAE-200
    layer2 = {'name' : '2nd-CAE',
            'nhid': 200,
            'tied_weights': True,
            'act_enc': 'sigmoid',
            'act_dec': None,
            'irange': 0.001,
            'corruption_class' : 'BinomialCorruptor',
            'autoenc_class': 'ContractingAutoencoder',
            'cost_class' : 'MeanSquaredError',
            # Training properties
            'base_lr': 0.001,
            'anneal_start': 100,
            'batchsize' : 20,
            'epochs' : 5,
            'proba' : [1,0,0],
            }
    
    # First layer = PCA-75 whiten
    layer3 = {'name' : '3st-PCA',
              'num_components': 3,
              'min_variance': 0,
              'whiten': False,
              # Training properties
              'proba' : [0,1,0]
              }
    
    # Experiment specific arguments
    conf = {'dataset' : 'avicenna',
            'expname' : 'dummy', # Used to create the submission file
            'transfer' : True,
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'saving_rate': 2, # (Default = 0)
            'alc_rate' : 2, # (Default = 0)
            'resulting_alc' : True, # (Default = False)
            'savedir' : './outputs',
            }

    # Load the dataset
    data = utils.load_data(conf)
    
    if conf['transfer']:
        # Data for the ALC proxy
        alc_train, alc_label = utils.filter_labels(data[0], data[3])
        data = data[:3]

    # First layer : train or load a PCA
    pca = create_pca(conf, layer1, data)
    
    data = [utils.sharedX(pca.function()(set.get_value()), borrow=True)
                      for set in data]
    
    # Second layer : train or load a DAE or CAE
    ae = create_da(conf, layer2, data, model=layer2['name'])
    
    # Compute the ALC for example with labels
    # TODO : Not functionnal yet
    #block = StackedBlocks([pca, ae])
    #alc = embed.score(block.function()(alc_train), alc_label)
    #print '... resulting ALC on train is', alc
    
    # Stack both layers and create submission file
    input = tensor.matrix()
    transform = theano.function([input], ae(pca(input)))
    utils.create_submission(conf, transform)
