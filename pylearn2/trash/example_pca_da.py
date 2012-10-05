"""
IG 2012-09-28: This script is broken and I'm pretty sure doesn't
relate to how the library works anymore.

An example of how to use the library so far."""
# Standard library imports
import sys
import time
import os
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
from pylearn2.corruption import GaussianCorruptor
from pylearn2.autoencoder import DenoisingAutoencoder,build_stacked_ae
from pylearn2.optimizer import SGDOptimizer
from auc import embed
import pylearn2.corruption
from pylearn2 import utils
from pylearn2.pca import PCA, CovEigPCA
from pylearn2.utils import BatchIterator



def create_pca(conf, layer, data, model=None):
    """
    Simple wrapper to either load a PCA or train it and save its parameters
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    clsname = layer.get('pca_class', 'CovEigPCA')

    # Guess the filename
    if model is not None:
        if model.endswith('.pkl'):
            filename = os.path.join(savedir, model)
        else:
            filename = os.path.join(savedir, model + '.pkl')
    else:
        filename = os.path.join(savedir, layer['name'] + '.pkl')

    print 'File name : ',filename
    # Try to load the model
    if model is not None:
        print '... loading layer:', clsname
        try:
            return PCA.load(filename)
        except Exception, e:
            print 'Warning: error while loading %s:' % clsname, e.args[0]
            print 'Switching back to training mode.'

    # Train the model
    print '... training layer:', clsname
    MyPCA = pylearn2.pca.get(clsname)
    pca = MyPCA.fromdict(layer)

    proba = utils.getboth(layer, conf, 'proba')
    blended = utils.blend(data, proba)
    pca.train(blended.get_value(borrow=True))

    pca.save(filename)
    return pca

def main_train(epochs, batchsize, solution='',sparse_penalty=0,sparsityTarget=0,sparsityTargetPenalty=0):

    # Experiment specific arguments
    conf_dataset = {'dataset' : 'avicenna',
                    'expname' : 'dummy', # Used to create the submission file
                    'transfer' : True,
                    'normalize' : True, # (Default = True)
                    'normalize_on_the_fly' : False, # (Default = False)
                    'randomize_valid' : True, # (Default = True)
                    'randomize_test' : True, # (Default = True)
                    'saving_rate': 0, # (Default = 0)
                    'savedir' : './outputs',
                   }

    # First layer = PCA-75 whiten
    pca_layer = {'name' : '1st-PCA',
                 'num_components': 75,
                 'min_variance': -50,
                 'whiten': True,
                 'pca_class' : 'CovEigPCA',
                 # Training properties
                 'proba' : [1, 0, 0],
                 'savedir' : './outputs',
                }


    # Load the dataset
    data = utils.load_data(conf_dataset)

    if conf_dataset['transfer']:
    # Data for the ALC proxy
        label = data[3]
        data = data[:3]



    # First layer : train or load a PCA
    pca = create_pca(conf_dataset, pca_layer, data, model=pca_layer['name'])
    data = [utils.sharedX(pca.function()(set.get_value(borrow=True)),borrow=True) for set in data]
    '''
    if conf_dataset['transfer']:
        data_train, label_train = utils.filter_labels(data[0], label)

        alc = embed.score(data_train, label_train)
        print '... resulting ALC on train (for PCA) is', alc
    '''


    nvis = utils.get_constant(data[0].shape[1]).item()

    conf = {
        'corruption_level': 0.1,
        'nhid': 200,
        'nvis': nvis,
        'anneal_start': 100,
        'base_lr': 0.001,
        'tied_weights': True,
        'act_enc': 'sigmoid',
        'act_dec': None,
        #'lr_hb': 0.10,
        #'lr_vb': 0.10,
        'tied_weights': True ,
        'solution': solution,
        'sparse_penalty': sparse_penalty,
        'sparsityTarget': sparsityTarget ,
        'sparsityTargetPenalty': sparsityTargetPenalty,
        'irange': 0,
    }

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Allocate a denoising autoencoder with binomial noise corruption.
    corruptor = GaussianCorruptor(conf['corruption_level'])
    da = DenoisingAutoencoder(corruptor, conf['nvis'], conf['nhid'],
                              conf['act_enc'], conf['act_dec'], conf['tied_weights'], conf['solution'], conf['sparse_penalty'],
                              conf['sparsityTarget'], conf['sparsityTargetPenalty'])

    # Allocate an optimizer, which tells us how to update our model.
    # TODO: build the cost another way
    cost = SquaredError(da)(minibatch, da.reconstruct(minibatch)).mean()
    trainer = SGDOptimizer(da, conf['base_lr'], conf['anneal_start'])
    updates = trainer.cost_updates(cost)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], cost, updates=updates)

    # Suppose we want minibatches of size 10
    proba = utils.getboth(conf, pca_layer, 'proba')
    iterator = BatchIterator(data, proba, batchsize)

    # Here's a manual training loop. I hope to have some classes that
    # automate this a litle bit.
    final_cost = 0
    for epoch in xrange(epochs):
        c = []
        for minibatch_data in iterator:
            minibatch_err = train_fn(minibatch_data)
            c.append(minibatch_err)
        final_cost = numpy.mean(c)
        print "epoch %d, cost : %f" % (epoch , final_cost)

    print '############################## Fin de l\'experience ############################'
    print 'Calcul de l\'ALC : '
    if conf_dataset['transfer']:
        data_train, label_train = utils.filter_labels(data[0], label)
        alc = embed.score(data_train, label_train)

        print 'Solution : ',solution
        print 'sparse_penalty = ',sparse_penalty
        print 'sparsityTarget = ',sparsityTarget
        print 'sparsityTargetPenalty = ',sparsityTargetPenalty
        print 'Final denoising error is : ',final_cost
        print '... resulting ALC on train is', alc
        return (alc,final_cost)
if __name__ == "__main__":
    main_train(2, 20)
