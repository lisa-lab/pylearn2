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
    import pylearn2
except ImportError:
    print >> sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

from auc import embed
import pylearn2.cost
import pylearn2.corruption
import pylearn2.rbm

from pylearn2 import utils
from pylearn2.pca import PCA
from pylearn2.base import StackedBlocks
from pylearn2.utils import BatchIterator
from pylearn2.optimizer import SGDOptimizer
from pylearn2.autoencoder import Autoencoder, ContractiveAutoencoder
from pylearn2.rbm import RBM

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

    # Save model parameters
    pca.save(filename)
    return pca

def create_ae(conf, layer, data, model=None):
    """
    This function basically train an autoencoder according
    to the parameters in conf, and save the learned model
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    clsname = layer['autoenc_class']

    # Guess the filename
    if model is not None:
        if model.endswith('.pkl'):
            filename = os.path.join(savedir, model)
        else:
            filename = os.path.join(savedir, model + '.pkl')
    else:
        filename = os.path.join(savedir, layer['name'] + '.pkl')

    # Try to load the model
    if model is not None:
        print '... loading layer:', clsname
        try:
            return Autoencoder.load(filename)
        except Exception, e:
            print 'Warning: error while loading %s:' % clsname, e.args[0]
            print 'Switching back to training mode.'

    # Set visible units size
    layer['nvis'] = utils.get_constant(data[0].shape[1], return_scalar=True)

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Retrieve the corruptor object (if needed)
    name = layer.get('corruption_class', 'DummyCorruptor')
    MyCorruptor = pylearn2.corruption.get(name)
    corruptor = MyCorruptor(layer.get('corruption_level', 0))

    # Allocate an denoising or contracting autoencoder
    MyAutoencoder = pylearn2.autoencoder.get(clsname)
    ae = MyAutoencoder.fromdict(layer, corruptor=corruptor)

    # Allocate an optimizer, which tells us how to update our model.
    MyCost = pylearn2.cost.get(layer['cost_class'])
    varcost = MyCost(ae)(minibatch, ae.reconstruct(minibatch))
    if isinstance(ae, ContractiveAutoencoder):
        alpha = layer.get('contracting_penalty', 0.1)
        penalty = alpha * ae.contraction_penalty(minibatch)
        varcost = varcost + penalty
    varcost = varcost.mean()
    trainer = SGDOptimizer(ae, layer['base_lr'], layer['anneal_start'])
    updates = trainer.cost_updates(varcost)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], varcost,
                               updates=updates,
                               name='train_fn')

    # Here's a manual training loop.
    print '... training layer:', clsname
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
    print '... final error with valid is', layer['error_valid']
    print '... final error with test  is', layer['error_test']

    # Save model parameters
    ae.save(filename)
    print '... final model has been saved as %s' % filename

    # Return the autoencoder object
    return ae

def create_rbm(conf, layer, data, label=None, model=None):
    """
    Loads or trains an RBM.
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    clsname = layer['rbm_class']

    # Guess the filename
    if model is not None:
        if model.endswith('.pkl'):
            filename = os.path.join(savedir, model)
        else:
            filename = os.path.join(savedir, model + '.pkl')
    else:
        filename = os.path.join(savedir, layer['name'] + '.pkl')

    # Try to load the model
    if model is not None:
        print '... trying to load layer:', clsname
        try:
            # This loads and checks that the loaded model is an RBM
            # or subclass
            return RBM.load(filename)
        except Exception, e:
            print 'Warning: error while loading %s from %s:' % (clsname, filename), e.args[0]
            print 'Training it instead.'

    # Set nvis
    layer['nvis'] = utils.get_constant(data[0].shape[1], return_scalar=True)

    # A symbolic minibatch input
    minibatch = tensor.matrix('minibatch')

    # RNG
    rng = numpy.random.RandomState(seed=layer.get('seed', 42))

    # The RBM itself
    RBMClass = pylearn2.rbm.get(clsname)
    rbm = RBMClass.fromdict(layer, rng=rng)

    # Sampler
    SamplerClass = pylearn2.rbm.get_sampler(layer['sampler'])
    sampler_kwargs = {
            'rbm': rbm,
            'particles': data[0].get_value(borrow=True)[0:layer['batch_size']].copy(),
            'rng': rng}
    if 'pcd_steps' in layer:
        sampler_kwargs['steps'] = layer['pcd_steps']
    if 'particles_min' in layer or 'particles_max' in layer:
        sampler_kwargs['particles_clip'] = (
                layer.get('particles_min', -numpy.inf),
                layer.get('particles_max', +numpy.inf))
    sampler = SamplerClass(**sampler_kwargs)

    # Optimizer
    OptimizerClass = pylearn2.optimizer.get(layer['optimizer'])
    if OptimizerClass == SGDOptimizer:
        optimizer_kwargs = {
                'params': rbm,
                'base_lr': layer['base_lr'],
                'anneal_start': layer.get('anneal_start', numpy.inf),
                }

        # convert ..._min and ..._max to ..._clip
        clipped_names = []
        for k in layer.iterkeys():
            if k[-4:] in ('_min', '_max'):
                if k[:-4] != 'particles':
                    clipped_names.append(k[:-4])
        for name in clipped_names:
            optimizer_kwargs['%s_clip' % name] = (
                    layer.get('%s_min' % name, -numpy.inf),
                    layer.get('%s_max' % name, +numpy.inf))

        optimizer = OptimizerClass(**optimizer_kwargs)
    else:
        raise NotImplementedError('Only SGDOptimizer is supported with RBMs '
                'at the moment.', OptimizerClass)


    updates = pylearn2.rbm.training_updates(
            visible_batch=minibatch,
            model=rbm,
            sampler=sampler,
            optimizer=optimizer)

    proxy_cost = rbm.reconstruction_error(minibatch, rng=sampler.s_rng)
    train_fn = theano.function(
            [minibatch],
            proxy_cost,
            updates=updates,
            name='train_fn')

    # Manual training loop,
    # copy/pasted from create_ae
    print '... training layer:', clsname
    start_time = time.clock()
    proba = utils.getboth(layer, conf, 'proba')
    iterator = BatchIterator(data, proba, layer['batch_size'])
    saving_counter = 0
    saving_rate = utils.getboth(layer, conf, 'saving_rate', 0)

    # For ALC
    if label is not None:
        data_train, label_train = utils.filter_labels(data[0], label)

    for epoch in xrange(layer['epochs']):
        c = []
        batch_time = time.clock()
        for minibatch_data in iterator:
            c.append(train_fn(minibatch_data))

        # Print time & cost
        train_time = time.clock() - batch_time
        print '... training epoch %d, time spent (min) %f, cost' % (
                epoch, train_time / 60.), numpy.mean(c)

        # Saving intermediate models
        if saving_rate != 0:
            saving_counter += 1
            if saving_counter % saving_rate == 0:
                rbm.save(os.path.join(savedir,
                    layer['name'] + '-epoch-%02d.pkl' % epoch))

                ## Yes, this is a hack
                if label is not None:
                    # Compute ALC on train
                    data_train_repr = utils.minibatch_map(
                            rbm.function(),
                            layer['batch_size'],
                            data_train,
                            output_width=layer['nhid'])
                    alc = embed.score(data_train_repr, label_train)
                    print '... train ALC at epoch %d: %f' % (epoch, alc)

    end_time = time.clock()
    layer['training_time'] = (end_time - start_time) / 60.
    print '... training ended after %f min' % layer['training_time']

    # Compute reconstruction error for valid and train data sets
    error_fn = theano.function([minibatch], proxy_cost, name='error_fn')
    layer['error_valid'] = error_fn(data[1].get_value(borrow=True)).item()
    layer['error_test'] = error_fn(data[2].get_value(borrow=True)).item()
    print '... final error with valid is', layer['error_valid']
    print '... final error with test  is', layer['error_test']

    # Save model parameters
    rbm.save(filename)
    print '... final model has been saved as %s' % filename

    # Return the RBM object
    return rbm

if __name__ == "__main__":
    # First layer = PCA-75 whiten
    layer1 = {'name': '1st-PCA',
              'num_components': 75,
              'min_variance': 0,
              'whiten': True,
              'pca_class': 'SVDPCA',
              # Training properties
              'proba': [1, 0, 0],
              'savedir': './outputs',
              }

    # Second layer = CAE-200
    layer2 = {'name': '2nd-CAE',
              'nhid': 200,
              'tied_weights': True,
              'act_enc': 'rectifier',
              'act_dec': None,
              'irange': 0.001,
              'cost_class': 'SquaredError',
              'autoenc_class': 'ContractiveAutoencoder',
              'corruption_class': 'BinomialCorruptor',
              'corruption_level': 0.3,  # For DenoisingAutoencoder
              'contracting_penalty': 0.1,  # For ContractingAutoencoder
              # Training properties
              'base_lr': 0.001,
              'anneal_start': 100,
              'batch_size': 1,
              'epochs': 10,
              'proba': [1, 0, 0],
              }

    # Third layer = PCA-3 no whiten
    layer3 = {'name': '3st-PCA',
              'num_components': 7,
              'min_variance': 0,
              'whiten': True,
              'pca_class': 'SVDPCA',
              # Training properties
              'proba': [0, 1, 0]
              }

    # Experiment specific arguments
    conf = {'dataset': 'avicenna',
            'expname': 'dummy',  # Used to create the submission file
            'transfer': True,
            'normalize': True,  # (Default = True)
            'normalize_on_the_fly': False,  # (Default = False)
            'randomize_valid': True,  # (Default = True)
            'randomize_test': True,  # (Default = True)
            'saving_rate': 0,  # (Default = 0)
            'savedir': './outputs',
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
