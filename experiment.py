"""An example of experiment made with the new library."""
# Standard library imports
import time
import os.path

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from framework import utils
from framework import cost
from framework import corruption
from framework.utils import BatchIterator
from framework.autoencoder import DenoisingAutoencoder
from framework.optimizer import SGDOptimizer

def train_da(conf):
    """
    This function basically train a denoising autoencoder according
    to the parameters in conf, and save the learned model
    """
    # Load the dataset
    print '... loading data'
    data = utils.load_data(conf)
    conf['n_vis'] = utils.get_constant(data[0].shape[1])

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Allocate a denoising autoencoder with a given noise corruption.
    corruptor = corruption.get(conf['corruption_class'])(conf)
    da = DenoisingAutoencoder(conf, corruptor)

    # Allocate an optimizer, which tells us how to update our model.
    cost_fn = cost.get(conf['cost_class'])(conf, da)([minibatch])
    trainer = SGDOptimizer(conf, da, cost_fn)
    train_fn = trainer.function([minibatch], name='train_fn')

    # Here's a manual training loop.
    print '... training model'
    start_time = time.clock()
    batch_time = start_time
    batchiter = BatchIterator(conf, data)
    saving_counter = 0
    saving_rate = conf.get('saving_rate',0)
    for epoch in xrange(conf['epochs']):
        c = []
        for minibatch_data in batchiter:
            c.append(train_fn(minibatch_data))

        # Saving intermediate models
        if saving_rate != 0:
            saving_counter += 1
            if saving_counter % saving_rate == 0:
                da.save(conf['saving_dir'], 'model-epoch-%02d.pkl' % epoch)
                
        # Print training time + cost
        train_time = time.clock() - batch_time
        batch_time += train_time
        print '... training epoch %d, time spent (min) %f, cost' \
            % (epoch, train_time / 60.), numpy.mean(c)

    end_time = time.clock()
    conf['training_time'] = (end_time - start_time) / 60.
    print '... training ended after %f min' % conf['training_time']

    # Compute denoising error for valid and train datasets.
    error_fn = theano.function([minibatch], cost_fn, name='error_fn')

    conf['error_valid'] = error_fn(data[1].value)
    conf['error_test'] = error_fn(data[2].value)
    print '... final denoising error with valid is', conf['error_valid']
    print '... final denoising error with test  is', conf['error_test']

    # Save model parameters
    da.save(conf['saving_dir'], 'model-final.pkl')
    print '... model has been saved into %smodel.pkl' % conf['saving_dir']

def submit(conf):
    """
    This function create a submission file with a
    model trained according to conf parameters
    """
    # Load the model parameters
    save_file = os.path.join(conf['saving_dir'], 'model-final.pkl')
    da = DenoisingAutoencoder.load(save_file)

    # Create transformation function
    minibatch = tensor.matrix()
    transform_fn = theano.function([minibatch],
                                   da([minibatch])[0],
                                   name='transform_fn')
    def _get_repr(x):
        y = utils.pre_process(conf, x)
        z = utils.post_process(conf, transform_fn(y))
        return z
    
    # Create submission file
    utils.create_submission(conf, _get_repr)


if __name__ == "__main__":
    conf = {# DA specific arguments
            'corruption_level': 0.3,
            'n_hid': 500,
            #'n_vis': 15, # Determined by the datasize
            'lr_anneal_start': 100,
            'base_lr': 0.01,
            'tied_weights': True,
            'act_enc': 'sigmoid',
            'act_dec': None,
            #'lr_hb': 0.10,
            #'lr_vb': 0.10,
            'irange': 0.001,
            'cost_class' : 'MeanSquaredError',
            'corruption_class' : 'GaussianCorruptor',
            # Experiment specific arguments
            'dataset' : 'avicenna',
            'expname' : 'myfirstexp',
            'batch_size' : 20,
            'epochs' : 5,
            'train_prop' : 1,
            'valid_prop' : 0,
            'test_prop' : 0,
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'saving_rate': 2, # (Default = 0)
            'saving_dir' : './outputs/',
            'submit_dir' : './outputs/',
            'compute_alc' : False, # (Default = False)
            # Pre-processing
            'prep_pca' : 0, # (Default = 0)
            # Post-processing
            'post_pca' : 1, # (Default = 0)
            }

    #train_da(conf)
    submit(conf)
