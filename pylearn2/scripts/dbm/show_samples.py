#!/usr/bin/env python
"""
Usage: python show_samples <path_to_a_saved_DBM.pkl>
Displays a batch of data from the DBM's training set.
Then interactively allows the user to run Gibbs steps
starting from that seed data to see how the DBM's MCMC
sampling changes the data.
"""

from __future__ import print_function

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import numpy as np
import sys
import time
from pylearn2.config import yaml_parse
from pylearn2.expr.basic import is_binary
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.utils import serial
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.six.moves import input, xrange


def init_viewer(dataset, rows, cols):
    """
    Initialisation of the PatchViewer with given rows and columns.

    Parameters
    ----------
    dataset: pylearn2 dataset
    rows: int
    cols: int
    """
    m = rows * cols
    vis_batch = dataset.get_batch_topo(m)

    _, patch_rows, patch_cols, channels = vis_batch.shape

    assert _ == m

    mapback = hasattr(dataset, 'mapback_for_viewer')

    pv = PatchViewer((rows, cols*(1+mapback)),
                     (patch_rows, patch_cols),
                     is_color=(channels == 3))
    return pv


def get_mapped_batch(dataset, design_batch):
    """
    Get mapped batch if 'mapback_for_viewer' is available with the dataset.

    Parameters
    ----------
    dataset: pylearn2 dataset
    design_batch: numpy array
    """
    if design_batch.ndim != 2:
        design_batch = dataset.get_design_matrix(design_batch)
    mapped_batch_design = dataset.mapback_for_viewer(design_batch)
    mapped_batch = dataset.get_topological_view(mapped_batch_design)

    return mapped_batch


def update_viewer(dataset, pv, vis_batch, rows, cols):
    """
    Function to update the viewer with a new visible batch.

    Parameters
    ----------
    dataset: pylearn2 dataset
    pv: PatchViewer
    vis_batch: numpy array
    rows: int
    cols: int
    """
    mapback = hasattr(dataset, 'mapback_for_viewer')
    display_batch = dataset.adjust_for_viewer(vis_batch)
    if display_batch.ndim == 2:
        display_batch = dataset.get_topological_view(display_batch)
    if mapback:
        mapped_batch = get_mapped_batch(dataset, vis_batch)
    for i in xrange(rows):
        row_start = cols * i
        for j in xrange(cols):
            pv.add_patch(display_batch[row_start+j, :, :, :],
                         rescale=False)
            if mapback:
                pv.add_patch(mapped_batch[row_start+j, :, :, :],
                             rescale=False)


def validate_all_samples(model, layer_to_state):
    """
    Validate samples

    Parameters
    ----------
    model: pylearn2 DBM model
    layer_to_state: dict
    """
    # Run some checks on the samples, this should help catch any bugs
    layers = [model.visible_layer] + model.hidden_layers

    def check_batch_size(l):
        if isinstance(l, (list, tuple)):
            map(check_batch_size, l)
        else:
            assert l.get_value().shape[0] == m

    for layer in layers:
        state = layer_to_state[layer]
        space = layer.get_total_state_space()
        space.validate(state)
        if 'DenseMaxPool' in str(type(layer)):
            p, h = state
            p = p.get_value()
            h = h.get_value()
            assert np.all(p == h)
            assert is_binary(p)
        if 'BinaryVisLayer' in str(type(layer)):
            v = state.get_value()
            assert is_binary(v)
        if 'Softmax' in str(type(layer)):
            y = state.get_value()
            assert is_binary(y)
            s = y.sum(axis=1)
            assert np.all(s == 1)
        if 'Ising' in str(type(layer)):
            s = state.get_value()
            assert is_binary((s + 1.) / 2.)


def get_sample_func(model, layer_to_state, x):
    """
    Construct the sample theano function.

    Parameters
    ----------
    model: pylearn2 model
    layer_to_state: dict
    x: int
    """
    theano_rng = MRG_RandomStreams(2012+9+18)

    if x > 0:
        sampling_updates = model.get_sampling_updates(
            layer_to_state,
            theano_rng,
            layer_to_clamp={model.visible_layer: True},
            num_steps=x)

        t1 = time.time()
        sample_func = function([], updates=sampling_updates)
        t2 = time.time()
        print('Clamped sampling function compilation took', t2-t1)
        sample_func()

    # Now compile the full sampling update
    sampling_updates = model.get_sampling_updates(layer_to_state,
                                                  theano_rng)
    assert layer_to_state[model.visible_layer] in sampling_updates

    t1 = time.time()
    sample_func = function([], updates=sampling_updates)
    t2 = time.time()
    print('Sampling function compilation took', t2-t1)

    return sample_func


def load_model(model_path, m):
    """
    Load given model.

    Parameters
    ----------
    model_path: str
        Path of the model to load.
    m: int
        Size of the batch.
    """
    print('Loading model...')
    model = serial.load(model_path)
    model.set_batch_size(m)

    return model


def show_samples(m, model_path):
    """
    Show samples given a DBM model.

    Parameters
    ----------
    m: int
        rows * cols
    model_path: str
        Path of the model.
    """
    model = load_model(model_path, m)

    print('Loading data (used for setting up visualization '
          'and seeding gibbs chain) ...')
    dataset_yaml_src = model.dataset_yaml_src
    dataset = yaml_parse.load(dataset_yaml_src)

    pv = init_viewer(dataset, rows, cols)

    if hasattr(model.visible_layer, 'beta'):
        beta = model.visible_layer.beta.get_value()
        print('beta: ', (beta.min(), beta.mean(), beta.max()))

    print('showing seed data...')
    vis_batch = dataset.get_batch_topo(m)
    update_viewer(dataset, pv, vis_batch, rows, cols)
    pv.show()

    print('How many Gibbs steps should I run with the seed data clamped?'
          '(negative = ignore seed data)')
    x = int(input())

    # Make shared variables representing the sampling state of the model
    layer_to_state = model.make_layer_to_state(m)
    # Seed the sampling with the data batch
    vis_sample = layer_to_state[model.visible_layer]

    validate_all_samples(model, layer_to_state)

    if x >= 0:
        if vis_sample.ndim == 4:
            vis_sample.set_value(vis_batch)
        else:
            design_matrix = dataset.get_design_matrix(vis_batch)
            vis_sample.set_value(design_matrix)

    validate_all_samples(model, layer_to_state)

    sample_func = get_sample_func(model, layer_to_state, x)

    while True:
        print('Displaying samples. '
              'How many steps to take next? (q to quit, ENTER=1)')
        while True:
            x = input()
            if x == 'q':
                quit()
            if x == '':
                x = 1
                break
            else:
                try:
                    x = int(x)
                    break
                except ValueError:
                    print('Invalid input, try again')

        for i in xrange(x):
            print(i)
            sample_func()

        validate_all_samples(model, layer_to_state)

        vis_batch = vis_sample.get_value()
        update_viewer(dataset, pv, vis_batch, rows, cols)
        pv.show()

        if 'Softmax' in str(type(model.hidden_layers[-1])):
            state = layer_to_state[model.hidden_layers[-1]]
            value = state.get_value()
            y = np.argmax(value, axis=1)
            assert y.ndim == 1
            for i in xrange(0, y.shape[0], cols):
                print(y[i:i+cols])


if __name__ == '__main__':
    rows = 10
    cols = 10
    m = rows * cols
    _, model_path = sys.argv

    show_samples(m, model_path)
