#!/usr/bin/env python
from __future__ import print_function

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
"""

Usage: python show_samples <path_to_a_saved_DBM.pkl>
Displays a batch of data from the DBM's training set.
Then interactively allows the user to run Gibbs steps
starting from that seed data to see how the DBM's MCMC
sampling changes the data.

"""

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


class SamplesViewer(object):
    """
    Parameters
    ----------
    model: Model
        The model from which to display the reconstructions.
    dataset: Dataset
        The dataset used to train the model.
    rows: int
        Number of rows to display.
    cols: int
        Number of columns to display.
    """
    def __init__(self, model, dataset, rows, cols):
        self.dataset = dataset
        self.model = model
        self.rows = rows
        self.cols = cols

        m = self.rows * self.cols
        self.vis_batch = self.dataset.get_batch_topo(m)

        _, patch_rows, patch_cols, channels = self.vis_batch.shape

        assert _ == m

        self.mapback = hasattr(self.dataset, 'mapback_for_viewer')

        self.pv = PatchViewer((self.rows, self.cols*(1+self.mapback)),
                              (patch_rows, patch_cols),
                              is_color=(channels == 3))

        if hasattr(self.model.visible_layer, 'beta'):
            beta = self.model.visible_layer.beta.get_value()
            print('beta: ', (beta.min(), beta.mean(), beta.max()))

    def get_mapped_batch(self, batch):
        """
        Parameters
        ----------
        batch: numpy array
            WRITEME
        """
        design_batch = batch
        if design_batch.ndim != 2:
            design_batch = self.dataset.get_design_matrix(design_batch)
        mapped_batch_design = self.dataset.mapback_for_viewer(design_batch)
        mapped_batch = self.dataset.get_topological_view(mapped_batch_design)

        return mapped_batch

    def update_viewer(self):
        """
        Method to update the viewer with a new visible batch.
        """
        display_batch = self.dataset.adjust_for_viewer(self.vis_batch)
        if display_batch.ndim == 2:
            display_batch = self.dataset.get_topological_view(display_batch)
        if self.mapback:
            mapped_batch = get_mapped_batch(self.vis_batch)
        for i in xrange(self.rows):
            row_start = self.cols * i
            for j in xrange(self.cols):
                self.pv.add_patch(display_batch[row_start+j, :, :, :],
                                  rescale=False)
                if self.mapback:
                    self.pv.add_patch(mapped_batch[row_start+j, :, :, :],
                                      rescale=False)

    def validate_all_samples(self):
        # Run some checks on the samples, this should help catch any bugs
        layers = [self.model.visible_layer] + self.model.hidden_layers

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

    def get_sample_func(self, vis_sample):
        if x >= 0:
            if vis_sample.ndim == 4:
                vis_sample.set_value(self.vis_batch)
            else:
                design_matrix = self.dataset.get_design_matrix(self.vis_batch)
                vis_sample.set_value(design_matrix)

        self.validate_all_samples()

        theano_rng = MRG_RandomStreams(2012+9+18)

        if x > 0:
            sampling_updates = self.model.get_sampling_updates(layer_to_state,
                                                               theano_rng,
                                                               layer_to_clamp={self.model.visible_layer: True},
                                                               num_steps=x)

            t1 = time.time()
            sample_func = function([], updates=sampling_updates)
            t2 = time.time()
            print('Clamped sampling function compilation took', t2-t1)
            sample_func()

        # Now compile the full sampling update
        sampling_updates = self.model.get_sampling_updates(layer_to_state,
                                                           theano_rng)
        assert layer_to_state[self.model.visible_layer] in sampling_updates

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


if __name__ == '__main__':
    rows = 10
    cols = 10
    m = rows * cols

    _, model_path = sys.argv

    model = load_model(model_path, m)

    print('Loading data (used for setting up visualization '
          'and seeding gibbs chain) ...')
    dataset_yaml_src = model.dataset_yaml_src
    dataset = yaml_parse.load(dataset_yaml_src)

    samples_viewer = SamplesViewer(model, dataset, rows, cols)

    print('showing seed data...')
    samples_viewer.update_viewer()
    samples_viewer.pv.show()

    print('How many Gibbs steps should I run with the seed data clamped?'
          '(negative = ignore seed data)')
    x = int(input())

    # Make shared variables representing the sampling state of the model
    layer_to_state = model.make_layer_to_state(m)
    # Seed the sampling with the data batch
    vis_sample = layer_to_state[model.visible_layer]

    samples_viewer.validate_all_samples()

    sample_func = samples_viewer.get_sample_func(vis_sample)

    while True:
        print('Displaying samples. How many steps to take next? (q to quit, ENTER=1)')
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

        samples_viewer.validate_all_samples()

        samples_viewer.vis_batch = vis_sample.get_value()
        samples_viewer.update_viewer()
        samples_viewer.pv.show()

        if 'Softmax' in str(type(model.hidden_layers[-1])):
            state = layer_to_state[model.hidden_layers[-1]]
            value = state.get_value()
            y = np.argmax(value, axis=1)
            assert y.ndim == 1
            for i in xrange(0, y.shape[0], cols):
                print(y[i:i+cols])
