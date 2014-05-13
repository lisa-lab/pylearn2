#!/usr/bin/env python
from __future__ import print_function

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
"""

Usage: python show_reconstructions <path_to_a_saved_DBM.pkl>
Displays a batch of data from the DBM's training set.
Then shows how the DBM reconstructs it if you run mean field
to estimate the hidden units, then do one mean field downward
pass from hidden_layers[0] to the visible layer.
"""
from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from theano.compat.six.moves import input, xrange
from theano import function


class ReconsViewer(object):
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
        self.rows = rows
        self.cols = cols

        m = self.rows * self.cols
        self.vis_batch = self.dataset.get_batch_topo(m)
        _, patch_rows, patch_cols, channels = self.vis_batch.shape

        assert _ == m

        self.mapback = hasattr(self.dataset, 'mapback_for_viewer')

        actual_cols = 2 * self.cols * (1 + self.mapback) * (1 + (channels == 2))
        self.pv = PatchViewer((self.rows, actual_cols),
                              (patch_rows, patch_cols),
                              is_color=(channels == 3))

        self.batch = model.visible_layer.space.make_theano_batch()
        self.topo = self.batch.ndim > 2
        reconstruction = model.reconstruct(self.batch)
        self.recons_func = function([self.batch], reconstruction)

        if hasattr(model.visible_layer, 'beta'):
            beta = model.visible_layer.beta.get_value()
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
            design_batch = self.dataset.get_design_matrix(design_batch.copy())
        mapped_design = self.dataset.mapback(design_batch.copy())
        mapped_batch = self.dataset.get_topological_view(mapped_design.copy())

        return mapped_batch

    def update_viewer(self):
        """
        Method to update the viewer with a new visible batch.
        """
        ipt = self.vis_batch.copy()
        if not self.topo:
            ipt = self.dataset.get_design_matrix(ipt)
        recons_batch = self.recons_func(ipt.astype(self.batch.dtype))
        if not self.topo:
            recons_batch = self.dataset.get_topological_view(recons_batch)
        if self.mapback:
            mapped_batch = self.get_mapped_batch(self.vis_batch)
            mapped_r_batch = self.get_mapped_batch(recons_batch.copy())
        for row in xrange(self.rows):
            row_start = self.cols * row
            for j in xrange(self.cols):
                vis_patch = self.vis_batch[row_start+j, :, :, :].copy()
                adjusted_vis_patch = self.dataset.adjust_for_viewer(vis_patch)
                if vis_patch.shape[-1] == 2:
                    self.pv.add_patch(adjusted_vis_patch[:, :, 1],
                                      rescale=False)
                    self.pv.add_patch(adjusted_vis_patch[:, :, 0],
                                      rescale=False)
                else:
                    self.pv.add_patch(adjusted_vis_patch, rescale=False)
                r = vis_patch

                if self.mapback:
                    self.pv.add_patch(dataset.adjust_for_viewer(
                        mapped_batch[row_start+j, :, :, :].copy()),
                        rescale=False)
                if recons_batch.shape[-1] == 2:
                    self.pv.add_patch(self.dataset.adjust_to_be_viewed_with(
                        recons_batch[row_start+j, :, :, 1].copy(), vis_patch),
                        rescale=False)
                    self.pv.add_patch(self.dataset.adjust_to_be_viewed_with(
                        recons_batch[row_start+j, :, :, 0].copy(), vis_patch),
                        rescale=False)
                else:
                    self.pv.add_patch(self.dataset.adjust_to_be_viewed_with(
                        recons_batch[row_start+j, :, :, :].copy(), vis_patch),
                        rescale=False)
                r = recons_batch[row_start+j, :, :, :]

                if self.mapback:
                    self.pv.add_patch(self.dataset.adjust_to_be_viewed_with(
                        mapped_r_batch[row_start+j, :, :, :].copy(),
                        mapped_batch[row_start+j, :, :, :].copy()),
                        rescale=False)


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


def load_dataset(dataset_yml, use_test_set):
    """
    Load the dataset used by the model.

    Parameters
    ----------
    dataset_yml: str
        Yaml description of the dataset.
    """

    print('Loading data...')
    dataset = yaml_parse.load(dataset_yml)

    if use_test_set == 'y':
        dataset = dataset.get_test_set()
    else:
        assert use_test_set == 'n'

    return dataset

if __name__ == '__main__':
    rows = 5
    cols = 10
    m = rows * cols
    _, model_path = sys.argv

    model = load_model(model_path, m)

    x = raw_input('use test set? (y/n) ')
    dataset = load_dataset(model.dataset_yaml_src, x)

    recons_viewer = ReconsViewer(model, dataset, rows, cols)

    while True:
        recons_viewer.update_viewer()
        recons_viewer.pv.show()
        print('Displaying reconstructions. (q to quit, ENTER = show more)')
        while True:
            x = input()
            if x == 'q':
                quit()
            if x == '':
                x = 1
                break
            else:
                print('Invalid input, try again')

        recons_viewer.vis_batch = dataset.get_batch_topo(m)
