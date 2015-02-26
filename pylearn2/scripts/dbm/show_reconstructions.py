#!/usr/bin/env python
"""
Usage: python show_reconstructions <path_to_a_saved_DBM.pkl>
Displays a batch of data from the DBM's training set.
Then shows how the DBM reconstructs it if you run mean field
to estimate the hidden units, then do one mean field downward
pass from hidden_layers[0] to the visible layer.
"""

from __future__ import print_function

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import sys
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.utils import serial
from theano.compat.six.moves import input, xrange
from theano import function


def init_viewer(dataset, rows, cols, vis_batch):
    """
    Initialisation of the PatchViewer with given rows and columns.

    Parameters
    ----------
    dataset: pylearn2 dataset
    rows: int
    cols: int
    vis_batch: numpy array
    """
    m = rows * cols
    _, patch_rows, patch_cols, channels = vis_batch.shape

    assert _ == m

    mapback = hasattr(dataset, 'mapback_for_viewer')
    actual_cols = 2 * cols * (1 + mapback) * (1 + (channels == 2))
    pv = PatchViewer((rows, actual_cols),
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
        design_batch = dataset.get_design_matrix(design_batch.copy())
    mapped_design = dataset.mapback(design_batch.copy())
    mapped_batch = dataset.get_topological_view(mapped_design.copy())

    return mapped_batch


def update_viewer(dataset, batch, rows, cols, pv, recons_func, vis_batch):
    """
    Function to update the viewer with a new visible batch.

    Parameters
    ----------
    dataset: pylearn2 dataset
    batch: numpy array
    rows: int
    cols: int
    pv: PatchViewer
    recons_func: theano function
    vis_batch: numpy array
    """
    mapback = hasattr(dataset, 'mapback_for_viewer')
    topo = batch.ndim > 2

    ipt = vis_batch.copy()
    if not topo:
        ipt = dataset.get_design_matrix(ipt)
    recons_batch = recons_func(ipt.astype(batch.dtype))
    if not topo:
        recons_batch = dataset.get_topological_view(recons_batch)
    if mapback:
        mapped_batch = get_mapped_batch(dataset, vis_batch)
        mapped_r_batch = get_mapped_batch(dataset, recons_batch.copy())
    for row in xrange(rows):
        row_start = cols * row
        for j in xrange(cols):
            vis_patch = vis_batch[row_start+j, :, :, :].copy()
            adjusted_vis_patch = dataset.adjust_for_viewer(vis_patch)
            if vis_patch.shape[-1] == 2:
                pv.add_patch(adjusted_vis_patch[:, :, 1],
                             rescale=False)
                pv.add_patch(adjusted_vis_patch[:, :, 0],
                             rescale=False)
            else:
                pv.add_patch(adjusted_vis_patch, rescale=False)

            if mapback:
                pv.add_patch(
                    dataset.adjust_for_viewer(
                        mapped_batch[row_start+j, :, :, :].copy()),
                    rescale=False)
                pv.add_patch(
                    dataset.adjust_to_be_viewed_with(
                        mapped_r_batch[row_start+j, :, :, :].copy(),
                        mapped_batch[row_start+j, :, :, :].copy()),
                    rescale=False)
            if recons_batch.shape[-1] == 2:
                pv.add_patch(
                    dataset.adjust_to_be_viewed_with(
                        recons_batch[row_start+j, :, :, 1].copy(), vis_patch),
                    rescale=False)
                pv.add_patch(
                    dataset.adjust_to_be_viewed_with(
                        recons_batch[row_start+j, :, :, 0].copy(), vis_patch),
                    rescale=False)
            else:
                pv.add_patch(
                    dataset.adjust_to_be_viewed_with(
                        recons_batch[row_start+j, :, :, :].copy(), vis_patch),
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


def show_reconstructions(m, model_path):
    """
    Show reconstructions of a given DBM model.

    Parameters
    ----------
    m: int
        rows * cols
    model_path: str
        Path of the model.
    """
    model = load_model(model_path, m)

    x = input('use test set? (y/n) ')
    dataset = load_dataset(model.dataset_yaml_src, x)
    vis_batch = dataset.get_batch_topo(m)
    pv = init_viewer(dataset, rows, cols, vis_batch)

    batch = model.visible_layer.space.make_theano_batch()
    reconstruction = model.reconstruct(batch)
    recons_func = function([batch], reconstruction)

    if hasattr(model.visible_layer, 'beta'):
        beta = model.visible_layer.beta.get_value()
        print('beta: ', (beta.min(), beta.mean(), beta.max()))

    while True:
        update_viewer(dataset, batch, rows, cols, pv, recons_func, vis_batch)
        pv.show()
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

        vis_batch = dataset.get_batch_topo(m)


if __name__ == '__main__':
    rows = 5
    cols = 10
    m = rows * cols
    _, model_path = sys.argv

    show_reconstructions(m, model_path)
