#!/usr/bin/env python
"""
Usage: python show_negative_chains.py <path_to_a_saved_DBM.pkl>
Show negative chains of a saved DBM model.
"""

from __future__ import print_function

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import sys
from pylearn2.utils import serial
from pylearn2.datasets import control
from pylearn2.config import yaml_parse
import numpy as np
from theano.compat.six.moves import xrange
from pylearn2.gui.patch_viewer import PatchViewer


def get_grid_shape(m):
    """
    Adjust the shape of the grid to show.

    Parameters
    ----------
    m: int
        Number of visible chains.
    """
    r = int(np.sqrt(m))
    c = m // r
    while r * c < m:
        c += 1

    return (r, c)


def get_vis_chains(layer_to_chains, model, dataset):
    """
    Get visible chains formatted for the path viewer.

    Parameters
    ----------
    layers_to_chains: dict
        Dictionary mapping layers to states.
    model: Model
        The model from which we get the visible chains.
    dataset:
        The dataset used to train the model.
    """
    vis_chains = layer_to_chains[model.visible_layer]
    vis_chains = vis_chains.get_value()
    print(vis_chains.shape)
    if vis_chains.ndim == 2:
        vis_chains = dataset.get_topological_view(vis_chains)
    print(vis_chains.shape)
    vis_chains = dataset.adjust_for_viewer(vis_chains)

    return vis_chains


def create_patch_viewer(grid_shape, vis_chains, m):
    """
    Add the patches to show.

    Parameters
    ----------
    grid_shape: tuple
        The shape of the grid to show.
    vis_chains: numpy array
        Visibles chains.
    m: int
        Number of visible chains.
    """
    pv = PatchViewer(grid_shape, vis_chains.shape[1:3],
                     is_color=vis_chains.shape[-1] == 3)

    for i in xrange(m):
        pv.add_patch(vis_chains[i, :], rescale=False)

    return pv


def show_negative_chains(model_path):
    """
    Display negative chains.

    Parameters
    ----------
    model_path: str
        The path to the model pickle file
    """
    model = serial.load(model_path)

    try:
        control.push_load_data(False)
        dataset = yaml_parse.load(model.dataset_yaml_src)
    finally:
        control.pop_load_data()

    try:
        layer_to_chains = model.layer_to_chains
    except AttributeError:
        print("This model doesn't have negative chains.")
        quit(-1)

    vis_chains = get_vis_chains(layer_to_chains, model, dataset)

    m = vis_chains.shape[0]
    grid_shape = get_grid_shape(m)

    return create_patch_viewer(grid_shape, vis_chains, m)

if __name__ == '__main__':
    ignore, model_path = sys.argv
    pv = show_negative_chains(model_path)
    pv.show()
