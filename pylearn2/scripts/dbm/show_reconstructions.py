#!/bin/env python
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
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
from theano import function

rows = 5
cols = 10
m = rows * cols

_, model_path = sys.argv

print 'Loading model...'
model = serial.load(model_path)
model.set_batch_size(m)


dataset_yaml_src = model.dataset_yaml_src

print 'Loading data...'
dataset = yaml_parse.load(dataset_yaml_src)



"""
sample from greyscale data to see how the model injects color
warnings.warn('hack!')
mean = vis_batch.mean(axis=3)
for ch in xrange(3):
    vis_batch[:,:,:,ch] = mean
"""

vis_batch = dataset.get_batch_topo(m)
_, patch_rows, patch_cols, channels = vis_batch.shape

assert _ == m

mapback = hasattr(dataset, 'mapback_for_viewer')

pv = PatchViewer((rows,2*cols*(1+mapback)), (patch_rows,patch_cols), is_color = (channels==3))

batch = model.visible_layer.space.make_theano_batch()
topo = batch.ndim > 2
reconstruction = model.reconstruct(batch)
recons_func = function([batch], reconstruction)

def show():
    ipt = vis_batch.copy()
    if not topo:
        ipt = dataset.get_design_matrix(ipt)
    recons_batch = recons_func(ipt)
    if not topo:
        recons_batch = dataset.get_topological_view(recons_batch)
    if mapback:
        design_vis_batch = vis_batch
        if design_vis_batch.ndim != 2:
            design_vis_batch = dataset.get_design_matrix(design_vis_batch.copy())
        mapped_batch_design = dataset.mapback(design_vis_batch.copy())
        mapped_batch = dataset.get_topological_view(
                mapped_batch_design.copy())
        design_r_batch = recons_batch.copy()
        if design_r_batch.ndim != 2:
            design_r_batch = dataset.get_design_matrix(design_r_batch.copy())
        mapped_r_design = dataset.mapback(design_r_batch.copy())
        mapped_r_batch = dataset.get_topological_view(mapped_r_design.copy())
    for row in xrange(rows):
        row_start = cols * row
        for j in xrange(cols):
            vis_patch = vis_batch[row_start+j,:,:,:].copy()
            adjusted_vis_patch = dataset.adjust_for_viewer(vis_patch)
            pv.add_patch(dataset.adjust_for_viewer(
                adjusted_vis_patch), rescale = False)
            r = vis_patch
            #print 'vis: '
            #for ch in xrange(3):
            #    chv = r[:,:,ch]
            #    print '\t',ch,(chv.min(),chv.mean(),chv.max())
            if mapback:
                pv.add_patch(dataset.adjust_for_viewer(
                    mapped_batch[row_start+j,:,:,:].copy()), rescale = False)
            pv.add_patch(dataset.adjust_to_be_viewed_with(
                recons_batch[row_start+j,:,:,:].copy(),
                vis_patch), rescale = False)
            r = recons_batch[row_start+j,:,:,:]
            #print 'recons: '
            #for ch in xrange(3):
            #    chv = r[:,:,ch]
            #    print '\t',ch,(chv.min(),chv.mean(),chv.max())
            if mapback:
                pv.add_patch(dataset.adjust_to_be_viewed_with(
                    mapped_r_batch[row_start+j,:,:,:].copy(),
                    mapped_batch[row_start+j,:,:,:].copy()),rescale = False)
    pv.show()


if hasattr(model.visible_layer, 'beta'):
    beta = model.visible_layer.beta.get_value()
    #model.visible_layer.beta.set_value(beta * 100.)
    print 'beta: ',(beta.min(), beta.mean(), beta.max())

while True:
    show()
    print 'Displaying reconstructions. (q to quit, ENTER = show more)'
    while True:
        x = raw_input()
        if x == 'q':
            quit()
        if x == '':
            x = 1
            break
        else:
            print 'Invalid input, try again'

    vis_batch = dataset.get_batch_topo(m)


