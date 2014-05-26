#!/usr/bin/env python

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
from pylearn2.gui.patch_viewer import PatchViewer

ignore, model_path = sys.argv
model = serial.load(model_path)

control.push_load_data(False)

dataset = yaml_parse.load(model.dataset_yaml_src)

try:
    layer_to_chains = model.layer_to_chains
except AttributeError:
    print "This model doesn't have negative chains."
    quit(-1)

vis_chains = layer_to_chains[model.visible_layer]
vis_chains = vis_chains.get_value()
if vis_chains.ndim == 2:
    vis_chains = dataset.get_topological_view(vis_chains)
vis_chains = dataset.adjust_for_viewer(vis_chains)

m = vis_chains.shape[0]
r = int(np.sqrt(m))
c = m // r
while r * c < m:
    c += 1

pv = PatchViewer((r,c), vis_chains.shape[1:3],
        is_color = vis_chains.shape[-1] == 3)

for i in xrange(m):
    pv.add_patch(vis_chains[i,:], rescale = False)

pv.show()
