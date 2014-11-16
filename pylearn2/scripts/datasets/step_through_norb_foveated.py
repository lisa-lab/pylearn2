from __future__ import print_function

__author__ = "Ian Goodfellow"
"""
A script for sequentially stepping through FoveatedNORB, viewing each image
and its label.
"""
import numpy as np

from theano.compat.six.moves import input

from pylearn2.datasets.norb_small import FoveatedNORB
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.utils import get_choice

print('Use test set?')
choices = {'y': 'test', 'n': 'train'}
which_set = choices[get_choice(choices)]

dataset = FoveatedNORB(which_set=which_set, center=True)

topo = dataset.get_topological_view()

b, r, c, ch = topo.shape

assert ch == 2

pv = PatchViewer((1, 2), (r, c), is_color=False)

i = 0
while True:
    patch = topo[i, :, :, :]
    patch = patch / np.abs(patch).max()

    pv.add_patch(patch[:, :, 1], rescale=False)
    pv.add_patch(patch[:, :, 0], rescale=False)

    pv.show()

    print(dataset.y[i])

    choices = {'g': 'goto image', 'q': 'quit'}

    if i + 1 < b:
        choices['n'] = 'next image'

    choice = get_choice(choices)

    if choice == 'q':
        quit()

    if choice == 'n':
        i += 1

    if choice == 'g':
        i = int(input('index: '))
