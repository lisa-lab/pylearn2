#! /usr/bin/env python

"""
A script for sequentially stepping through SmallNORB, viewing each image and
its label. Intended as a demonstration of how to iterate through NORB images,
and as a way of testing SmallNORB's iterator().
"""

__author__ = "Matthew Koichi Grimes"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = __author__
__license__ = "3-clause BSD"
__maintainer__ = __author__
__email__ = "mkg alum mit edu (@..)"

import argparse, pickle, sys
from matplotlib import pyplot
from pylearn2.datasets.norb import SmallNORB
#from pylearn2.gui.patch_viewer import PatchViewer
# from pylearn2.utils import get_choice

# silences warnings about missing docstrings
#pylint: disable=C0111

def main():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Step-through visualizer for SmallNORB dataset")

        parser.add_argument("--which_set",
                            default='train',
                            required=True,
                            help=("'train', 'test', or the path to a "
                                  "SmallNORB .pkl file"))
        return parser.parse_args(sys.argv[1:])

    def load_norb(args):
        if args.which_set in ('test', 'train'):
            return SmallNORB(args.which_set, True)
        else:
            norb_file = open(args.which_set)
            return pickle.load(norb_file)

    args = parse_args()
    norb = load_norb(args)
    topo_space = norb.get_stereo_data_specs(topo=True, flatten=False)[0]
    topo_images_space = topo_space.components[0]
    vec_images_space = norb.get_data_specs()[0].components[0]

    iterator = norb.iterator(mode='sequential',
                             batch_size=1,
                             data_specs=norb.get_data_specs())


    figure, axes = pyplot.subplots(1, 2, squeeze=True)
    figure.canvas.set_window_title('Small NORB dataset (%sing set)' %
                                   norb.which_set)
    label_text = figure.suptitle("title text",
                                 x=0.1,
                                 horizontalalignment="left")

    def paint_batches(batches):
        if batches is None:
            label_text.set_text("Iteration complete. Press 'q' to quit.")
        else:
            vec_stereo_pair, labels = batches
            topo_stereo_pair = vec_images_space.np_format_as(vec_stereo_pair,
                                                             topo_images_space)

            # print "img shapes: %s, %s, labels: %s" % (topo_stereo_pair[0].shape,
            #                                           topo_stereo_pair[1].shape,
            #                                           labels)

            label_text.set_text("labels: %s" % str(labels))
            for axis, image in zip(axes, topo_stereo_pair):
                axis.imshow(image, cmap='gray')

        figure.canvas.draw()

    def on_key_press(event):
        print 'key pressed: "%s"' % event.key

        if event.key == 'right' or event.key == 'space':
            try:
                paint_batches(iterator.next())
            except StopIteration:
                paint_batches(None)

        if event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    # next() increments iter, but returns the previously pointed-to value.
    paint_batches(iterator.next())
                                 #

    # for datum in norb.iterator(mode='sequential',
    #                            batch_size=1,
    #                            data_specs=norb.get_data_specs()):
    #     vec_stereo_pair, labels = datum
    #     topo_stereo_pair = vec_images_space.np_format_as(vec_stereo_pair,
    #                                                      topo_images_space)

    #     print "img shapes: %s, %s, labels: %s" % (topo_stereo_pair[0].shape,
    #                                               topo_stereo_pair[1].shape,
    #                                               labels)
    #     sys.exit(0)


# print 'Use test set?'
# choices = {'y': 'test', 'n': 'train'}
# which_set = choices[get_choice(choices)]

# dataset = FoveatedNORB(which_set=which_set, center=True)

# topo = dataset.get_topological_view()

# b, r, c, ch = topo.shape

# assert ch == 2

# pv = PatchViewer((1, 2), (r, c), is_color=False)

# i = 0
# while True:
#     patch = topo[i, :, :, :]
#     patch = patch / np.abs(patch).max()

#     pv.add_patch(patch[:,:,1], rescale=False)
#     pv.add_patch(patch[:,:,0], rescale=False)

#     pv.show()

#     print dataset.y[i]

#     choices = {'g': 'goto image', 'q': 'quit'}

#     if i + 1 < b:
#         choices['n'] = 'next image'

#     choice = get_choice(choices)

#     if choice == 'q':
#         quit()

#     if choice == 'n':
#         i += 1

#     if choice == 'g':
#         i = int(raw_input('index: '))

if __name__ == "__main__":
    main()
