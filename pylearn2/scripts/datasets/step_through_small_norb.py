#! /usr/bin/env python

"""
A script for sequentially stepping through SmallNORB, viewing each image and
its label.

Intended as a demonstration of how to iterate through NORB images,
and as a way of testing SmallNORB's StereoViewConverter.

If you just want an image viewer, consider
pylearn2/scripts/show_binocular_grayscale_images.py,
which is not specific to SmallNORB.
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
from pylearn2.utils import safe_zip


def main():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Step-through visualizer for SmallNORB dataset")

        parser.add_argument("--which_set",
                            default='train',
                            required=True,
                            help=("'train', 'test', or the path to a "
                                  "SmallNORB .pkl file"))
        return parser.parse_args()

    def load_norb(args):
        if args.which_set in ('test', 'train'):
            return SmallNORB(args.which_set, True)
        else:
            norb_file = open(args.which_set)
            return pickle.load(norb_file)

    args = parse_args()
    norb = load_norb(args)
    topo_space = norb.view_converter.topo_space  # does not include label space
    vec_space = norb.get_data_specs()[0].components[0]

    figure, axes = pyplot.subplots(1, 2, squeeze=True)
    figure.suptitle("Press space to step through, or 'q' to quit.")

    def draw_and_increment(iterator):
        """
        Draws the image pair currently pointed at by the iterator,
        then increments the iterator.
        """

        def draw(batch_pair):
            for axis, image_batch in safe_zip(axes, batch_pair):
                assert image_batch.shape[0] == 1
                grayscale_image = image_batch[0, :, :, 0]
                axis.imshow(grayscale_image, cmap='gray')

            figure.canvas.draw()

        def get_values_and_increment(iterator):
            try:
                vec_stereo_pair, labels = norb_iter.next()
            except StopIteration:
                return (None, None)

            topo_stereo_pair = vec_space.np_format_as(vec_stereo_pair,
                                                      topo_space)
            return topo_stereo_pair, labels

        batch_pair, labels = get_values_and_increment(norb_iter)
        draw(batch_pair)

    norb_iter = norb.iterator(mode='sequential',
                              batch_size=1,
                              data_specs=norb.get_data_specs())

    def on_key_press(event):
        if event.key == ' ':
            draw_and_increment(norb_iter)
        if event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)
    draw_and_increment(norb_iter)
    pyplot.show()


if __name__ == "__main__":
    main()
