#!/usr/bin/env python

import argparse
import pickle
import sys
import numpy as np
from matplotlib import pyplot
from pylearn2.datasets import norb


class SmallNorbBrowser():

    def __init__(self, instance_index, current_labels, current_label_type,
                 label_text, figure, axes, num_label_types):
        self.instance_index = instance_index
        self.current_labels = current_labels
        self.new_to_old_instance = []
        self.current_label_type = current_label_type
        self.label_text = label_text
        self.label_to_index = []
        self.num_labels_by_type = []
        self.num_label_types = num_label_types
        self.figure = figure
        self.axes = axes

    def remap_instances(self, which_set, labels):
        if which_set == 'train':
            self.new_to_old_instance = [4, 6, 7, 8, 9]
        elif which_set == 'test':
            self.new_to_old_instance = [0, 1, 2, 3, 5]

        num_inst = len(self.new_to_old_instance)
        old_to_new_instance = np.ndarray(10, 'int')
        old_to_new_instance.fill(-1)
        old_to_new_instance[self.new_to_old_instance] = np.arange(num_inst)

        instance_slice = np.index_exp[:, self.instance_index]
        old_instances = labels[instance_slice]

        new_instances = old_to_new_instance[old_instances]
        labels[instance_slice] = new_instances

        azimuth_index = norb.SmallNORB.label_type_to_index['azimuth']
        azimuth_slice = np.index_exp[:, azimuth_index]
        labels[azimuth_slice] = labels[azimuth_slice] / 2

        return self.new_to_old_instance, labels

    def get_new_azimuth_degrees(self, scalar_label):
        return 20 * scalar_label

    def redraw(self, redraw_text, redraw_images):
        if redraw_text:
            cl = self.current_labels

            lines = [
                'category: %s' % norb.SmallNORB.get_category(cl[0]),
                'instance: %d' % self.new_to_old_instance[cl[1]],
                'elevation: %d' % norb.SmallNORB.get_elevation_degrees(cl[2]),
                'azimuth: %d' % self.get_new_azimuth_degrees(cl[3]),
                'lighting: %d' % cl[4]]

            lt = self.current_label_type[0]
            lines[lt] = '==> ' + lines[lt]
            text = ('Up/down arrows choose label, left/right arrows change it'
                    '\n\n' +
                    '\n'.join(lines))
            self.label_text.set_text(text)

        if redraw_images:
            index = self.label_to_index[tuple(self.current_labels)]

            image_pair = values[index, :, :, :]
            for i in range(2):
                self.axes[i].imshow(image_pair[i, :, :], cmap='gray')

        self.figure.canvas.draw()

    def on_key_press(self, event):

        def add_mod(arg, step, size):
            return (arg + size + step) % size

        def incr_label_type(step):
            self.current_label_type[0] = add_mod(self.current_label_type[0],
                                                 step, num_label_types)

        def incr_label(step):
            lt = self.current_label_type[0]
            num_labels = self.num_labels_by_type[lt]
            self.current_labels[lt] = add_mod(self.current_labels[lt],
                                              step, num_labels)

        if event.key == 'up':
            incr_label_type(-1)
            self.redraw(True, False)
        elif event.key == 'down':
            incr_label_type(1)
            self.redraw(True, False)
        elif event.key == 'left':
            incr_label(-1)
            self.redraw(True, True)
        elif event.key == 'right':
            incr_label(1)
            self.redraw(True, True)
        elif event.key == 'q':
            sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browser for SmallNORB dataset.")

    parser.add_argument('--which_set',
                        default='train',
                        help="'train', 'test', or the path to a .pkl file")

    parser.add_argument('--zca',
                        default=None,
                        help=("if --which_set points to a .pkl "
                              "file storing a ZCA-preprocessed "
                              "NORB dataset, you can optionally "
                              "enter the preprocessor's .pkl "
                              "file path here to undo the "
                              "ZCA'ing for visualization "
                              "purposes."))

    return parser.parse_args()


def get_data(args):
    if args.which_set in ('train', 'test'):
        dataset = norb.SmallNORB(args.which_set, True)
    else:
        with open(args.which_set) as norb_file:
            dataset = pickle.load(norb_file)
            if len(dataset.y.shape) < 2 or dataset.y.shape[1] == 1:
                print("This viewer does not support NORB datasets that "
                      "only have classification labels.")
                sys.exit(1)

        if args.zca is not None:
            with open(args.zca) as zca_file:
                zca = pickle.load(zca_file)
                dataset.X = zca.inverse(dataset.X)

    num_examples = dataset.X.shape[0]

    topo_shape = ((num_examples, ) +
                  tuple(dataset.view_converter.shape))
    assert topo_shape[-1] == 1
    topo_shape = topo_shape[:-1]
    values = dataset.X.reshape(topo_shape)
    labels = np.array(dataset.y, 'int')
    return values, labels, dataset.which_set


if __name__ == '__main__':
    args = parse_args()
    values, labels, which_set = get_data(args)

    instance_index = norb.SmallNORB.label_type_to_index['instance']
    num_label_types = len(norb.SmallNORB.num_labels_by_type)
    current_labels = np.zeros(num_label_types, 'int')
    current_label_type = [0, ]

    figure, axes = pyplot.subplots(1, 2, squeeze=True)

    figure.canvas.set_window_title('Small NORB dataset (%sing set)' %
                                   which_set)

    # shift subplots down to make more room for the text
    figure.subplots_adjust(bottom=0.05)

    label_text = figure.suptitle("title text",
                                 x=0.1,
                                 horizontalalignment="left")

    browser = SmallNorbBrowser(instance_index, current_labels,
                               current_label_type, label_text, figure, axes,
                               num_label_types)

    browser.figure.canvas.mpl_connect('key_press_event', browser.on_key_press)

    # For programming convenience, internally remap the instance labels to be
    # 0-4, and the azimuth labels to be 0-17. The user will still only see the
    # original, unmodified label values.
    new_to_old_instance, new_labels = browser.remap_instances(which_set,
                                                              labels)

    # Maps a label vector to the corresponding index in <values>
    browser.num_labels_by_type = np.array(norb.SmallNORB.num_labels_by_type,
                                          'int')
    browser.num_labels_by_type[instance_index] = len(new_to_old_instance)

    browser.label_to_index = np.ndarray(browser.num_labels_by_type, 'int')
    browser.label_to_index.fill(-1)

    for i, label in enumerate(new_labels):
        browser.label_to_index[tuple(label)] = i

    # all elements have been set
    assert not np.any(browser.label_to_index == -1)

    browser.redraw(True, True)

    pyplot.show()
