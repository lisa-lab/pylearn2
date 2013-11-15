#!/usr/bin/env python

import sys, argparse
import numpy
from matplotlib import pyplot
from pylearn2.datasets import norb


def main():
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Browser for SmallNORB dataset.")

        parser.add_argument('--which_set', 
                            default='train',
                            help="'train' or 'test'")

        return parser.parse_args()


    def get_data(which_set):
        dataset = norb.SmallNORB(which_set, True)
        num_examples = dataset.get_data()[0].shape[0]
        iterator = dataset.iterator(mode = 'sequential', 
                                    batch_size = num_examples,
                                    topo = True, 
                                    targets = True)
        values, labels = iterator.next()
        return values, numpy.array(labels, 'int')

    args = parse_args()
    values, labels = get_data(args.which_set)

    # For programming convenience, internally remap the instance labels to be
    # 0-4, and the azimuth labels to be 0-17. The user will still only see the
    # original, unmodified label values.

    instance_index = norb.SmallNORB.label_type_to_index['instance']

    def remap_instances(which_set, labels):
        if which_set == 'train':
            new_to_old_instance = [4, 6, 7, 8, 9]
        elif which_set == 'test':
            new_to_old_instance = [0, 1, 2, 3, 5]

        num_instances = len(new_to_old_instance)
        old_to_new_instance = numpy.ndarray(10, 'int')
        old_to_new_instance.fill(-1)
        old_to_new_instance[new_to_old_instance] = numpy.arange(num_instances)

        instance_slice = numpy.index_exp[:, instance_index]
        old_instances = labels[instance_slice]

        new_instances = old_to_new_instance[old_instances]
        labels[instance_slice] = new_instances

        azimuth_index = norb.SmallNORB.label_type_to_index['azimuth']
        azimuth_slice = numpy.index_exp[:, azimuth_index]
        labels[azimuth_slice] = labels[azimuth_slice] / 2

        return new_to_old_instance

    new_to_old_instance = remap_instances(args.which_set, labels)

    def get_new_azimuth_degrees(scalar_label):
        return 20 * scalar_label;
    
    # Maps a label vector to the corresponding index in <values>
    num_labels_by_type = numpy.array(norb.SmallNORB.num_labels_by_type, 'int')
    num_labels_by_type[instance_index] = len(new_to_old_instance)

    label_to_index = numpy.ndarray(num_labels_by_type, 'int')
    label_to_index.fill(-1)

    for i, label in enumerate(labels):
        label_to_index[tuple(label)] = i

    assert not numpy.any(label_to_index == -1)  # all elements have been set

    figure, axes = pyplot.subplots(1,2, squeeze=True)

    figure.canvas.set_window_title('Small NORB dataset (%sing set)' % 
                                   args.which_set)

    # shift subplots down to make more room for the text
    figure.subplots_adjust(bottom=0.05)

    num_label_types = len(norb.SmallNORB.num_labels_by_type)
    current_labels = numpy.zeros(num_label_types, 'int')
    current_label_type = [0,]

    label_text = figure.suptitle("title text",
                                 x= .1, 
                                 horizontalalignment = "left")

    def redraw(redraw_text, redraw_images):
        if redraw_text:
            cl = current_labels

            lines = [
                'category: %s' % norb.SmallNORB.get_category(cl[0]),
                'instance: %d' % new_to_old_instance[cl[1]],
                'elevation: %d' % norb.SmallNORB.get_elevation_degrees(cl[2]),
                'azimuth: %d' % get_new_azimuth_degrees(cl[3]),
                'lighting: %d' % cl[4]]

            lt = current_label_type[0]
            lines[lt] = '==> ' + lines[lt]
            text = ('Up/down arrows choose label, left/right arrows change it'
                    '\n\n' + 
                    '\n'.join(lines))
            label_text.set_text(text)

        if redraw_images:
            index = label_to_index[tuple(current_labels)]

            image_pair = values[index, :, :, :]
            for i in range(2):
                axes[i].imshow(image_pair[i, :, :], cmap='gray')

        figure.canvas.draw()


    def on_key_press(event):

        def add_mod(arg, step, size):
            return (arg + size + step) % size

        def incr_label_type(step):
            current_label_type[0] = add_mod(current_label_type[0],
                                            step,
                                            num_label_types)
        def incr_label(step):
            lt = current_label_type[0]
            num_labels = num_labels_by_type[lt]
            current_labels[lt] = add_mod(current_labels[lt], step, num_labels)

            
        if event.key == 'up':
            incr_label_type(-1);
            redraw(True, False)
        elif event.key == 'down':
            incr_label_type(1)
            redraw(True, False)
        elif event.key == 'left':
            incr_label(-1)
            redraw(True, True)
        elif event.key == 'right':
            incr_label(1)
            redraw(True, True)
        elif event.key == 'q':
            sys.exit(0)


    figure.canvas.mpl_connect('key_press_event', on_key_press)
    redraw(True, True)
    
    pyplot.show()


if __name__ == '__main__':
    main()
