#!/usr/bin/env python

import sys, argparse
import numpy
from matplotlib import pyplot
from pylearn2.datasets.norb import Norb
from pylearn2.utils import safe_zip


def main():
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Browser for NORB dataset.")

        parser.add_argument('--which_set',
                            type=str,
                            default="train",
                            help="'train', or 'test'")

        result = parser.parse_args()

        if not (result.which_set in ('train', 'test')):
            print "type of which_set: ", type(result.which_set)
            print ("--which_set must be one of 'train' or 'test'. "
                   "(Was '%s')." % result.which_set)
            sys.exit(1)

        return result

    args = parse_args()

    print "loading %s set..." % args.which_set
    dataset = Norb(args.which_set, True)
    print "...loaded"

    # Indexes into the first 5 labels, which live on a 5-D grid.
    grid_indices = [0, ] * 5

    def make_grid_to_short_label():
        unique_values = [sorted(list(frozenset(column)))
                         for column
                         in dataset.y[:, :5].transpose()]

        # Removes the '-1' labels corresponding to blank images, since they
        # aren't contained in the label grid.
        for d in range(1, len(unique_values)):
            assert unique_values[d][0] == -1
            unique_values[d] = unique_values[d][1:]

        return unique_values

    grid_to_short_label = make_grid_to_short_label()

    def make_label_to_row_indices():
        result = {}

        # print dataset.y
        for row_index, label in enumerate(dataset.y):

            short_label = tuple(label[:5])
            if result.get(short_label, None) is None:
                result[short_label] = []

            result[short_label].append(row_index)

        return result

    # maps 5-D label vector to a list of row indices for dataset.X, dataset.y
    # that have those labels.
    label_to_row_indices = make_label_to_row_indices()

    # indexes into the row index lists returned by label_to_row_indices
    object_image_index = [0, ]
    blank_image_index = [0, ]

    def get_short_label(grid_indices):
        category = grid_to_short_label[0][grid_indices[0]]

        if category == 5:  # category == 'blank'
            return tuple(dataset.blank_label[:5])
        else:
            return tuple(grid_to_short_label[i][g]
                         for i, g in enumerate(grid_indices))

    def get_row_indices(grid_indices):
        short_label = get_short_label(grid_indices)
        return label_to_row_indices.get(short_label, None)

    # Index into grid_indices currently being edited
    grid_dimension = [0, ]

    figure, all_axes = pyplot.subplots(1, 3, squeeze=True, figsize=(10, 3.5))

    figure.canvas.set_window_title("NORB dataset (%sing set)" %
                                   args.which_set)

    label_text = figure.suptitle('Up/down arrows choose label, '
                                 'left/right arrows change it',
                                 x=0.1,
                                 horizontalalignment="left")

    # Hides axes' tick marks
    for axes in all_axes:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

    text_axes, image_axes = (all_axes[0], all_axes[1:])

    text_axes.set_frame_on(False)  # Hides background of text_axes

    # Makes an array of label type names
    label_types = [None, ] * len(Norb.label_type_to_index)
    for label_type, index in dataset.label_type_to_index.items():
        label_types[index] = label_type

    def redraw(redraw_text, redraw_images):
        category = grid_to_short_label[0][grid_indices[0]]
        row_indices = get_row_indices(grid_indices)

        if row_indices is None:
            row_index = None
            image_index = 0
            num_images = 0
        else:
            image_index = (blank_image_index
                           if category == 5  # i.e. category == 'blank'
                           else object_image_index)[0]
            row_index = row_indices[image_index]
            num_images = len(row_indices)

        def draw_text():
            if row_indices is None:
                current_label = (tuple(get_short_label(grid_indices)) +
                                 (0, ) * 6)
            else:
                current_label = dataset.y[row_index, :]

            label_values = [v[i] for v, i
                            in safe_zip(dataset.label_to_value_maps,
                                        current_label)]

            lines = ['%s: %s' % (t, v)
                     for t, v
                     in safe_zip(label_types, label_values)]

            # Inserts image number & blank line between editable and
            # fixed labels.
            lines = (lines[:5] +
                     ['image: %d of %d' % (image_index, num_images),
                      '\n'] +
                     lines[5:])

            # prepends the current index's line with an arrow.
            lines[grid_dimension[0]] = '==> ' + lines[grid_dimension[0]]

            text_axes.clear()

            # "transAxes": 0, 0 = bottom-left, 1, 1 at upper-right.
            text_axes.text(0, 0,  # coords
                           '\n'.join(lines),
                           transform=text_axes.transAxes)

        def draw_images():
            if row_indices is None:
                for axis in image_axes:
                    axis.clear()
            else:
                data_row = dataset.X[row_index:row_index + 1, :]
                image_pair = dataset.get_topological_view(mat=data_row,
                                                          single_tensor=True)

                # Shaves off the singleton dimensions (batch # and channel #).
                image_pair = image_pair[0, :, :, :, 0]

                for axis, image in safe_zip(image_axes, image_pair):
                    axis.imshow(image, cmap='gray')

        if redraw_text:
            draw_text()

        if redraw_images:
            draw_images()

        figure.canvas.draw()

    def on_key_press(event):

        def add_mod(arg, step, size):
            return (arg + size + step) % size

        def incr_index_type(step):
            grid_dimension[0] = add_mod(grid_dimension[0],
                                        step,
                                        # +1 for the image number:
                                        len(grid_indices) + 1)

        def incr_index(step):
            assert step in (0, -1, 1), ("Step was %d" % step)

            image_index = (blank_image_index
                           if grid_indices[0] == 5  # category == 'blank'
                           else object_image_index)

            if grid_dimension[0] == 5:  # i.e. the image index
                row_indices = get_row_indices(grid_indices)
                if row_indices is None:
                    image_index[0] = 0
                else:
                    # increment the image index
                    image_index[0] = add_mod(image_index[0],
                                             step,
                                             len(row_indices))
            else:
                # increment one of the grid indices
                gd = grid_dimension[0]
                grid_indices[gd] = add_mod(grid_indices[gd],
                                           step,
                                           len(grid_to_short_label[gd]))

                row_indices = get_row_indices(grid_indices)
                if row_indices is None:
                    image_index[0] = 0
                else:
                    # some grid indices have 2 images instead of 3.
                    image_index[0] = min(image_index[0], len(row_indices))

        # Disables left/right key if we're currently showing a blank,
        # and the current index type is neither 'category' (0) nor
        # 'image number' (5)
        disable_left_right = ((grid_indices[0] == 5) and  # category == 'blank'
                              not (grid_dimension[0] in (0, 5)))

        if event.key == 'up':
            incr_index_type(-1)
            redraw(True, False)
        elif event.key == 'down':
            incr_index_type(1)
            redraw(True, False)
        elif event.key == 'q':
            sys.exit(0)
        elif not disable_left_right:
            if event.key == 'left':
                incr_index(-1)
                redraw(True, True)
            elif event.key == 'right':
                incr_index(1)
                redraw(True, True)

    figure.canvas.mpl_connect('key_press_event', on_key_press)
    redraw(True, True)

    print "displaying"
    pyplot.show()


if __name__ == '__main__':
    main()
