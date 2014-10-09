#!/usr/bin/env python

"""
A browser for the NORB and small NORB datasets. Navigate the images by
choosing the values for the label vector. Note that for the 'big' NORB
dataset, you can only set the first 5 label dimensions. You can then cycle
through the 3-12 images that fit those labels.
"""

import sys
import os
import argparse
import numpy
import warnings

try:
    import matplotlib
    from matplotlib import pyplot
except ImportError, import_error:
    warnings.warn("Can't use this script without matplotlib.")
    matplotlib = None
    pyplot = None

from pylearn2.datasets.new_norb import NORB
from pylearn2.utils import safe_zip, serial


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Browser for NORB dataset.")

    parser.add_argument('--which_norb',
                        type=str,
                        required=False,
                        choices=('big', 'small'),
                        help="'Selects the (big) NORB, or the Small NORB.")

    parser.add_argument('--which_set',
                        type=str,
                        required=False,
                        choices=('train', 'test', 'both'),
                        help="'train', or 'test'")

    parser.add_argument('--pkl',
                        type=str,
                        required=False,
                        help=".pkl file of NORB dataset")

    parser.add_argument('--stereo_viewer',
                        action='store_true',
                        help="Swaps left and right stereo images, so you "
                        "can see them in 3D by crossing your eyes.")

    parser.add_argument('--no_norm',
                        action='store_true',
                        help="Don't normalize pixel values")

    result = parser.parse_args()

    if (result.pkl is not None) == (result.which_norb is not None or
                                    result.which_set is not None):
        print("Must supply either --pkl, or both --which_norb and "
              "--which_set.")
        sys.exit(1)

    if (result.which_norb is None) != (result.which_set is None):
        print("When not supplying --pkl, you must supply both "
              "--which_norb and --which_set.")
        sys.exit(1)

    if result.pkl is not None:
        if not result.pkl.endswith('.pkl'):
            print("--pkl must be a filename that ends in .pkl")
            sys.exit(1)

        if not os.path.isfile(result.pkl):
            print("couldn't find --pkl file '%s'" % result.pkl)
            sys.exit(1)

    return result


def _make_grid_to_short_label(dataset):
    """
    Returns an array x such that x[a][b] gives label index a's b'th unique
    value. In other words, it maps label grid indices a, b to the
    corresponding label value.
    """
    unique_values = [sorted(list(frozenset(column)))
                     for column
                     in dataset.y[:, :5].transpose()]

    # If dataset contains blank images, removes the '-1' labels
    # corresponding to blank images, since they aren't contained in the
    # label grid.
    category_index = dataset.label_name_to_index['category']
    unique_categories = unique_values[category_index]
    category_to_name = dataset.label_to_value_funcs[category_index]
    if any(category_to_name(category) == 'blank'
           for category in unique_categories):
        for d in range(1, len(unique_values)):
            assert unique_values[d][0] == -1, ("unique_values: %s" %
                                               str(unique_values))
            unique_values[d] = unique_values[d][1:]

    return unique_values


def _get_blank_label(dataset):
    """
    Returns the label vector associated with blank images.

    If dataset is a Small NORB (i.e. it has no blank images), this returns
    None.
    """

    category_index = dataset.label_name_to_index['category']
    category_to_name = dataset.label_to_value_funcs[category_index]
    blank_label = 5

    try:
        blank_name = category_to_name(blank_label)
    except ValueError:
        # Returns None if there is no 'blank' category (e.g. if we're using
        # the small NORB dataset.
        return None

    assert blank_name == 'blank'

    blank_rowmask = dataset.y[:, category_index] == blank_label
    blank_labels = dataset.y[blank_rowmask, :]

    if not blank_rowmask.any():
        return None

    if not numpy.all(blank_labels[0, :] == blank_labels[1:, :]):
        raise ValueError("Expected all labels of category 'blank' to have "
                         "the same value, but they differed.")

    return blank_labels[0, :].copy()


def _make_label_to_row_indices(labels):
    """
    Returns a map from short labels (the first 5 elements of the label
    vector) to the list of row indices of rows in the dense design matrix
    with that label.

    For Small NORB, all unique short labels have exactly one row index.

    For big NORB, a short label can have 0-N row indices.
    """
    result = {}

    for row_index, label in enumerate(labels):

        short_label = tuple(label[:5])
        if result.get(short_label, None) is None:
            result[short_label] = []

        result[short_label].append(row_index)

    return result


def main():
    """Top-level function."""

    args = _parse_args()

    if args.pkl is not None:
        dataset = serial.load(args.pkl)
    else:
        dataset = NORB(args.which_norb, args.which_set)

    # Indexes into the first 5 labels, which live on a 5-D grid.
    grid_indices = [0, ] * 5

    grid_to_short_label = _make_grid_to_short_label(dataset)

    # Maps 5-D label vector to a list of row indices for dataset.X, dataset.y
    # that have those labels.
    label_to_row_indices = _make_label_to_row_indices(dataset.y)

    # Indexes into the row index lists returned by label_to_row_indices.
    object_image_index = [0, ]
    blank_image_index = [0, ]
    blank_label = _get_blank_label(dataset)

    # Index into grid_indices currently being edited
    grid_dimension = [0, ]

    dataset_is_stereo = 's' in dataset.view_converter.axes
    figure, all_axes = pyplot.subplots(1,
                                       3 if dataset_is_stereo else 2,
                                       squeeze=True,
                                       figsize=(10, 3.5))

    set_name = (os.path.split(args.pkl)[1] if args.which_set is None
                else "%sing set" % args.which_set)
    figure.canvas.set_window_title("NORB dataset (%s)" % set_name)

    label_text = figure.suptitle('Up/down arrows choose label, '
                                 'left/right arrows change it',
                                 x=0.1,
                                 horizontalalignment="left")

    # Hides axes' tick marks
    for axes in all_axes:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

    text_axes, image_axes = (all_axes[0], all_axes[1:])
    image_captions = (('left', 'right') if dataset_is_stereo
                      else ('mono image', ))

    if args.stereo_viewer:
        image_captions = tuple(reversed(image_captions))

    for image_ax, caption in safe_zip(image_axes, image_captions):
        image_ax.set_title(caption)

    text_axes.set_frame_on(False)  # Hides background of text_axes

    def is_blank(grid_indices):
        assert len(grid_indices) == 5
        assert all(x >= 0 for x in grid_indices)

        ci = dataset.label_name_to_index['category']  # category index
        category = grid_to_short_label[ci][grid_indices[ci]]
        category_name = dataset.label_to_value_funcs[ci](category)
        return category_name == 'blank'

    def get_short_label(grid_indices):
        """
        Returns the first 5 elements of the label vector pointed to by
        grid_indices. We use the first 5, since they're the labels used by
        both the 'big' and Small NORB datasets.
        """

        # Need to special-case the 'blank' category, since it lies outside of
        # the grid.
        if is_blank(grid_indices):   # won't happen with SmallNORB
            return tuple(blank_label[:5])
        else:
            return tuple(grid_to_short_label[i][g]
                         for i, g in enumerate(grid_indices))

    def get_row_indices(grid_indices):
        short_label = get_short_label(grid_indices)
        return label_to_row_indices.get(short_label, None)

    axes_to_pixels = {}

    def redraw(redraw_text, redraw_images):
        row_indices = get_row_indices(grid_indices)

        if row_indices is None:
            row_index = None
            image_index = 0
            num_images = 0
        else:
            image_index = (blank_image_index
                           if is_blank(grid_indices)
                           else object_image_index)[0]
            row_index = row_indices[image_index]
            num_images = len(row_indices)

        def draw_text():
            if row_indices is None:
                padding_length = dataset.y.shape[1] - len(grid_indices)
                current_label = (tuple(get_short_label(grid_indices)) +
                                 (0, ) * padding_length)
            else:
                current_label = dataset.y[row_index, :]

            label_names = dataset.label_index_to_name

            label_values = [label_to_value(label) for label_to_value, label
                            in safe_zip(dataset.label_to_value_funcs,
                                        current_label)]

            lines = ['%s: %s' % (t, v)
                     for t, v
                     in safe_zip(label_names, label_values)]

            if dataset.y.shape[1] > 5:
                # Inserts image number & blank line between editable and
                # fixed labels.
                lines = (lines[:5] +
                         ['No such image' if num_images == 0
                          else 'image: %d of %d' % (image_index + 1,
                                                    num_images),
                          '\n'] +
                         lines[5:])

            # prepends the current index's line with an arrow.
            lines[grid_dimension[0]] = '==> ' + lines[grid_dimension[0]]

            text_axes.clear()

            # "transAxes": 0, 0 = bottom-left, 1, 1 at upper-right.
            text_axes.text(0, 0.5,  # coords
                           '\n'.join(lines),
                           verticalalignment='center',
                           transform=text_axes.transAxes)

        def draw_images():
            if row_indices is None:
                for axis in image_axes:
                    axis.clear()
            else:
                data_row = dataset.X[row_index:row_index + 1, :]

                axes_names = dataset.view_converter.axes
                assert len(axes_names) in (4, 5)
                assert axes_names[0] == 'b'
                assert axes_names[-3] == 0
                assert axes_names[-2] == 1
                assert axes_names[-1] == 'c'

                def draw_image(image, axes):
                    assert len(image.shape) == 2
                    norm = matplotlib.colors.NoNorm() if args.no_norm else None
                    axes_to_pixels[axes] = image
                    axes.imshow(image, norm=norm, cmap='gray')

                if 's' in axes_names:
                    image_pair = \
                        dataset.get_topological_view(mat=data_row,
                                                     single_tensor=True)
                    # Shaves off the singleton dimensions
                    # (batch # and channel #), leaving just 's', 0, and 1.
                    image_pair = tuple(image_pair[0, :, :, :, 0])

                    if args.stereo_viewer:
                        image_pair = tuple(reversed(image_pair))

                    for axis, image in safe_zip(image_axes, image_pair):
                        draw_image(image, axis)
                else:
                    image = dataset.get_topological_view(mat=data_row)
                    image = image[0, :, :, 0]
                    draw_image(image, image_axes[0])

        if redraw_text:
            draw_text()

        if redraw_images:
            draw_images()

        figure.canvas.draw()

    default_status_text = ("mouseover image%s for pixel values" %
                           ("" if len(image_axes) == 1 else "s"))
    status_text = figure.text(0.5, 0.1, default_status_text)

    def on_mouse_motion(event):
        original_text = status_text.get_text()

        if event.inaxes not in image_axes:
            status_text.set_text(default_status_text)
        else:
            pixels = axes_to_pixels[event.inaxes]
            row = int(event.ydata + .5)
            col = int(event.xdata + .5)
            status_text.set_text("Pixel value: %g" % pixels[row, col])

        if status_text.get_text != original_text:
            figure.canvas.draw()

    def on_key_press(event):

        def add_mod(arg, step, size):
            return (arg + size + step) % size

        def incr_index_type(step):
            num_dimensions = len(grid_indices)
            if dataset.y.shape[1] > 5:
                # If dataset is big NORB, add one for the image index
                num_dimensions += 1

            grid_dimension[0] = add_mod(grid_dimension[0],
                                        step,
                                        num_dimensions)

        def incr_index(step):
            assert step in (0, -1, 1), ("Step was %d" % step)

            image_index = (blank_image_index
                           if is_blank(grid_indices)
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
        disable_left_right = (is_blank(grid_indices) and
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
    figure.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
    redraw(True, True)

    pyplot.show()


if __name__ == '__main__':
    main()
