#! /usr/bin/env python

"""
Interactive viewer for the convolutional weights in a pickled model.

Unlike ./show_weights, this shows one unit's weights at a time. This
allows it to display weights from higher levels (which can have 100s
of input channels), not just the first.
"""

import os
import sys
import warnings
import argparse
import numpy
from pylearn2.models.mlp import MLP, ConvElemwise, CompositeLayer
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.utils import safe_zip, serial
from pylearn2.space import Conv2DSpace

try:
    from matplotlib import pyplot
except ImportError as import_error:
    warnings.warn("Can't use this script without matplotlib.")
    pyplot = None


def _parse_args():
    parser = argparse.ArgumentParser(
        description=("Interactive browser of convolutional weights. "
                     "Up/down keys switch layers. "
                     "Left/right keys switch units."))

    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help=".pkl file of model")

    result = parser.parse_args()

    if os.path.splitext(result.input)[1] != '.pkl':
        print("Expected --input to end in .pkl, got %s." % result.input)
        sys.exit(1)

    return result


def _get_conv_layers(layer, result=None):
    '''
    Returns a list of the convolutional layers in a model.

    Returns
    -------
    rval: list
      Lists the convolutional layers (ConvElemwise, MaxoutConvC01B).
    '''

    if result is None:
        result = []

    if isinstance(layer, (MLP, CompositeLayer)):
        for sub_layer in layer.layers:
            _get_conv_layers(sub_layer, result)
    elif isinstance(layer, (MaxoutConvC01B, ConvElemwise)):
        result.append(layer)

    return result


def _get_conv_weights_bc01(layer):
    '''
    Returns a conv. layer's weights in BC01 format.

    Parameters
    ----------
    layer: MaxoutConvC01B or ConvElemwise

    Returns
    -------
    rval: numpy.ndarray
      The kernel weights in BC01 axis order. (B: output channels, C: input
      channels)
    '''

    assert isinstance(layer, (MaxoutConvC01B, ConvElemwise))
    weights = layer.get_params()[0].get_value()

    if isinstance(layer, MaxoutConvC01B):
        c01b = Conv2DSpace(shape=weights.shape[1:3],
                           num_channels=weights.shape[0],
                           axes=('c', 0, 1, 'b'))

        bc01 = Conv2DSpace(shape=c01b.shape,
                           num_channels=c01b.num_channels,
                           axes=('b', 'c', 0, 1))

        weights = c01b.np_format_as(weights, bc01)
    elif isinstance(layer, ConvElemwise):
        weights = weights[:, :, ::-1, ::-1]  # reverse 0, 1 axes

    return weights


def _num_conv_units(conv_layer):
    '''
    Returns a conv layer's number of output channels.
    '''

    assert isinstance(conv_layer, (MaxoutConvC01B, ConvElemwise))

    weights = conv_layer.get_params()[0].get_value()

    if isinstance(conv_layer, MaxoutConvC01B):
        return weights.shape[-1]
    elif isinstance(conv_layer, ConvElemwise):
        return weights.shape[0]


def main():
    "Entry point of script."

    args = _parse_args()

    model = serial.load(args.input)
    if not isinstance(model, MLP):
        print("Expected the .pkl file to contain an MLP, got a %s." %
              str(model.type))
        sys.exit(1)

    def get_figure_and_axes(conv_layers, window_width=800):
        kernel_display_width = 20
        margin = 5
        grid_square_width = kernel_display_width + margin
        num_columns = window_width // grid_square_width

        max_num_channels = numpy.max([layer.get_input_space().num_channels
                                      for layer in conv_layers])
        # pdb.set_trace()
        num_rows = max_num_channels // num_columns
        if num_rows * num_columns < max_num_channels:
            num_rows += 1

        assert num_rows * num_columns >= max_num_channels

        window_width = 15

        # '* 1.8' comse from the fact that rows take up about 1.8 times as much
        # space as columns, due to the title text.
        window_height = window_width * ((num_rows * 1.8) / num_columns)
        figure, all_axes = pyplot.subplots(num_rows,
                                           num_columns,
                                           squeeze=False,
                                           figsize=(window_width,
                                                    window_height))

        for unit_index, axes in enumerate(all_axes.flat):
            subplot_title = axes.set_title('%d' % unit_index)
            subplot_title.set_size(8)
            subplot_title.set_color((.3, .3, .3))

        # Hides tickmarks
        for axes_row in all_axes:
            for axes in axes_row:
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)

        return figure, all_axes

    conv_layers = _get_conv_layers(model)
    figure, all_axes = get_figure_and_axes(conv_layers)
    title_text = figure.suptitle("title")
    pyplot.tight_layout(h_pad=.1, w_pad=.5)  # in inches

    layer_index = numpy.array(0)
    unit_indices = numpy.zeros(len(model.layers), dtype=int)

    def redraw():
        '''
        Draws the currently selected convolutional kernel.
        '''

        axes_list = all_axes.flatten()
        layer = conv_layers[layer_index]
        unit_index = unit_indices[layer_index, ...]
        weights = _get_conv_weights_bc01(layer)[unit_index, ...]

        active_axes = axes_list[:weights.shape[0]]

        for axes, weights in safe_zip(active_axes, weights):
            axes.set_visible(True)
            axes.imshow(weights, cmap='gray', interpolation='nearest')

        assert len(frozenset(active_axes)) == len(active_axes)

        unused_axes = axes_list[len(active_axes):]
        assert len(frozenset(unused_axes)) == len(unused_axes)
        assert len(axes_list) == len(active_axes) + len(unused_axes)

        for axes in unused_axes:
            axes.set_visible(False)

        title_text.set_text("Layer %s, unit %d" %
                            (layer.layer_name,
                             unit_indices[layer_index]))

        figure.canvas.draw()

    def on_key_press(event):
        "Callback for key press events"

        def increment(index, size, step):
            """
            Increments an index in-place.

            Parameters
            ----------
            index: numpy.ndarray
              scalar (0-dim array) of dtype=int. Non-negative.

            size: int
              One more than the maximum permissible index.

            step: int
              -1, 0, or 1.
            """
            assert index >= 0
            assert step in (0, -1, 1)

            index[...] = (index + size + step) % size

        if event.key in ('up', 'down'):
            increment(layer_index,
                      len(conv_layers),
                      1 if event.key == 'up' else -1)
            unit_index = unit_indices[layer_index]
            redraw()
        elif event.key in ('right', 'left'):
            unit_index = unit_indices[layer_index:layer_index + 1]
            increment(unit_index,
                      _num_conv_units(conv_layers[layer_index]),
                      1 if event.key == 'right' else -1)
            redraw()
        elif event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)
    redraw()
    pyplot.show()


if __name__ == '__main__':
    main()
