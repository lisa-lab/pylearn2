"""
A theano / pylearn2 wrapper for cuda-convnet's convFilterActs function.
"""
__authors__ = "David Warde-Farley and Ian Goodfellow"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["David Warde-Farley and Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

"""
This module may contain code copied directly or modified from cuda-convnet.
The copyright and licensing notice for this code is reproduced below:


/*
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

"""

from theano.sandbox.cuda import CudaNdarrayType
from theano.gof import Apply
from pylearn2.sandbox.cuda_convnet.base_acts import BaseActs

class ImageActs(BaseActs):
    """
    Transpose of FilterActs.

    This is intended to be a very low-level, performance-oriented op.

    It will not try to fix the input for you. That would slow it down.
    The input must be in the right format. If not, it raises an exception.

    Currently, this op must be inserted manually, not by optimizations.

    Note that below the term "input" refers to the input to FilterActs.
    This op does the tranpose of that, so its output is sized like FilterActs' input.

    images:          (output channels, rows, cols, batch_size)
    filters:         (input channels, filter rows, filter cols, output channels)
                     rows must be the same as cols
                     output channels must be a multiple of 16

    output:         (input channels, input rows, input cols, batch size)

    Note: all of these convolution routines are optimized for the case when
    the number of images (i.e. the minibatch size) is a multiple of 128.
    Other batch sizes will work, but Alex "made no attempt whatsoever
    to make them work fast."
    """
    cpp_source_file = "img_acts.cu"

    def make_node(self, hid_acts, filters):

        if not isinstance(hid_acts.type, CudaNdarrayType):
            raise TypeError("ImageActs: expected hid_acts.type to be CudaNdarrayType, "
                    "got " + str(hid_acts.type))

        if not isinstance(filters.type, CudaNdarrayType):
            raise TypeError("ImageActs: expected filters.type to be CudaNdarrayType, "
                    "got " + str(filters.type))


        channels_broadcastable = filters.type.broadcastable[3]
        batch_broadcastable = hid_acts.type.broadcastable[3]
        # Computing whether the rows and columns are broadcastable requires doing
        # arithmetic on quantities that are known only at runtime, like the specific
        # shape of the image and kernel
        rows_broadcastable = False
        cols_broadcastable = False

        targets_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        targets = targets_type()

        return Apply(self, [hid_acts, filters], [targets])

    def c_code(self, node, name, inputs, outputs, sub):
        hid_acts, filters = inputs
        targets, = outputs
        fail = sub['fail']

        # convFilterActs will multiply targets by scaleTargets
        # then add scaleOutput * (the convolution value)
        # We could make use of this to implement an inplace
        # addconv op but for this op we just want to compute
        # the convolution so we set them to 0 and 1 respectively
        # Note: there is another version of convFilterActs that
        # does not take these arguments, but it is just a wrapper
        # around the version that does take them, so we save
        # a function call by using the version that we use.
        basic_setup = """
        #define scaleTargets 0
        #define scaleOutput 1
        """

        if self.dense_connectivity:
            basic_setup += """
            #define numGroups 1
            """

        if self.pad != 0:
            raise NotImplementedError()
        else:
            basic_setup += """
            #define paddingStart 0
            """

        if self.stride != 1:
            raise NotImplementedError()
        else:
            basic_setup += """
            #define moduleStride 1
        """


        # The amount of braces that must be closed at the end
        num_braces = 0

        # Convert images int nv_hid_acts, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_hid_acts = """
        if (%(hid_acts)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "hid_acts must have nd=4, got nd=%%i", %(hid_acts)s->nd);
            %(fail)s;
        }

        { //setup_nv_hid_acts brace 1
        const int *hid_act_dims = CudaNdarray_HOST_DIMS(%(hid_acts)s);
        const int numFilters = hid_act_dims[0];
        const int hidActsSizeY = hid_act_dims[1];
        const int hidActsSizeX = hid_act_dims[2];
        const int batch_size = hid_act_dims[3];
        NVMatrix nv_hid_acts(%(hid_acts)s, numFilters * hidActsSizeY *
                                           hidActsSizeX, batch_size, "image_acts:nv_hid_acts");
        int img_channels = -1;
        """
        num_braces += 1

        # Convert filters into nv_filters, an NVMatrix, for compatibility
        # with the cuda-convnet functions

        setup_nv_filters = """
        if (%(filters)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
            "filters must have nd=4, got nd=%%i", %(filters)s->nd);
            %(fail)s;
        }

        { // setup_nv_filters brace 1
        const int * filters_dims = CudaNdarray_HOST_DIMS(%(filters)s);
        const int filter_channels = filters_dims[0];
        const int filter_rows = filters_dims[1];
        const int filter_cols = filters_dims[2];
        const int num_filters = filters_dims[3];

        if ((num_filters %% (numGroups * 16)) != 0)
        {
            PyErr_Format(PyExc_ValueError,
            "Each group must have a multiple of 16 channels, but num_filters %%%% (numGroups * 16) = %%d %%%% ( %%d * 16) = %%d.",
            num_filters, numGroups, num_filters %% (numGroups * 16));
            %(fail)s;
        }

        if (filter_rows != filter_cols)
        {
            PyErr_Format(PyExc_ValueError,
            "filter must be square, but have shape (%%d, %%d).",
            filter_rows, filter_cols);
            %(fail)s;
        }

        { // setup_nv_filters brace 2


        NVMatrix nv_filters(%(filters)s, filter_channels * filter_rows *
        filter_cols, num_filters, "img_acts:nv_filters");
        """
        num_braces += 2

        if self.pad != 0:
            raise NotImplementedError()
        else:
            target_rows = "hidActsSizeY + filter_rows - 1"
            target_cols = "hidActsSizeX + filter_cols - 1"

        setup_nv_targets = """


        int target_dims [] = {
            filter_channels,
            %(target_rows)s,
            %(target_cols)s,
            batch_size };

        #define numModulesY hid_act_dims[1]
        #define numModulesX hid_act_dims[2]

        if (CudaNdarray_prep_output(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_filters brace # 1
        const int imgSizeY = %(target_rows)s;
        const int imgSizeX = %(target_cols)s;

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1]
         * target_dims[2], target_dims[3], "image_acts: nv_targets");

        """

        num_braces += 1

        # note: imgSizeX is not specified here, it is computed internally
        # (in _filterActsSparse) by the lines:
        # int imgPixels = images.getNumRows() / numImgColors;
        # int imgSizeX = imgPixels / imgSizeY;
        #
        # note: numFilters is not specified here. it is determined by
        # nv_filters.getNumCols()
        #
        # note: the size of the filters is determined by dividing
        # nv_filters.getNumRows() by numFilterColors
        #
        do_convolution = """
        convImgActs(nv_hid_acts, nv_filters, nv_targets,
                    imgSizeY, imgSizeX, numModulesY,
                    paddingStart, moduleStride, filter_channels,
                    numGroups);
        """

        braces = '}' * num_braces

        rval = basic_setup + \
                setup_nv_hid_acts + \
                setup_nv_filters + \
                setup_nv_targets + \
                do_convolution + \
                braces

        rval = rval % locals()

        return rval

