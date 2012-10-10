"""
A theano / pylearn2 wrapper for cuda-convnet's convFilterActs function.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
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

from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda import CudaNdarrayType
from theano.gof import Apply
from pylearn2.sandbox.cuda_convnet.shared_code import get_NVMatrix_code
from pylearn2.sandbox.cuda_convnet.shared_code import load_code
from pylearn2.sandbox.cuda_convnet.shared_code import this_dir
import warnings

class FilterActs(GpuOp):
    """
    2D convolution implemented on GPU.
    Technically not a true convolution, as it does not flip the kernel.

    This is intended to be a very low-level, performance-oriented op.

    It will not try to fix the input for you. That would slow it down.
    The input must be in the right format. If not, it raises an exception.

    Currently, this op must be inserted manually, not by optimizations.


    images:          (channels, rows, cols, batch_size)
    filters:         (input channels, filter rows, filter cols, output channels)
                     rows must be the same as cols
                     output channels must be a multiple of 16

    output:         (output channels, output rows, output cols, batch size)

    Note: all of these convolution routines are optimized for the case when
    the number of images (i.e. the minibatch size) is a multiple of 128.
    Other batch sizes will work, but Alex made no attempt whatsoever
    to make them work fast.
    """

    def __init__(self):
        self.pad = 0 # TODO: support other amounts of padding
        self.dense_connectivity = True #TODO: support sparse connectivity pattern
        self.stride = 1 # TODO: support other strides. TODO: figure out Alex's code. There's only one stride var, does it assume stride is same in both directions?

    def make_node(self, images, filters):

        if not isinstance(images.type, CudaNdarrayType):
            raise TypeError("FilterActs: expected images.type to be CudaNdarrayType, "
                    "got "+str(images.type))

        if not isinstance(filters.type, CudaNdarrayType):
            raise TypeError("FilterActs: expected filters.type to be CudaNdarrayType, "
                    "got "+str(filters.type))


        channels_broadcastable = filters.type.broadcastable[3]
        batch_broadcastable = images.type.broadcastable[3]
        # Computing whether the rows and columns are broadcastable requires doing
        # arithmetic on quantities that are known only at runtime, like the specific
        # shape of the image and kernel
        rows_broadcastable = False
        cols_broadcastable = False

        targets_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        targets = targets_type()

        return Apply(self, [images, filters], [targets])

    def c_compile_args(self):
        flags = ["-I"+this_dir]
        warnings.warn("FilterActs uses -g")
        flags += [ '-g' ]
        return flags

    def c_support_code(self):

        rval = get_NVMatrix_code()
        rval += '#include "cudaconv2.cuh"'
        rval += load_code("filter_acts.cu")
        return rval

    def c_code(self, node, name, inputs, outputs, sub):

        images, filters = inputs
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

        # Convert images int nv_images, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_images = """
        if (%(images)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "images must have nd=4, got nd=%%i", %(images)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1
        const int * images_dims = CudaNdarray_HOST_DIMS(%(images)s);
        const int img_channels = images_dims[0];
        const int imgSizeY = images_dims[1];
        const int imgSizeX = images_dims[2];
        const int batch_size = images_dims[3];
        NVMatrix nv_images(%(images)s, img_channels * imgSizeY * imgSizeX, batch_size);
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

        if (numGroups * filter_channels != img_channels)
        {
            PyErr_Format(PyExc_ValueError,
            "# input channels mismatch. images have %%d but filters have %%d groups of %%d for a total of %%d.",
            img_channels, numGroups, filter_channels, numGroups * filter_channels);
            %(fail)s;
        }

        if ((num_filters %% (numGroups * 16)) != 0)
        {
            PyErr_Format(PyExc_ValueError,
            "Each group must have a multiple of 16 channels, but num_filters %% (numGroups * 16) = %%d %% ( %%d * 16) = %%d.",
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
        filter_cols, num_filters);

        """

        num_braces += 2

        if self.pad != 0:
            raise NotImplementedError()
        else:
            target_rows = "imgSizeY - filter_rows + 1"
            target_cols = "imgSizeX - filter_cols + 1"

        setup_nv_targets = """


        int target_dims [] = {
            num_filters,
            %(target_rows)s,
            %(target_cols)s,
            batch_size };

        #define numModulesY target_dims[1]
        #define numModulesX target_dims[2]

        if (CudaNdarray_ensure_dims(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_filters brace # 1

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1]
         * target_dims[2], target_dims[3]);

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
        convFilterActs(nv_images, nv_filters, nv_targets,
                       imgSizeY, numModulesY, numModulesX,
                       paddingStart, moduleStride, img_channels,
                       numGroups, scaleTargets, scaleOutput);
        """

        braces = '}' * num_braces

        rval = basic_setup + \
                setup_nv_images + \
                setup_nv_filters + \
                setup_nv_targets + \
                do_convolution + \
                braces

        rval = rval % locals()

        return rval

    def c_code_cache_version(self):
        warnings.warn("FilterActs does not use c_code_cache_version")
        return ()
