"""
A theano / pylearn2 wrapper for cuda-convnet's convFilterActs function.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow and David Warde-Farley"]
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
from pylearn2.sandbox.cuda_convnet.base_acts import UnimplementedError
import warnings

class WeightActs(BaseActs):
    """
    Transforms the gradient on the output of FilterActs into the gradient
    on FilterActs' weights.

    This is intended to be a very low-level, performance-oriented op.

    It will not try to fix the input for you. That would slow it down.
    The input must be in the right format. If not, it raises an exception.

    Currently, this op must be inserted manually, not by optimizations.

    Note that the word "input" below refers to the input to FilterActs.

    images:          (input channels, rows, cols, batch_size)
                        input channels must be divisible by 4.
    hid_grads:       (output channels, rows, cols, batch_size)
                     output channels must be a multiple of 16

    filters:         (input channels, filter rows, filter cols, output channels)
                     filter rows must be the same as filter cols

    Note: all of these convolution routines are optimized for the case when
    the number of images (i.e. the minibatch size) is a multiple of 128.
    Other batch sizes will work, but Alex "made no attempt whatsoever
    to make them work fast."
    """
    cpp_source_file = "weight_acts.cu"

    def make_node(self, images, hid_grads):
        if not isinstance(images.type, CudaNdarrayType):
            raise TypeError("WeightActs: expected images.type "
                            "to be CudaNdarrayType, "
                            "got " + str(images.type))

        if not isinstance(hid_grads.type, CudaNdarrayType):
            raise TypeError("WeightActs: expected hid_acts.type "
                            "to be CudaNdarrayType, "
                            "got " + str(hid_grads.type))

        input_channels_broadcastable = images.type.broadcastable[0]
        # We don't know anything about filter_rows or filter_cols at compile
        # time, so we assume they're not broadcastable.
        filter_rows_broadcastable = False
        filter_cols_broadcastable = False
        output_channels_broadcastable = hid_grads.type.broadcastable[0]

        weights_grads_type = CudaNdarrayType(
                (input_channels_broadcastable,
                 filter_rows_broadcastable,
                 filter_cols_broadcastable,
                 output_channels_broadcastable))

        weights_grads = weights_grads_type()

        return Apply(self, [images, hid_grads], [weights_grads])

    def c_code(self, node, name, inputs, outputs, sub):
        partial_sum = self.partial_sum if self.partial_sum is not None else 0
        images, hid_grads = inputs
        weights_grads, = outputs
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

        basic_setup += """
        #define paddingStart -%d
        int partialSum = %(partial_sum)d > 0 ? %(partial_sum)d : num_Modules;
        if (numModule % partialSum > 0) {
            PyErr_Format(PyExc_ValueError,
                "partialSum must divide numModules, but partialSum=%%d and "
                "numModules=%%d", partialSum, numModules);
            %(fail)s;
        }
        """ % self.pad

        if self.stride != 1:
            raise UnimplementedError()
        else:
            basic_setup += """
            #define moduleStride 1
        """
        if self.copy_non_contiguous:
            raise UnimplementedError()
        else:
            basic_setup += "#define WEIGHTACTS_COPY_NON_CONTIGUOUS 0\n"

        # The amount of braces that must be closed at the end
        num_braces = 0

        # Convert images int nv_images, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_images = self._argument_contiguity_check("images") + """
        if (%(images)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "images must have nd=4, got nd=%%i", %(images)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1
        const int * hid_grads_dims = CudaNdarray_HOST_DIMS(%(hid_grads)s);
        const int * images_dims = CudaNdarray_HOST_DIMS(%(images)s);
        const int img_channels = images_dims[0];
        const int imgSizeY = images_dims[1];
        const int imgSizeX = images_dims[2];
        const int batch_size = images_dims[3];
        const int check_channels = 1;
        NVMatrix nv_images(%(images)s, img_channels * imgSizeY * imgSizeX, batch_size, "weight_acts: nv_images");
        """
        num_braces += 1

        # Convert hid_grads int nv_hid_grads, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_hid_grads = self._argument_contiguity_check("hid_grads") + """
        if (%(hid_grads)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "hid_grads must have nd=4, got nd=%%i", %(hid_grads)s->nd);
            %(fail)s;
        }

        { //setup_nv_hid_grads brace 1
        const int *hid_grads_dims = CudaNdarray_HOST_DIMS(%(hid_grads)s);
        const int numFilters = hid_grads_dims[0];
        const int hidGradsSizeY = hid_grads_dims[1];
        const int hidGradsSizeX = hid_grads_dims[2];
        const int numModules = hidGradsSizeX * hidGradsSizeY;
        const int batch_size = hid_grads_dims[3];
        NVMatrix nv_hid_grads(%(hid_grads)s, numFilters * hidGradsSizeY *
                                           hidGradsSizeX, batch_size, "weight_acts:nv_hid_grads");
        const int check_channels = 0;
        """
        num_braces += 1


        setup_nv_weights_grads = """
        int filters_dims[4];
        // filters:  (input channels, filter rows, filter cols, output channels)
        filters_dims[0] = img_channels;
        filters_dims[1] = imgSizeY - hidGradsSizeY + 1 - 2 * paddingStart;
        filters_dims[2] = imgSizeX - hidGradsSizeX + 1 - 2 * paddingStart;
        assert(filters_dims[1] == filters_dims[2]); // only square kernels are supported
        filters_dims[3] = numFilters;
        const int filterSize = filters_dims[1];
        int partialsum_storage_dims[5];
        for (int i = 1; i < 5; i++)
        {
            partialsum_storage_dims[i] = filters_dims[i - 1];
        }
        partialsum_storage_dims[0] = numModules / partialSum;
        CudaNdarray *partialsum_storage = NULL;
        if (partialSum != numModules &&
            CudaNdarray_prep_output(&partialsum_storage, 5,
                                    partialsum_storage_dims)
        {
            %(fail)s;
        }

        for (int i = 0; i < 4; i++)
        {
            if (filters_dims[i] <= 0)
            {
                printf("filters_dims[%%d] = %%d\\n", i, filters_dims[i]);
                assert(false);
            }
        }
        if (CudaNdarray_prep_output(& %(weights_grads)s, 4, filters_dims))
        {
            Py_DECREF(partialsum_storage);
            %(fail)s;
        }

        { // setup_nv_weights_grad brace # 1

        NVMatrix nv_weights_grads(%(weights_grads)s, filters_dims[0] * filterSize * filterSize, numFilters,
        "weight_acts:nv_weights_grads");

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
        run_kernel = """

        if (partialSum == numModules)
            _weightActs(nv_images, nv_hid_grads, nv_weights_grads,
                        imgSizeY, hidGradsSizeY, hidGradsSizeX, filterSize,
                        paddingStart, moduleStride, img_channels, numGroups,
                        partialSum, 0, 1);
        else {
            NVMatrix nv_partialsum(partialsum_storage, numModules / partialSum
            * filters_dims[0] * filterSize * filterSize, numFilters,
            "weight_acts: nv_partialsum");
            _weightActs(nv_images, nv_hid_grads, nv_partialsum,
                        imgSizeY, hidGradsSizeY, hidGradsSizeX, filterSize,
                        paddingStart, moduleStride, img_channels, numGroups,
                        partialSum, 0, 1);
            partialsum_storage.reshape(numModules / partialSum, filters_dims[0] *
                      filterSize * filterSize * numFilters);

            // sum out axis 0 of nv_partialsum
            #define AXIS 0
            // scale the contents of nv_weights_grads by 0
            // i.e., clear out its pre-existing content
            #define SCALE_THIS 0
            // scale the new sum by 1, i.e., don't do any scaling
            #define SCALE_SUM 1
            nv_weights_grads.addSum(nv_partialsum, AXIS, SCALE_THIS, SCALE_SUM);
        }
        }
        """

        warnings.warn("WeightActs does not attempt to use Alex's "
                      "partialSum flag intelligently. This probably "
                      "means our performance is suboptimal.")

        braces = '}' * num_braces

        rval = (basic_setup +
                setup_nv_images +
                setup_nv_hid_grads +
                setup_nv_weights_grads +
                run_kernel +
                braces)

        rval = rval % locals()

        return rval

