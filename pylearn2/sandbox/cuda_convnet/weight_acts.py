"""
A theano / pylearn2 wrapper for cuda-convnet's convFilterActs function.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow and David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

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

from theano.misc.strutil import render_string
from theano.sandbox.cuda import CudaNdarrayType
from theano.gof import Apply
from pylearn2.sandbox.cuda_convnet.base_acts import BaseActs
from pylearn2.sandbox.cuda_convnet.base_acts import UnimplementedError


class WeightActs(BaseActs):
    """
    Transforms the gradient on the output of FilterActs into the gradient
    on FilterActs' weights.

    This is intended to be a very low-level, performance-oriented op.

    It will not try to fix the input for you. That would slow it down.
    The input must be in the right format. If not, it raises an exception.

    Currently, this op must be inserted manually, not by optimizations.

    Note that the word "input" below refers to the input to FilterActs.

    * images: (input channels, rows, cols, batch_size)
      Input channels must be divisible by 4.
    * hid_grads: (output channels, rows, cols, batch_size)
      Output channels must be a multiple of 16.
    * filters: (input channels, filter rows, filter cols, output channels)
      Filter rows must be the same as filter cols.

    Notes
    -----
    All of these convolution routines are optimized for the case
    when the number of images (i.e. the minibatch size) is a multiple
    of 128. Other batch sizes will work, but Alex "made no attempt
    whatsoever to make them work fast."
    """

    # __eq__ and __hash__ are defined in BaseActs.
    # If you add an __init__ method that adds new members to WeightActs,
    # you may need to implement a new version of __eq__ and __hash__
    # in WeightActs, that considers these parameters.

    def make_node(self, images, hid_grads, output_shape):
        """
        .. todo::

            WRITEME
        """
        if not isinstance(images.type, CudaNdarrayType):
            raise TypeError("WeightActs: expected images.type "
                            "to be CudaNdarrayType, "
                            "got " + str(images.type))

        if not isinstance(hid_grads.type, CudaNdarrayType):
            raise TypeError("WeightActs: expected hid_acts.type "
                            "to be CudaNdarrayType, "
                            "got " + str(hid_grads.type))

        assert images.ndim == 4
        assert hid_grads.ndim == 4

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

        partial_sums_type = CudaNdarrayType(
            (False,) * 5
        )
        weights_grads = weights_grads_type()
        partial_sums = partial_sums_type()

        return Apply(self, [images, hid_grads, output_shape],
                     [weights_grads, partial_sums])

    def flops(self, inputs, outputs):
        """ Useful with the hack in profilemode to print the MFlops"""
        images, kerns, output_shape = inputs
        out, partial = outputs
        # The partial sum is just a way to specify how to compute
        # stuff inside the op.  It don't change the number of flops.
        assert images[3] == kerns[3]
        # nb mul and add by output pixed
        flops = kerns[1] * kerns[2] * 2
        #nb flops by output image
        flops *= out[1] * out[2]
        # for all outputs images#n_stack==self.imshp[0]
        flops *= images[3] * kerns[0] * images[0]
        return flops

    def c_headers(self):
        """
        .. todo::

            WRITEME
        """
        # For some reason, the function called in the C code (_weightActs)
        # is not defined in cudaconv2.cuh, so I defined it in weight_acts.cuh
        headers = super(WeightActs, self).c_headers()
        headers.append('weight_acts.cuh')
        return headers

    def c_code(self, node, name, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        partial_sum = self.partial_sum if self.partial_sum is not None else 0
        images, hid_grads, output_shape = inputs
        weights_grads, partialsum_storage = outputs
        fail = sub['fail']
        pad = self.pad

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
        #define paddingStart (-%(pad)d)
        const int *hid_grads_dims = CudaNdarray_HOST_DIMS(%(hid_grads)s);
        const int hidGradsSizeY = hid_grads_dims[1];
        const int hidGradsSizeX = hid_grads_dims[2];
        const int numModules = hidGradsSizeX * hidGradsSizeY;
        int partialSum = %(partial_sum)d > 0 ? %(partial_sum)d : numModules;

        // using this expression instead of numModules %% partialSum
        // because nvcc+msvc9 yield a strange behaviour when using %%
        if ( numModules - (numModules / partialSum) * partialSum != 0) {
            PyErr_Format(PyExc_ValueError,
                "partialSum must divide numModules, but partialSum=%%d and "
                "numModules=%%d", partialSum, numModules);
            %(fail)s;
        }
        """

        basic_setup += """
        #define moduleStride %d
        """ % self.stride
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
        const int * images_dims = CudaNdarray_HOST_DIMS(%(images)s);
        const int img_channels = images_dims[0];
        if (img_channels > 3 && img_channels %% 4 != 0)
        {
            PyErr_Format(PyExc_ValueError,
                "images must have 3 or fewer channels, or have a multiple of 4 channels, got %%i",
                img_channels);
            %(fail)s;
        }

        { //setup_nv_images brace 2
        const int * hid_grads_dims = CudaNdarray_HOST_DIMS(%(hid_grads)s);
        const int imgSizeY = images_dims[1];
        const int imgSizeX = images_dims[2];
        const int batch_size = images_dims[3];
        NVMatrix nv_images(%(images)s, img_channels * imgSizeY * imgSizeX, batch_size, "weight_acts: nv_images");
        """
        num_braces += 2

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
        const int numFilters = hid_grads_dims[0];
        const int batch_size = hid_grads_dims[3];
        NVMatrix nv_hid_grads(%(hid_grads)s, numFilters * hidGradsSizeY *
                                           hidGradsSizeX, batch_size, "weight_acts:nv_hid_grads");
        """
        num_braces += 1

        setup_nv_weights_grads = """
        int filters_dims[4];
        // filters:  (input channels, filter rows, filter cols, output channels)

        npy_intp *shape_dims = PyArray_DIMS(%(output_shape)s);
        npy_intp target_rows, target_cols;
        PyArrayObject *casted_shape;
        PyArray_Descr *intp_dtype;
        if (PyArray_NDIM(%(output_shape)s) != 1) {
            PyErr_Format(PyExc_ValueError,
                         "output shape must be a vector, got %%d-tensor",
                         PyArray_NDIM(%(output_shape)s));
            %(fail)s;
        }
        else if (shape_dims[0] != 2)
        {
            PyErr_Format(PyExc_ValueError,
                         "output shape must be length 2, got %%d",
                         (int)shape_dims[0]);
            %(fail)s;
        }
        else if ((PyArray_DESCR(%(output_shape)s))->kind != 'i' &&
                 (PyArray_DESCR(%(output_shape)s))->kind != 'u')
        {
            PyErr_SetString(PyExc_TypeError,
                            "output shape must have integer or uint dtype");
            %(fail)s;
        }
        intp_dtype = PyArray_DescrFromType(NPY_INTP);
        casted_shape = (PyArrayObject *)PyArray_CastToType(%(output_shape)s,
                                                           intp_dtype, 0);
        target_rows = *((npy_intp *)PyArray_GETPTR1(casted_shape, 0));
        target_cols = *((npy_intp *)PyArray_GETPTR1(casted_shape, 1));
        filters_dims[0] = img_channels;
        filters_dims[1] = target_rows;
        filters_dims[2] = target_cols;
        if (filters_dims[1] != filters_dims[2])
        {
            PyErr_Format(PyExc_ValueError,
            "filter must be square, but have shape (%%d, %%d).",
            filters_dims[1], filters_dims[2]);
            %(fail)s;
        }
        else if (moduleStride > filters_dims[1]) {
            PyErr_Format(PyExc_ValueError,
            "stride %%d greater than filter size (%%d, %%d)",
            moduleStride, filters_dims[1], filters_dims[2]);
            %(fail)s;
        }
        filters_dims[3] = numFilters;
        const int filterSize = filters_dims[1];
        int partialsum_storage_dims[5];
        for (int i = 1; i < 5; i++)
        {
            partialsum_storage_dims[i] = filters_dims[i - 1];
        }
        partialsum_storage_dims[0] = numModules / partialSum;
        if (partialSum != numModules &&
            CudaNdarray_prep_output(&%(partialsum_storage)s, 5,
                                    partialsum_storage_dims))
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
            %(fail)s;
        }

        { // setup_nv_weights_grad brace # 1

        NVMatrix nv_weights_grads(%(weights_grads)s, filters_dims[0] * filterSize
                                  * filterSize, numFilters,
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
            NVMatrix nv_partialsum(%(partialsum_storage)s, (numModules / partialSum) *
                     filters_dims[0] * filterSize * filterSize, numFilters,
                     "weight_acts: nv_partialsum");
            _weightActs(nv_images, nv_hid_grads, nv_partialsum,
                        imgSizeY, hidGradsSizeY, hidGradsSizeX, filterSize,
                        paddingStart, moduleStride, img_channels, numGroups,
                        partialSum, 0, 1);
            nv_partialsum.reshape((numModules / partialSum), filters_dims[0] * filterSize * filterSize * numFilters);

            // sum out axis 0 of nv_partialsum
            #define AXIS 0
            // scale the contents of nv_weights_grads by 0
            // i.e., clear out its pre-existing content
            #define SCALE_THIS 0
            // scale the new sum by 1, i.e., don't do any scaling
            #define SCALE_SUM 1
            nv_weights_grads.addSum(nv_partialsum, AXIS, SCALE_THIS, SCALE_SUM);
        }
        """

        braces = '}' * num_braces

        rval = (basic_setup +
                setup_nv_images +
                setup_nv_hid_grads +
                setup_nv_weights_grads +
                run_kernel +
                braces)

        rval = render_string(rval, locals())

        return rval

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        return (7,)
