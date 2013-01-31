"""
Base class for wrapping
"""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["David Warde-Farley", "Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"

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

import warnings
from theano.sandbox.cuda import GpuOp
from pylearn2.sandbox.cuda_convnet.shared_code import get_NVMatrix_code
from pylearn2.sandbox.cuda_convnet.shared_code import load_code
from pylearn2.sandbox.cuda_convnet.shared_code import this_dir


class BaseActs(GpuOp):
    """
    Shared code for wrapping various convnet operations.
    """
    cpp_source_file = None

    def __init__(self):
        # TODO: support other amounts of padding
        self.pad = 0
        # TODO: support sparse connectivity pattern
        self.dense_connectivity = True
        # TODO: support other strides.
        # TODO: figure out Alex's code. There's only one stride var, does it
        # assume stride is same in both directions?
        self.stride = 1

    def c_compile_args(self):
        flags = ["-I" + this_dir]
        return flags

    def c_support_code(self):
        rval = get_NVMatrix_code()
        rval += '#include "cudaconv2.cuh"\n'
        if self.cpp_source_file is not None:
            rval += load_code(self.cpp_source_file)
        return rval

    def c_code_cache_version(self):
        warnings.warn("No C-code cache version for %s" %
                      self.__class__.__name__)
        return ()

    def filter_setup(self):
        """Introduces two opening braces."""
        return """
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

        if (check_channels && numGroups * filter_channels != img_channels)
        {
            PyErr_Format(PyExc_ValueError,
            "# input channels mismatch. images have %%d but filters have %%d groups of %%d for a total of %%d.",
            img_channels, numGroups, filter_channels, numGroups * filter_channels);
            %(fail)s;
        }

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
        filter_cols, num_filters);
        """
