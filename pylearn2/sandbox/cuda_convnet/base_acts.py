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

    def __init__(self, pad=0, partial_sum=None):
        self.partial_sum = partial_sum
        self.pad = pad
        # TODO: support sparse connectivity pattern
        self.dense_connectivity = True
        # TODO: support other strides.
        # TODO: figure out Alex's code. There's only one stride var, does it
        # assume stride is same in both directions?
        self.stride = 1
        self.copy_non_contiguous = 0

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

    def _argument_contiguity_check(self, arg_name):
        return """
        if (!CudaNdarray_is_c_contiguous(%%(%(arg_name)s)s))
        {
            if (!(%(class_name_caps)s_COPY_NON_CONTIGUOUS)) {
                PyErr_SetString(PyExc_ValueError,
                    "%(class)s: %(arg_name)s must be C contiguous");
                %%(fail)s;
            }
        }
        """ % {
            'class': self.__class__.__name__,
            'arg_name': arg_name,
            'class_name_caps': self.__class__.__name__.upper(),
        }

class UnimplementedError(Exception):
    """
    Like NotImplementedError, but designed not to be caught and suppressed
    by theano.
    """
