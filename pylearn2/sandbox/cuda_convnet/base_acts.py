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
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
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
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

"""

import warnings

import theano
from theano.compat import get_unbound_function
from theano import config
from theano.sandbox.cuda import GpuOp

from pylearn2.sandbox.cuda_convnet.shared_code import this_dir
from pylearn2.sandbox.cuda_convnet.convnet_compile import convnet_available
from pylearn2.sandbox.cuda_convnet.convnet_compile import cuda_convnet_loc
from pylearn2.utils import py_integer_types

import pylearn2.sandbox.cuda_convnet.pthreads


class BaseActs(GpuOp):
    """
    Shared code for wrapping various convnet operations.
    """
    def __init__(self, pad=0, partial_sum=None, stride=1):

        if not isinstance(pad, py_integer_types):
            raise TypeError("pad must be an int")
        if not (pad >= 0):
            raise ValueError("bad value of pad (must be non-negative): " +
                             str(pad))

        self.partial_sum = partial_sum
        self.pad = pad
        self.stride = stride
        self.copy_non_contiguous = 0
        # TODO: support sparse connectivity pattern
        self.dense_connectivity = True

    def c_header_dirs(self):
        if config.pthreads.inc_dir:
            return [this_dir, config.pthreads.inc_dir]
        else:
            return [this_dir]

    def c_headers(self):
        return ['nvmatrix.cuh', 'cudaconv2.cuh']

    def c_code_cache_version(self):
        warnings.warn("No C-code cache version for %s" %
                      self.__class__.__name__)
        return ()

    def c_lib_dirs(self):
        if config.pthreads.lib_dir:
            return [cuda_convnet_loc, config.pthreads.lib_dir]
        else:
            return [cuda_convnet_loc]

    def c_libraries(self):
        if config.pthreads.lib:
            return ['cuda_convnet', config.pthreads.lib]
        else:
            return ['cuda_convnet']

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

    def _argument_dimension_check(self, arg_name, ndim):
        return """
        if (%%(%(arg_name)s)s->nd != %(ndim)d)
        {
            PyErr_Format(PyExc_ValueError,
                "%(arg_name)s must have ndim=%(ndim)d, got nd=%%%%i",
                %%(%(arg_name)s)s->nd);
            %%(fail)s;
        }
        """ % locals()

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.partial_sum == other.partial_sum and
                self.pad == other.pad and
                self.dense_connectivity == other.dense_connectivity and
                self.stride == other.stride and
                self.copy_non_contiguous == other.copy_non_contiguous)

    def __hash__(self):
        msg = []
        msg.append(self.__class__.__name__)
        for val in (self.partial_sum, self.pad, self.dense_connectivity,
                    self.stride, self.copy_non_contiguous):
            msg.append(str(val))

        return hash(tuple(msg))

    # Make sure the cuda_convnet library is compiled and up-to-date
    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        if not convnet_available():
            raise RuntimeError('Could not compile cuda_convnet')

        return super(BaseActs, self).make_thunk(node, storage_map,
                                                compute_map, no_recycling)

# This is needed as otherwise DebugMode will consider that
# BaseActs.make_thunk do something else then the default code, and
# would duplicate verification.
theano.compile.debugmode.default_make_thunk.append(
    get_unbound_function(BaseActs.make_thunk))


class UnimplementedError(Exception):
    """
    Like NotImplementedError, but designed not to be caught and suppressed
    by theano.
    """
