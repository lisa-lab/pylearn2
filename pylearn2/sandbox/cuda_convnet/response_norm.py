"""
A theano / pylearn2 wrapper for cuda-convnet's response normalization
functions.
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

import theano
from theano.sandbox.cuda import CudaNdarrayType
from theano.gof import Apply, local_optimizer, TopoOptimizer
from pylearn2.sandbox.cuda_convnet.base_acts import BaseActs


class CrossMapNorm(BaseActs):
    """
    Parameters
    ----------
    size_f ; int
        Filter neighbourhood size. Must be >= 1 and <= the number
        of filters (I think).

    add_scale : float
        Constant that scales the sum in the denominator (alpha).

    pow_scale : float
        Exponent to which the denominator is raised (beta).

    blocked : bool
        Controls the "block-wise" behaviour in a way I don't quite
        understand.
    """
    def __init__(self, size_f, add_scale, pow_scale, blocked):
        if size_f < 0:
            raise ValueError("size_f must be positive (got %d)" % size_f)
        self._size_f = int(size_f)
        self._add_scale = float(add_scale)
        self._pow_scale = float(pow_scale)
        self._blocked = bool(blocked)

    def __hash__(self):
        return hash((self._size_f, self._add_scale, self._pow_scale,
                     self._blocked))

    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

    def make_node(self, images):
        if not isinstance(images.type, CudaNdarrayType):
            raise TypeError("CrossMapNorm: expected images.type to be CudaNdarrayType, "
                    "got " + str(images.type))

        assert images.ndim == 4

        targets_broadcastable = images.type.broadcastable
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        denoms = targets_type()
        targets = targets_type()

        return Apply(self, [images], [targets, denoms])

    def c_code(self, node, name, inputs, outputs, sub):
        images, = inputs
        targets, denoms = outputs
        fail = sub['fail']
        num_braces = 0

        size_f = self._size_f
        add_scale = self._add_scale
        pow_scale = self._pow_scale
        blocked = "true" if self._blocked else "false"

        basic_setup = """
        #define CROSSMAPNORM_COPY_NON_CONTIGUOUS 0
        int sizeF = %(size_f)d;
        float addScale = %(add_scale)f;
        float powScale = %(pow_scale)f;
        bool blocked = %(blocked)s;
        """

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
        const int numFilters = images_dims[0];
        const int imgSizeY = images_dims[1];
        const int imgSizeX = images_dims[2];
        const int batch_size = images_dims[3];
        if (numFilters %% 16 != 0)
        {
            PyErr_Format(PyExc_ValueError, "CrossMapNorm: images.shape[0] "
                         "must be a multiple of 16, but got %%d",
                         images_dims[0]);
            %(fail)s;
        }
        if (sizeF > images_dims[0]) {
            PyErr_Format(PyExc_ValueError, "CrossMapNorm: size_f "
                         "is %%d but images.shape[0] is %%d", sizeF,
                         images_dims[0]);
            %(fail)s;
        }
        if (imgSizeY != imgSizeX) {
            PyErr_Format(PyExc_ValueError, "CrossMapNorm: images "
                         "must be square; got (%%d, %%d)", imgSizeY, imgSizeX);
            %(fail)s;
        }
        { // setup_nv_images brace 2
        NVMatrix nv_images(%(images)s, numFilters * imgSizeY * imgSizeX, batch_size,
        "response_norm:nv_images");
        """

        num_braces += 2

        setup_nv_targets = """
        const int *target_dims = images_dims;

        if (CudaNdarray_prep_output(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_targets brace #1

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1]
         * target_dims[2], target_dims[3], "response_norm:nv_targets");
        """
        num_braces += 1

        setup_nv_denoms = """
        const int *denoms_dims = images_dims;

        if (CudaNdarray_prep_output(& %(denoms)s, 4, denoms_dims))
        {
            %(fail)s;
        }

        { // setup_nv_denoms brace #1

        NVMatrix nv_denoms(%(targets)s, denoms_dims[0] * denoms_dims[1]
         * denoms_dims[2], denoms_dims[3], "response_norm:nv_denoms");
        """

        num_braces += 1

        do_normalize = """
        convResponseNormCrossMap(nv_images, nv_denoms, nv_targets, numFilters, sizeF,
                                 addScale, powScale, blocked);
        """

        braces = '}' * num_braces

        rval = (basic_setup +
                setup_nv_images +
                setup_nv_targets +
                setup_nv_denoms +
                do_normalize +
                braces)

        rval = rval % locals()

        return rval

    def grad(self, inputs, dout):
        images, = inputs
        acts, denoms = self(images)
        dout, _ = dout  # Ignore the gradient on "denoms"
        grad_op = CrossMapNormUndo(self._size_f, self._add_scale,
                                   self._pow_scale, self._blocked,
                                   inplace=False)
        return [grad_op(images, acts, denoms, dout)[0]]

    def __str__(self):
        return (self.__class__.__name__ +
                "[size_f=%d,add_scale=%f,pow_scale=%f,blocked=%s]"
                % (self._size_f, self._add_scale, self._pow_scale,
                   self._blocked))

    def c_code_cache_version(self):
        return (2,)


class CrossMapNormUndo(CrossMapNorm):
    def __init__(self, size_f, add_scale, pow_scale, blocked, inplace=False):
        self._scale_targets = 0
        self._scale_outputs = 1
        self._inplace = inplace
        if inplace:
            self.destroy_map = {1: [1]}
        super(CrossMapNormUndo, self).__init__(size_f, add_scale, pow_scale,
                                               blocked)

    def __hash__(self):
        super_hash = super(CrossMapNormUndo, self).__hash__()
        return hash((super_hash, self._inplace))

    def make_node(self, images, acts, denoms, dout):
        if not isinstance(images.type, CudaNdarrayType):
            inputs = images, acts, denoms, dout
            names = "images", "acts", "denoms", "dout"
            for name, var in zip(names, inputs):
                if not isinstance(var.type, CudaNdarrayType):
                    raise TypeError("CrossMapNormUndo: expected %s.type "
                                    "to be CudaNdarrayType, "
                                    "got %s" (name, str(images.type)))
        assert images.ndim == 4
        assert acts.ndim == 4
        assert denoms.ndim == 4
        assert dout.ndim == 4
        # Not strictly necessary I don't think
        assert images.type.broadcastable == acts.type.broadcastable
        assert images.type.broadcastable == denoms.type.broadcastable
        assert images.type.broadcastable == dout.type.broadcastable

        targets_broadcastable = tuple(images.type.broadcastable)
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        targets = targets_type()
        out_acts = targets_type()
        return Apply(self, [images, acts, denoms, dout], [targets, out_acts])

    def c_code(self, node, name, inputs, outputs, sub):
        images, acts, denoms, dout = inputs
        targets, out_acts = outputs
        fail = sub['fail']
        num_braces = 0
        size_f = self._size_f
        add_scale = self._add_scale
        pow_scale = self._pow_scale
        blocked = "true" if self._blocked else "false"
        inplace = "true" if self._inplace else "false"
        scale_targets = int(self._scale_targets)
        scale_outputs = int(self._scale_outputs)

        basic_setup = """
        #define CROSSMAPNORMUNDO_COPY_NON_CONTIGUOUS 0
        int sizeF = %(size_f)d;
        float addScale = %(add_scale)f;
        float powScale = %(pow_scale)f;
        bool blocked = %(blocked)s;
        int scaleTargets = %(scale_targets)s;
        int scaleOutput = %(scale_outputs)s;
        """
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
        const int numFilters = images_dims[0];
        const int imgSizeY = images_dims[1];
        const int imgSizeX = images_dims[2];
        const int batch_size = images_dims[3];
        if (sizeF > images_dims[0]) {
            PyErr_Format(PyExc_ValueError, "CrossMapNormUndo: size_f "
                         "is %%d but images.shape[0] is %%d", sizeF,
                         images_dims[0]);
            %(fail)s;
        }
        if (numFilters %% 16 != 0)
        {
            PyErr_Format(PyExc_ValueError, "CrossMapNorm: images.shape[0] "
                         "must be a multiple of 16, but got %%d",
                         images_dims[0]);
            %(fail)s;
        }
        if (imgSizeY != imgSizeX) {
            PyErr_Format(PyExc_ValueError, "CrossMapNormUndo: images "
                         "must be square; got (%%d, %%d)", imgSizeY, imgSizeX);
            %(fail)s;
        }
        { // setup_nv_images brace 2
        NVMatrix nv_images(%(images)s, numFilters * imgSizeY * imgSizeX, batch_size,
        "response_norm:nv_images");
        """
        num_braces += 2
        setup_nv_acts = self._argument_contiguity_check("acts") + """
        if (%(acts)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "acts must have nd=4, got nd=%%i", %(acts)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1
        const int * acts_dims = CudaNdarray_HOST_DIMS(%(acts)s);
        if (acts_dims[0] != images_dims[0] ||
            acts_dims[1] != images_dims[1] ||
            acts_dims[2] != images_dims[2] ||
            acts_dims[3] != images_dims[3]) {
            PyErr_SetString(PyExc_ValueError, "CrossMapNormUndo: acts must "
                                              " have same shape as images");
            %(fail)s;
        }
        { // setup_nv_acts brace 2

        // XXX: Don't create this, we pass nv_out_acts instead.
        // NVMatrix nv_acts(%(acts)s, numFilters * imgSizeY * imgSizeX, batch_size,
        // "response_norm:nv_acts");
        """
        num_braces += 2
        setup_nv_denoms = self._argument_contiguity_check("denoms") + """
        if (%(denoms)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "denoms must have nd=4, got nd=%%i", %(denoms)s->nd);
            %(fail)s;
        }

        { //setup_nv_acts brace 1
        const int * denoms_dims = CudaNdarray_HOST_DIMS(%(denoms)s);
        if (denoms_dims[0] != images_dims[0] ||
            denoms_dims[1] != images_dims[1] ||
            denoms_dims[2] != images_dims[2] ||
            denoms_dims[3] != images_dims[3]) {
            PyErr_SetString(PyExc_ValueError, "CrossMapNormUndo: denoms must "
                                              " have same shape as images");
            %(fail)s;
        }
        { // setup_nv_denoms brace 2
        NVMatrix nv_denoms(%(denoms)s, numFilters * imgSizeY * imgSizeX, batch_size,
        "response_norm:nv_denoms");
        """
        num_braces += 2

        setup_nv_dout = self._argument_contiguity_check("dout") + """
        if (%(dout)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "dout must have nd=4, got nd=%%i", %(dout)s->nd);
            %(fail)s;
        }

        { //setup_nv_dout brace 1
        const int * dout_dims = CudaNdarray_HOST_DIMS(%(dout)s);
        if (dout_dims[0] != images_dims[0] ||
            dout_dims[1] != images_dims[1] ||
            dout_dims[2] != images_dims[2] ||
            dout_dims[3] != images_dims[3]) {
            PyErr_SetString(PyExc_ValueError, "CrossMapNormUndo: dout must "
                                              " have same shape as images");
            %(fail)s;
        }
        { // setup_nv_denoms brace 2
        NVMatrix nv_dout(%(denoms)s, numFilters * imgSizeY * imgSizeX, batch_size,
        "response_norm:nv_dout");
        """
        num_braces += 2
        setup_nv_targets = """
        const int *target_dims = images_dims;

        if (CudaNdarray_prep_output(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_targets brace #1

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1]
         * target_dims[2], target_dims[3], "response_norm:nv_targets");
        """
        num_braces += 1

        setup_nv_out_acts = """
        const int *out_acts_dims = images_dims;

        #if %(inplace)s
        // XXX: is this right?
        Py_XDECREF(%(out_acts)s);
        %(out_acts)s = %(acts)s;
        Py_INCREF(%(out_acts)s);
        #else
        if (CudaNdarray_prep_output(& %(out_acts)s, 4, out_acts_dims)) {
            Py_DECREF(%(targets)s);
            %(fail)s;
        }
        if (CudaNdarray_CopyFromCudaNdarray(%(out_acts)s, %(acts)s)) {
            Py_DECREF(%(targets)s);
            Py_DECREF(%(out_acts)s);
            %(fail)s;
        }
        #endif
        { // setup_nv_out_acts brace #1

        NVMatrix nv_out_acts(%(out_acts)s, target_dims[0] * target_dims[1]
         * target_dims[2], target_dims[3], "response_norm:nv_out_acts");
        """
        num_braces += 1
        undo_normalize = """
        convResponseNormCrossMapUndo(nv_dout, nv_denoms, nv_images,
                                     nv_out_acts, nv_targets, numFilters,
                                     sizeF, addScale, powScale, blocked,
                                     scaleTargets, scaleOutput);
        """
        rval = "\n".join((basic_setup,
                          setup_nv_images,
                          setup_nv_acts,
                          setup_nv_denoms,
                          setup_nv_dout,
                          setup_nv_targets,
                          setup_nv_out_acts,
                          undo_normalize,
                          "}" * num_braces))
        return rval % locals()

    def grad(self, inputs, dout):
        raise NotImplementedError()

    @property
    def inplace(self):
        return self._inplace

    def as_inplace(self):
        if self._inplace:
            raise ValueError("%s instance is already inplace, can't convert" %
                             self.__class__.__name__)
        return self.__class__(self._size_f, self._add_scale, self._pow_scale,
                              self._blocked, inplace=True)

    def __str__(self):
        return (self.__class__.__name__ +
                "[size_f=%d,add_scale=%f,pow_scale=%f,blocked=%s, inplace=%s]"
                % (self._size_f, self._add_scale, self._pow_scale,
                   self._blocked, self._inplace))

    def c_code_cache_version(self):
        return (2,)


@local_optimizer([None])
def local_crossmapnormundo_inplace(node):
    if isinstance(node.op, CrossMapNormUndo) and not node.op.inplace:
        new_op = node.op.as_inplace()
        new_node = new_op(*node.inputs)
        return new_node
    return False


theano.compile.optdb.register('local_crossmapnormundo_inplace',
                              TopoOptimizer(local_crossmapnormundo_inplace,
                                            failure_callback=TopoOptimizer.warn_inplace),
                              80, 'fast_run', 'inplace')
