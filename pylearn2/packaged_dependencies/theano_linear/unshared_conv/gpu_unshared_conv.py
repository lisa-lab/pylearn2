"""
WRITEME
"""
import inspect
import os
import StringIO

import theano
from theano.sandbox.cuda import CudaNdarrayType
from theano.gof import local_optimizer
from theano.sandbox.cuda.opt import register_opt
from theano.sandbox.cuda import gpu_from_host, host_from_gpu

from .unshared_conv import FilterActs
from .unshared_conv import WeightActs
from .unshared_conv import ImgActs

_this_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))


# XXX: move to cuda.opt and refactor there
def any_from_gpu(*vv):
    """
    .. todo::

        WRITEME
    """
    for v in  vv:
        if v.owner and v.owner.op == host_from_gpu:
            return True
    return False


# XXX: move to cuda.opt and refactor there
def any_gpu_client(*vv):
    """
    .. todo::

        WRITEME
    """
    for v in vv:
        for (cl, pos) in v.clients:
            if cl.op == gpu_from_host:
                return True
    return False


class Base(theano.Op):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    module_stride : WRITEME
    partial_sum : WRITEME
    """
    def __init__(self, module_stride, partial_sum):
        self.module_stride = module_stride
        self.partial_sum = partial_sum

    def _attributes(self):
        """
        .. todo::

            WRITEME
        """
        return (
                self.module_stride,
                self.partial_sum,
                )

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (type(self) == type(other)
                and self._attributes() == other._attributes())

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self), self._attributes()))

    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return '%s{module_stride=%i,partial_sum=%i}' % (
                self.__class__.__name__,
                self.module_stride,
                self.partial_sum,
                )


class GpuFilterActs(Base):
    """
    .. todo::

        WRITEME
    """
    def make_node(self, images, filters):
        """
        .. todo::

            WRITEME
        """
        ibcast = images.broadcastable
        fbcast = filters.broadcastable
        igroups, icolors_per_group, irows, icols, icount = ibcast
        fmodulesR, fmodulesC, fcolors, frows, fcols = fbcast[:-2]
        fgroups, filters_per_group = fbcast[-2:]
        hbcast = (fgroups, filters_per_group, fmodulesR, fmodulesC, icount)
        if not isinstance(images.type, CudaNdarrayType):
            raise TypeError('gpu_filter_acts requires CudaNdarray images',
                    images)
        if not isinstance(filters.type, CudaNdarrayType):
            raise TypeError('gpu_filter_acts requires CudaNdarray filters',
                    filters)
        htype = CudaNdarrayType(broadcastable=hbcast)
        return theano.gof.Apply(self,
                [images, filters],
                [htype()])

    def c_support_code(self):
        """
        .. todo::

            WRITEME
        """
        cufile = open(os.path.join(_this_dir, 'filter_acts.cu'))
        return cufile.read()

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        return ()

    def c_code(self, node, nodename, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        images, filters, = inputs
        responses, = outputs
        fail = sub['fail']
        moduleStride = str(self.module_stride)
        sio = StringIO.StringIO()

        print >> sio, """

        //XXX: actually the rightmost images dimension can be strided
        if (!CudaNdarray_is_c_contiguous(%(images)s))
        {
            PyErr_Format(PyExc_NotImplementedError,
                "images not c contiguous");
            %(fail)s;
        }

        if (!CudaNdarray_is_c_contiguous(%(filters)s))
        {
            PyErr_Format(PyExc_NotImplementedError,
                "filters not c contiguous");
            %(fail)s;
        }

        if (%(images)s->nd != 5)
        {
            PyErr_Format(PyExc_TypeError,
                "images ndim (%%i) must be 5",
                %(images)s->nd);
            %(fail)s;
        }

        if (%(filters)s->nd != 7)
        {
            PyErr_Format(PyExc_TypeError,
                "filters ndim (%%i) must be 7",
                %(filters)s->nd);
            %(fail)s;
        }
        //fprintf(stderr, "really running on GPU\\n");

        { // new scope, new vars

            int igroups           = CudaNdarray_HOST_DIMS(%(images)s)[0];
            int icolors_per_group = CudaNdarray_HOST_DIMS(%(images)s)[1];
            int irows             = CudaNdarray_HOST_DIMS(%(images)s)[2];
            int icols             = CudaNdarray_HOST_DIMS(%(images)s)[3];
            int icount            = CudaNdarray_HOST_DIMS(%(images)s)[4];

            int fmodulesR         = CudaNdarray_HOST_DIMS(%(filters)s)[0];
            int fmodulesC         = CudaNdarray_HOST_DIMS(%(filters)s)[1];
            int fcolors           = CudaNdarray_HOST_DIMS(%(filters)s)[2];
            int frows             = CudaNdarray_HOST_DIMS(%(filters)s)[3];
            int fcols             = CudaNdarray_HOST_DIMS(%(filters)s)[4];
            int fgroups           = CudaNdarray_HOST_DIMS(%(filters)s)[5];
            int filters_per_group = CudaNdarray_HOST_DIMS(%(filters)s)[6];

            // XXX: use this parameter properly
            int paddingStart = 0;
            int imgStride = icount;
            float scaleTargets = 0.0;
            float scaleOutput = 1.0;
            bool conv = false;

            if (igroups != fgroups)
            {
                PyErr_Format(PyExc_ValueError,
                    "igroups != fgroups (%%i != %%i)",
                    igroups, fgroups);
                %(fail)s;
            }

            if (icolors_per_group != fcolors)
            {
                PyErr_Format(PyExc_ValueError,
                    "icolors_per_group != fcolors (%%i != %%i)",
                    icolors_per_group,
                    fcolors);
                %(fail)s;
            }

            if (!%(responses)s)
            {
                Py_XDECREF(%(responses)s);
                int dims[5];
                dims[0] = fgroups;
                dims[1] = filters_per_group;
                dims[2] = fmodulesR;
                dims[3] = fmodulesC;
                dims[4] = icount;
                %(responses)s = (CudaNdarray*)CudaNdarray_NewDims(5, dims);
                if (!%(responses)s)
                {
                    %(fail)s;
                }
            }

            assert(CudaNdarray_is_c_contiguous(%(responses)s));

            if (_filterActs(
                    igroups,
                    icolors_per_group,
                    irows,
                    icols,
                    icount,
                    fmodulesR,
                    fmodulesC,
                    frows,
                    fcols,
                    filters_per_group,
                    CudaNdarray_DEV_DATA(%(images)s),
                    CudaNdarray_DEV_DATA(%(filters)s),
                    CudaNdarray_DEV_DATA(%(responses)s),
                    paddingStart,
                    %(moduleStride)s,
                    imgStride,
                    scaleTargets,
                    scaleOutput,
                    conv))
            {
                %(fail)s;
            }
        } // end bogus scope used for vars

        """

        return sio.getvalue() % locals()


@register_opt()
@local_optimizer([FilterActs])
def insert_gpu_filter_acts(node):
    """
    .. todo::

        WRITEME
    """
    if isinstance(node.op, FilterActs):
        images, filters = node.inputs
        if any_from_gpu(images, filters) or any_gpu_client(*node.outputs):
            gpu_filter_acts = GpuFilterActs(
                    module_stride=node.op.module_stride,
                    partial_sum=1)
            return [host_from_gpu(gpu_filter_acts(
                gpu_from_host(images),
                gpu_from_host(filters)))]

class GpuWeightActs(Base):
    """
    .. todo::

        WRITEME
    """
    def make_node(self, images, hidacts, frows, fcols):
        """
        .. todo::

            WRITEME
        """
        if self.partial_sum != 1:
            # this corresponds to grad when doing convolution
            raise NotImplementedError('partial sum')
        frows = theano.tensor.as_tensor_variable(frows)
        fcols = theano.tensor.as_tensor_variable(fcols)
        if frows.dtype[:3] not in ('int', 'uin'):
            raise TypeError(frows)
        if fcols.dtype[:3] not in ('int', 'uin'):
            raise TypeError(frows)
        if frows.ndim:
            raise TypeError('frows should be scalar', frows)
        if fcols.ndim:
            raise TypeError('fcols should be scalar', fcols)

        igroups, icolors, irows, icols, icount = images.type.broadcastable
        hgroups, hcolors, hrows, hcols, hcount = hidacts.type.broadcastable
        otype = theano.sandbox.cuda.CudaNdarrayType(
                broadcastable=(hrows, hcols, icolors,
                    False, False, hgroups, hcolors))
        return theano.Apply(self,
                [images, hidacts, frows, fcols],
                [otype()])

    def c_support_code(self):
        """
        .. todo::

            WRITEME
        """
        cufile = open(os.path.join(_this_dir, 'weight_acts.cu'))
        return cufile.read()

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        return ()

    def c_code(self, node, nodename, inames, onames, sub):
        """
        .. todo::

            WRITEME
        """
        images, hidacts, frows, fcols = inames
        dweights, = onames
        fail = sub['fail']
        moduleStride = str(self.module_stride)

        sio = StringIO.StringIO()

        print >> sio, """

        if (!CudaNdarray_is_c_contiguous(%(images)s))
        {
            //XXX: Alex's code actually supports the rightmost images
            //     dimension strided
            PyErr_Format(PyExc_NotImplementedError,
                "images not c contiguous");
            %(fail)s;
        }

        if (!CudaNdarray_is_c_contiguous(%(hidacts)s))
        {
            PyErr_Format(PyExc_NotImplementedError,
                "hidacts not c contiguous");
            %(fail)s;
        }

        if (%(images)s->nd != 5)
        {
            PyErr_Format(PyExc_TypeError,
                "images ndim (%%i) must be 5",
                %(images)s->nd);
            %(fail)s;
        }

        if (%(hidacts)s->nd != 5)
        {
            PyErr_Format(PyExc_TypeError,
                "hidacts ndim (%%i) must be 5",
                %(images)s->nd);
            %(fail)s;
        }

        if (PyArray_NDIM(%(frows)s) != 0)
        {
            PyErr_Format(PyExc_TypeError,
                "frows ndim (%%i) must be 0",
                PyArray_NDIM(%(frows)s));
            %(fail)s;
        }

        if (PyArray_NDIM(%(fcols)s) != 0)
        {
            PyErr_Format(PyExc_TypeError,
                "fcols ndim (%%i) must be 0",
                PyArray_NDIM(%(fcols)s));
            %(fail)s;
        }

        { // new scope, new vars

            int igroups           = CudaNdarray_HOST_DIMS(%(images)s)[0];
            int icolors_per_group = CudaNdarray_HOST_DIMS(%(images)s)[1];
            int irows             = CudaNdarray_HOST_DIMS(%(images)s)[2];
            int icols             = CudaNdarray_HOST_DIMS(%(images)s)[3];
            int icount            = CudaNdarray_HOST_DIMS(%(images)s)[4];

            int hgroups           = CudaNdarray_HOST_DIMS(%(hidacts)s)[0];
            int hcolors_per_group = CudaNdarray_HOST_DIMS(%(hidacts)s)[1];
            int hrows             = CudaNdarray_HOST_DIMS(%(hidacts)s)[2];
            int hcols             = CudaNdarray_HOST_DIMS(%(hidacts)s)[3];
            int hcount            = CudaNdarray_HOST_DIMS(%(hidacts)s)[4];

            int fmodulesR = hrows;
            int fmodulesC = hcols;
            int fcolors = icolors_per_group;
            int frows = ((dtype_%(frows)s *) PyArray_DATA(%(frows)s))[0];
            int fcols = ((dtype_%(fcols)s *) PyArray_DATA(%(fcols)s))[0];
            int fgroups = hgroups;
            int filters_per_group = hcolors_per_group;

            // XXX: use this parameter properly
            int paddingStart = 0;
            int imgStride = icount;
            float scaleTargets = 0.0;
            float scaleOutput = 1.0;
            int moduleStride = %(moduleStride)s;
            int partialSum = 1; // set to 0 for convolution.

            if (igroups != hgroups)
            {
                PyErr_Format(PyExc_ValueError,
                    "igroups != hgroups (%%i != %%i)",
                    igroups, hgroups);
                %(fail)s;
            }

            if (icolors_per_group != fcolors)
            {
                PyErr_Format(PyExc_ValueError,
                    "icolors_per_group != fcolors (%%i != %%i)",
                    icolors_per_group,
                    fcolors);
                %(fail)s;
            }

            if (icount != hcount)
            {
                PyErr_Format(PyExc_ValueError,
                    "icount != hcount (%%i != %%i)",
                    icount,
                    hcount);
                %(fail)s;
            }

            // XXX: CHECK SHAPE IS CORRECT
            if (!%(dweights)s)
            {
                Py_XDECREF(%(dweights)s);
                int dims[7];
                dims[0] = fmodulesR;
                dims[1] = fmodulesC;
                dims[2] = fcolors;
                dims[3] = frows;
                dims[4] = fcols;
                dims[5] = fgroups;
                dims[6] = filters_per_group;

                %(dweights)s = (CudaNdarray*)CudaNdarray_NewDims(7, dims);
                if (!%(dweights)s)
                {
                    %(fail)s;
                }
            }

            assert(CudaNdarray_is_c_contiguous(%(dweights)s));

            if (_weightActs(
                    igroups,
                    icolors_per_group,
                    irows,
                    icols,
                    icount,
                    fmodulesR,
                    fmodulesC,
                    frows,
                    fcols,
                    filters_per_group,
                    CudaNdarray_DEV_DATA(%(images)s),
                    CudaNdarray_DEV_DATA(%(hidacts)s),
                    CudaNdarray_DEV_DATA(%(dweights)s),
                    paddingStart,
                    moduleStride,
                    imgStride,
                    scaleTargets,
                    scaleOutput,
                    partialSum))
            {
                %(fail)s;
            }
        } // end bogus scope used for vars

        """

        return sio.getvalue() % locals()


@register_opt()
@local_optimizer([WeightActs])
def insert_gpu_weight_acts(node):
    """
    .. todo::

        WRITEME
    """
    if isinstance(node.op, WeightActs):
        """
        .. todo::

            WRITEME
        """
        images, hidacts, frows, fcols = node.inputs
        if any_from_gpu(images, hidacts) or any_gpu_client(*node.outputs):
            gpu_weight_acts = GpuWeightActs(
                    module_stride=node.op.module_stride,
                    partial_sum=1)
            return [host_from_gpu(gpu_weight_acts(
                gpu_from_host(images),
                gpu_from_host(hidacts),
                frows,
                fcols,
                ))]


class GpuImgActs(Base):
    """
    .. todo::

        WRITEME
    """
    def make_node(self, filters, hidacts, irows, icols):
        """
        .. todo::

            WRITEME
        """
        irows = theano.tensor.as_tensor_variable(irows)
        icols = theano.tensor.as_tensor_variable(icols)
        if irows.dtype[:3] not in ('int', 'uin'):
            raise TypeError(irows)
        if icols.dtype[:3] not in ('int', 'uin'):
            raise TypeError(irows)
        if irows.ndim:
            raise TypeError('irows should be scalar', irows)
        if icols.ndim:
            raise TypeError('icols should be scalar', icols)
        return theano.gof.Apply(self,
                [filters, hidacts, irows, icols],
                [hidacts.type()])


    def c_support_code(self):
        """
        .. todo::

            WRITEME
        """
        cufile = open(os.path.join(_this_dir, 'raw_img_acts.cu'))
        return cufile.read()

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        return ()

    def c_code(self, node, nodename, inames, onames, sub):
        """
        .. todo::

            WRITEME
        """
        filters, hidacts, irows, icols = inames
        dimages, = onames
        fail = sub['fail']
        moduleStride = str(self.module_stride)

        sio = StringIO.StringIO()

        print >> sio, """

        if (!CudaNdarray_is_c_contiguous(%(filters)s))
        {
            //XXX: Alex's code actually supports the rightmost images
            //     dimension strided
            PyErr_Format(PyExc_NotImplementedError,
                "images not c contiguous");
            %(fail)s;
        }

        if (!CudaNdarray_is_c_contiguous(%(hidacts)s))
        {
            PyErr_Format(PyExc_NotImplementedError,
                "hidacts not c contiguous");
            %(fail)s;
        }

        if (%(filters)s->nd != 7)
        {
            PyErr_Format(PyExc_TypeError,
                "images ndim (%%i) must be 7",
                %(filters)s->nd);
            %(fail)s;
        }

        if (%(hidacts)s->nd != 5)
        {
            PyErr_Format(PyExc_TypeError,
                "hidacts ndim (%%i) must be 5",
                %(hidacts)s->nd);
            %(fail)s;
        }

        if (PyArray_NDIM(%(irows)s) != 0)
        {
            PyErr_Format(PyExc_TypeError,
                "frows ndim (%%i) must be 0",
                PyArray_NDIM(%(irows)s));
            %(fail)s;
        }

        if (PyArray_NDIM(%(icols)s) != 0)
        {
            PyErr_Format(PyExc_TypeError,
                "fcols ndim (%%i) must be 0",
                PyArray_NDIM(%(icols)s));
            %(fail)s;
        }

        { // new scope, new vars

            int fmodulesR         = CudaNdarray_HOST_DIMS(%(filters)s)[0];
            int fmodulesC         = CudaNdarray_HOST_DIMS(%(filters)s)[1];
            int fcolors           = CudaNdarray_HOST_DIMS(%(filters)s)[2];
            int frows             = CudaNdarray_HOST_DIMS(%(filters)s)[3];
            int fcols             = CudaNdarray_HOST_DIMS(%(filters)s)[4];
            int fgroups           = CudaNdarray_HOST_DIMS(%(filters)s)[5];
            int filters_per_group = CudaNdarray_HOST_DIMS(%(filters)s)[6];

            int hgroups           = CudaNdarray_HOST_DIMS(%(hidacts)s)[0];
            int hcolors_per_group = CudaNdarray_HOST_DIMS(%(hidacts)s)[1];
            int hrows             = CudaNdarray_HOST_DIMS(%(hidacts)s)[2];
            int hcols             = CudaNdarray_HOST_DIMS(%(hidacts)s)[3];
            int hcount            = CudaNdarray_HOST_DIMS(%(hidacts)s)[4];

            int igroups           = fgroups;
            int icolors_per_group = fcolors;
            int irows             = ((dtype_%(irows)s *) PyArray_DATA(%(irows)s))[0];
            int icols             = ((dtype_%(icols)s *) PyArray_DATA(%(icols)s))[0];
            int icount            = hcount;


            // TODO: use this parameter properly
            int paddingStart = 0;
            float scaleTargets = 0.0;
            float scaleOutput = 1.0;
            int moduleStride = %(moduleStride)s;
            bool conv = 0;

            if (hgroups != fgroups)
            {
                PyErr_Format(PyExc_ValueError,
                    "hgroups != fgroups (%%i != %%i)",
                    hgroups, fgroups);
                %(fail)s;
            }

            if (hcolors_per_group != filters_per_group)
            {
                PyErr_Format(PyExc_ValueError,
                    "hcolors_per_group != filters_per_group (%%i != %%i)",
                    hcolors_per_group,
                    filters_per_group);
                %(fail)s;
            }

            // XXX: CHECK SHAPE IS CORRECT
            if (!%(dimages)s)
            {
                Py_XDECREF(%(dimages)s);
                int dims[5];
                dims[0] = igroups;
                dims[1] = icolors_per_group;
                dims[2] = irows;
                dims[3] = icols;
                dims[4] = icount;

                %(dimages)s = (CudaNdarray*)CudaNdarray_NewDims(5, dims);
                if (!%(dimages)s)
                {
                    %(fail)s;
                }
            }

            assert(CudaNdarray_is_c_contiguous(%(dimages)s));

            if (paddingStart + (fmodulesR - 1) * moduleStride + frows < irows)
            {
                PyErr_Format(PyExc_ValueError,
                    "uhoh123: %%i %%i %%i %%i %%i",
                    paddingStart,
                    fmodulesR,
                    moduleStride,
                    frows,
                    irows);
                %(fail)s;
            }

            if (_imgActs(
                    fgroups,
                    filters_per_group,
                    fcolors,
                    hcount,
                    fmodulesR,
                    fmodulesC,
                    frows,
                    fcols,
                    irows,
                    icols,
                    CudaNdarray_DEV_DATA(%(filters)s),
                    CudaNdarray_DEV_DATA(%(hidacts)s),
                    CudaNdarray_DEV_DATA(%(dimages)s),
                    paddingStart,
                    moduleStride,
                    scaleTargets,
                    scaleOutput,
                    conv))
            {
                %(fail)s;
            }
        } // end bogus scope used for vars

        """

        return sio.getvalue() % locals()



@register_opt()
@local_optimizer([ImgActs])
def insert_gpu_img_acts(node):
    """
    .. todo::

        WRITEME
    """
    if isinstance(node.op, ImgActs):
        filters, hidacts, irows, icols = node.inputs
        if any_from_gpu(filters, hidacts) or any_gpu_client(*node.outputs):
            gpu_img_acts = GpuImgActs(
                    module_stride=node.op.module_stride,
                    partial_sum=1)
            return [host_from_gpu(gpu_img_acts(
                gpu_from_host(filters),
                gpu_from_host(hidacts),
                irows,
                icols,
                ))]
