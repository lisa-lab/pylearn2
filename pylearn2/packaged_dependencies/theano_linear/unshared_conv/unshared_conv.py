"""
XXX
"""

import numpy
import theano
import StringIO

def any_symbolic(*args):
    """
    Return True iff any a in `args` is a theano Variable
    """
    for a in args:
        if isinstance(a, theano.Variable):
            return True
    return False

def not_symbolic(*args):
    return not any_symbolic(*args)


class Base(theano.Op):
    def __init__(self,
            module_stride=1,
            ):
        self.module_stride = module_stride

    def _attributes(self):
        return (
                self.module_stride,
                )

    def __eq__(self, other):
        return (type(self) == type(other)
                and self._attributes() == other._attributes())

    def __hash__(self):
        return hash((type(self), self._attributes()))

    def __str__(self):
        return '%s{module_stride=%i}' % (
                self.__class__.__name__,
                self.module_stride,
                )


class FilterActs(Base):
    """
    Images of shape: colors x
    Filters are of shape:
        channels
    """

    def make_node(self, images, filters):
        images = theano.tensor.as_tensor_variable(images)
        filters = theano.tensor.as_tensor_variable(filters)
        ibcast = images.broadcastable
        fbcast = filters.broadcastable
        igroups, icolors_per_group, irows, icols, icount = ibcast
        fmodulesR, fmodulesC, fcolors, frows, fcols = fbcast[:-2]
        fgroups, filters_per_group = fbcast[-2:]
        hbcast = (fgroups, filters_per_group, fmodulesR, fmodulesC, icount)
        htype = theano.tensor.TensorType(
                dtype=images.dtype,
                broadcastable=hbcast)
        if images.dtype != filters.dtype:
            raise TypeError('dtype mismatch', (images, filters))
        return theano.gof.Apply(self,
                [images, filters],
                [htype()])

    def perform(self, node, iargs, ostor):
        images, filters = iargs

        igroups, icolors_per_group, irows, icols, icount = images.shape
        fmodulesR, fmodulesC, fcolors, frows, fcols = filters.shape[:-2]
        fgroups, filters_per_group = filters.shape[-2:]

        hshape = self.infer_shape(node, (images.shape, filters.shape))[0]

        hidacts = numpy.zeros(hshape, dtype=images.dtype)

        for mR in xrange(fmodulesR):
            for mC in xrange(fmodulesC):
                for gg in xrange(igroups):
                    img_r_offset = mR * self.module_stride
                    img_c_offset = mC * self.module_stride
                    rc_images = images[gg, :,
                            img_r_offset:img_r_offset + frows,
                            img_c_offset:img_c_offset + fcols,
                            :]
                    rc_filters = filters[mR, mC, :, :, :, gg, :]
                    # rc_images are fcolors x frows x fcols x count
                    # rc_filters are fcolors x frows x fcols x fpg
                    rc_hidacts = numpy.dot(
                        rc_filters.reshape(-1, filters_per_group).T,
                        rc_images.reshape(-1, icount))
                    hidacts[gg, :, mR, mC, :] = rc_hidacts
        ostor[0][0] = hidacts

        if 0:
            print 'FilterActs shapes: images', images.shape
            print 'FilterActs shapes: filters', filters.shape
            print 'FilterActs shapes: hidacts', hidacts.shape

    def grad(self, inputs, goutputs):
        images, filters = inputs
        frows = filters.shape[3]
        fcols = filters.shape[4]
        irows = images.shape[2]
        icols = images.shape[3]
        gimages = ImgActs(module_stride=self.module_stride)(
                filters, goutputs[0], irows, icols)
        gfilters = WeightActs(module_stride=self.module_stride)(
                images, goutputs[0], frows, fcols)
        return [gimages, gfilters]

    def infer_shape(self, node, shapes):
        ishape, fshape = shapes

        igroups, icolors_per_group, irows, icols, icount = ishape
        fmodulesR, fmodulesC, fcolors, frows, fcols = fshape[:-2]
        fgroups, filters_per_group = fshape[-2:]

        if not any_symbolic(irows, icols) and irows != icols:
            raise ValueError("non-square image argument",
                    (irows, icols))
        if not any_symbolic(frows, fcols) and frows != fcols:
            raise ValueError("non-square filter shape",
                    (frows, fcols))
        if not any_symbolic(fmodulesR, fmodulesC) and fmodulesR != fmodulesC:
            raise ValueError('non-square filter grouping',
                    (fmodulesR, fmodulesC))
        if (not any_symbolic(icolors_per_group, fcolors)
                and icolors_per_group != fcolors):
            raise ValueError("color counts don't match",
                    (icolors_per_group, fcolors))

        hshape = (fgroups, filters_per_group, fmodulesR, fmodulesC, icount)
        return [hshape]


class WeightActs(Base):
    """
    Images of shape: colors x
    Filters are of shape:
        channels
    """

    def make_node(self, images, hidacts, frows, fcols):
        images, hidacts, frows, fcols = map(theano.tensor.as_tensor_variable,
                [images, hidacts, frows, fcols])
        if frows.dtype[:3] not in ('int', 'uin'):
            raise TypeError(frows)
        if fcols.dtype[:3] not in ('int', 'uin'):
            raise TypeError(frows)
        if frows.ndim:
            raise TypeError('frows should be scalar', frows)
        if fcols.ndim:
            raise TypeError('fcols should be scalar', fcols)

        if images.dtype != hidacts.dtype:
            raise TypeError('images and hidacts dtype mismatch',
                    (images.dtype, hidacts.dtype))

        igroups, icolors, irows, icols, icount = images.type.broadcastable
        hgroups, hcolors, hrows, hcols, hcount = hidacts.type.broadcastable
        otype = theano.tensor.TensorType(
                dtype=images.dtype,
                broadcastable=(hrows, hcols, icolors,
                    False, False, hgroups, hcolors))
        return theano.Apply(self,
                [images, hidacts, frows, fcols],
                [otype()])

    def perform(self, node, iargs, ostor):
        images, hidacts, frows, fcols = iargs

        if frows != fcols:
            # this could be implemented, but GPU case doesn't do it
            raise NotImplementedError("non-square filter shape",
                    (frows, fcols))

        #igroups, icolors_per_group, irows, icols, icount = images.shape
        hgroups, hcolors_per_group, hrows, hcols, hcount = hidacts.shape
        fshape = list(self.infer_shape(node,
                (images.shape, hidacts.shape, (), ()))[0])
        fcolors = fshape[2]
        fshape[3] = frows
        fshape[4] = fcols

        filters = numpy.zeros(fshape, dtype=images.dtype)

        for mR in xrange(hrows):
            for mC in xrange(hcols):
                for gg in xrange(hgroups):
                    img_r_offset = mR * self.module_stride
                    img_c_offset = mC * self.module_stride
                    rc_images = images[gg, :,
                            img_r_offset:img_r_offset + frows,
                            img_c_offset:img_c_offset + fcols,
                            :]
                    # rc_images is fcolors x frows x fcols x count

                    rc_hidacts = hidacts[gg, :, mR, mC, :]
                    # rc_hidacts is fpg x count

                    rc_filters = numpy.dot(
                            rc_images.reshape(-1, hcount),
                            rc_hidacts.T)
                    filters[mR, mC, :, :, :, gg, :] = rc_filters.reshape(
                            (fcolors, frows, fcols, hcolors_per_group))
        ostor[0][0] = filters

    def grad(self, inputs, goutputs):
        images, hidacts, frows, fcols = inputs
        gfilters, = goutputs
        irows = images.shape[2]
        icols = images.shape[3]
        gimages = ImgActs(module_stride=self.module_stride)(
                gfilters, hidacts, irows, icols)
        ghidacts = FilterActs(module_stride=self.module_stride)(
                images, gfilters)
        return [gimages, ghidacts, None, None]

    def infer_shape(self, node, shapes):
        images, hidacts, frows, fcols = node.inputs
        ishape, hshape, frowshp, fcolshp = shapes
        igroups, icolors_per_group, irows, icols, icount = ishape
        hgroups, hcolors_per_group, hrows, hcols, hcount = hshape

        fmodulesR = hrows
        fmodulesC = hcols
        fcolors = icolors_per_group
        # frows already assigned
        # fcols already assigned
        fgroups = hgroups
        filters_per_group = hcolors_per_group

        fshape = (fmodulesR, fmodulesC, fcolors, frows, fcols, fgroups,
                    filters_per_group)

        if not_symbolic(irows, icols) and irows != icols:
            raise NotImplementedError("non-square image argument",
                    (irows, icols))
        if not_symbolic(hrows, hcols) and hrows != hcols:
            raise NotImplementedError("non-square filter shape",
                    (hrows, hcols))
        if not_symbolic(icount, hcount) and icount != hcount:
            raise NotImplementedError("different number of images",
                    (icount, hcount))
        if not_symbolic(igroups, hgroups) and igroups != hgroups:
            raise ValueError("hgroups must match igroups",
                    igroups, hgroups)

        return [fshape]


class ImgActs(Base):
    """
    XXX
    """
    def make_node(self, filters, hidacts, irows, icols):
        filters, hidacts, irows, icols = map(theano.tensor.as_tensor_variable,
                [filters, hidacts, irows, icols])
        if irows.dtype[:3] not in ('int', 'uin'):
            raise TypeError(irows)
        if icols.dtype[:3] not in ('int', 'uin'):
            raise TypeError(irows)
        if irows.ndim:
            raise TypeError('irows should be scalar', irows)
        if icols.ndim:
            raise TypeError('icols should be scalar', icols)
        if filters.ndim != 7:
            raise TypeError('filters must be 7d tensor', filters)
        if hidacts.ndim != 5:
            raise TypeError('hidacts must be 5d tensor', filters)
        if filters.dtype != hidacts.dtype:
            raise TypeError('filters and hidacts must have matching dtype',
                    (filters, hidacts))
        return theano.gof.Apply(self,
                [filters, hidacts, irows, icols],
                [hidacts.type()])

    def perform(self, node, iargs, ostor):
        filters, hidacts, irows, icols = iargs

        hgroups, hcolors_per_group, hrows, hcols, hcount = hidacts.shape

        fmodulesR, fmodulesC, fcolors, frows, fcols = filters.shape[:-2]
        fgroups, filters_per_group = filters.shape[-2:]

        igroups = fgroups
        icolors_per_group = fcolors
        icount = hcount

        #print 'IMGACTS: NODE OUTPUTS[0]'
        #print theano.printing.debugprint(node.outputs[0])
        #print 'FILTERS SHAPE:', filters.shape
        #print 'HIDACTS SHAPE:', hidacts.shape
        if hrows != hcols:
            raise NotImplementedError("non-square hidacts argument",
                    (hrows, hcols))
        if frows != fcols:
            raise NotImplementedError("non-square filter shape",
                    (frows, fcols))
        if fmodulesR != fmodulesC:
            raise NotImplementedError('non-square filter grouping',
                    (fmodulesR, fmodulesC))
        if hcolors_per_group != filters_per_group:
            raise ValueError("color counts don't match",
                    (hcolors_per_group, filters_per_group))
        if irows != icols:
            raise NotImplementedError("non-square image argument",
                    (irows, icols))

        images = numpy.zeros(
                (igroups, icolors_per_group, irows, icols, icount),
                dtype=hidacts.dtype)

        for mR in xrange(fmodulesR):
            for mC in xrange(fmodulesC):
                for gg in xrange(igroups):
                    rc_filters = filters[mR, mC, :, :, :, gg, :]
                    # rc_filters is fcolors x frows x fcols x fpg

                    rc_hidacts = hidacts[gg, :, mR, mC, :]
                    # rc_hidacts is fpg x icount

                    img_r_offset = mR * self.module_stride
                    img_c_offset = mC * self.module_stride
                    images[gg, :,
                            img_r_offset:img_r_offset + frows,
                            img_c_offset:img_c_offset + fcols,
                            :] += numpy.dot(
                                    rc_filters.reshape(-1, filters_per_group),
                                    rc_hidacts
                                    ).reshape(
                                    (fcolors, frows, fcols, icount))
        ostor[0][0] = images

    def grad(self, inputs, goutputs):
        filters, hidacts, irows, icols = inputs
        gimages, = goutputs
        frows = filters.shape[3]
        fcols = filters.shape[4]
        gfilters = WeightActs(module_stride=self.module_stride)(
                gimages, hidacts, frows, fcols)
        ghidacts = FilterActs(module_stride=self.module_stride)(
                gimages, filters)
        return [gfilters, ghidacts, None, None]


