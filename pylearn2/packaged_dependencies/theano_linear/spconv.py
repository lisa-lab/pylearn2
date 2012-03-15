"""
Convolution-like operations with sparse matrix multiplication.

To read about different sparse formats, see U{http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps}.

@todo: Automatic methods for determining best sparse format?
"""
#### COPIED FROM hpu/icml09/sp.py

import numpy
from scipy import sparse as scipy_sparse

import theano
import theano.sparse
from theano import sparse, gof, Op, tensor
from theano.printing import Print

raise ImportError("THIS OLD CODE'S TESTS ARE BIT-ROTTEN")

class RasterOrders(object):
    @staticmethod
    def row_col_channel(row, col, channel, n_rows, n_cols, n_channels):
        return row * n_cols * n_channels + col * n_channels + channel
    @staticmethod
    def channel_row_col(row, col, channel, n_rows, n_cols, n_channels):
        return channel * n_rows * n_cols + row * n_cols + col

def conv_out_shp(IR, IC, KR, KC, border_mode, subsample):
    ssR, ssC = subsample
    def ceildiv(x, y):
        r = x // y
        if r * y < x:
            return r + 1
        return r
    if border_mode == 'valid':
        OR, OC = ceildiv(IR - KR + 1,ssR), ceildiv(IC - KC + 1,ssC)
    elif border_mode == 'full':
        OR, OC = ceildiv(IR + KR - 1,ssR), ceildiv(IC + KC - 1,ssC)
    else:
        raise NotImplementedError(border_mode)
    return OR, OC

def sp_extract_patches(IR, IC, KR, KC, CH,
        input_raster_order,
        output_raster_order,
        subsample,
        border_mode,
        flip_patches):
    """
    Construct a sparse matrix such that multiplication with a rasterized image
    produces the concatenation of rasterized image patches.

    The original image is presumed to be in row-major, channel-minor order:
    R(0,0) G(0,0) B(0,0) R(0,1), G(0,1), B(0,1), ...

    """

    ssR, ssC = subsample
    OR, OC = conv_out_shp(IR, IC, KR, KC, border_mode, subsample)

    rval = scipy_sparse.lil_matrix((IR*IC*CH, OR*OC*KR*KC*CH))

    if not flip_patches:
        raise NotImplementedError()

    def in_pos(i, j, k):
        return input_raster_order(i, j, k, IR, IC, CH)
    def out_pos(i, j, k):
        return output_raster_order(i, j, k, KR, KC, CH)

    for orow in range(OR):
        for ocol in range(OC):
            for krow in range(KR):
                for kcol in range(KC):
                    assert flip_patches
                    if border_mode == 'valid':
                        irow = orow*ssR + KR - krow - 1
                        icol = ocol*ssC + KC - kcol - 1
                    else:
                        irow = orow*ssR - krow
                        icol = ocol*ssC - kcol
                    if (0 <= irow < IR) and (0 <= icol < IC):
                        for ch in range(CH):
                            t = out_pos(krow,kcol,ch)
                            T = KR * KC * CH
                            try:
                                i = in_pos(irow, icol, ch)
                                j = orow*OC*T + ocol*T + t
                                rval[i, j] = 1
                            except IndexError:
                                print rval.shape, i, j, IR, IC, KR, KC, OR, OC
                                raise
    return rval

def conv2d_channel_minor(images, kerns, ishp4, kshp4, subsample=(1,1),
             border_mode='valid'):
    # start by computing output dimensions, size, etc
    B, IR, IC, C = ishp4
    K, KR, KC, CH = kshp4
    assert C == CH # number of channels must match

    OR, OC = conv_out_shp(IR, IC, KR, KC, border_mode, subsample)
    oshp = (B, OR, OC, K)

    # construct indices and index pointers for sparse matrix, which, when multiplied
    # with input images will generate a stack of image patches
    patch_extractor = sp_extract_patches(IR, IC, KR, KC, CH,
            RasterOrders.row_col_channel,
            RasterOrders.row_col_channel,
            subsample,
            border_mode,
            flip_patches=True).tocsc()

    #print IR, IC, KR, KC, CH, patch_extractor.shape, patch_extractor.nnz
    patches = sparse.structured_dot(
            images.flatten(2),
            patch_extractor)

    # compute output of linear classifier
    patch_stack = patches.reshape((B*OR*OC, KR*KC*CH))

    # kern is of shape: nkern x ksize*number_of_input_features
    # output is thus of shape: bsize*outshp x nkern
    output = tensor.dot(patch_stack, kerns.flatten(2).T).reshape((B, OR, OC, K))

    return output, oshp

def conv2d(images, kerns, ishp4, kshp4, subsample=(1,1),
             border_mode='valid'):
    # start by computing output dimensions, size, etc
    B, C, IR, IC = ishp4
    K, CH, KR, KC = kshp4
    assert C == CH # number of channels must match

    OR, OC = conv_out_shp(IR, IC, KR, KC, border_mode, subsample)
    oshp = (B, OR, OC, K)

    # construct indices and index pointers for sparse matrix, which, when multiplied
    # with input images will generate a stack of image patches
    patch_extractor = sp_extract_patches(IR, IC, KR, KC, CH,
            RasterOrders.channel_row_col,
            RasterOrders.channel_row_col,
            subsample,
            border_mode,
            flip_patches=True).tocsc()

    #print IR, IC, KR, KC, CH, patch_extractor.shape, patch_extractor.nnz
    patches = sparse.structured_dot(
            images.flatten(2),
            patch_extractor)

    # compute output of linear classifier
    patch_stack = patches.reshape((B*OR*OC, KR*KC*CH))

    # kern is of shape: nkern x ksize*number_of_input_features
    # output is thus of shape: bsize*outshp x nkern
    output = tensor.dot(patch_stack, kerns.flatten(2).T).reshape((B, OR, OC, K))

    return output, oshp



def register_specialize(lopt, *tags, **kwargs):
    theano.compile.optdb['specialize'].register((kwargs and kwargs.pop('name')) or lopt.__name__, lopt, 'fast_run', *tags)


class Remove0(Op):
    """
    Remove explicit zeros from a sparse matrix, and resort indices
    """
    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self,node, (x,), (z,)):
        if x.format != 'csc':
            raise TypeError('Remove0 only works on csc matrices')

        M, N = x.shape

        data = x.data
        indices = x.indices
        indptr = x.indptr

        #TODO: try using ndarrays and then prune() on the result
        new_data = []
        new_indices = []
        new_indptr = [0]

        for j in xrange(0, N):
            for i_idx in xrange(indptr[j], indptr[j+1]):
                if data[i_idx] != 0:
                    new_data.append(data[i_idx])
                    new_indices.append(indices[i_idx])
            new_indptr.append(len(new_indices))

        z[0] = sparse.csc_matrix((new_data, new_indices, new_indptr), (M,N))

    def grad(self, (x,), (gz,)):
        return [gz]

remove0 = Remove0()

class EnsureSortedIndices(Op):
    """
    Remove explicit zeros from a sparse matrix, and resort indices
    """
    inplace=False

    def __init__(self, inplace):
        self.inplace=inplace
        if self.inplace:
            self.view_map = {0:[0]}

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self,node, (x,), (z,)):
        z[0] = x.ensure_sorted_indices(inplace=self.inplace)

    def grad(self, (x,), (gz,)):
        return [gz]

ensure_sorted_indices = EnsureSortedIndices(inplace=False)

def clean(x):
    return ensure_sorted_indices(remove0(x))


def max_pool(images, imgshp, maxpoolshp):
    """Implements a max pooling layer

    Takes as input a 2D tensor of shape batch_size x img_size and performs max pooling.
    Max pooling downsamples by taking the max value in a given area, here defined by
    maxpoolshp. Outputs a 2D tensor of shape batch_size x output_size.

    @param images: 2D tensor containing images on which to apply convolution.
                   Assumed to be of shape batch_size x img_size
    @param imgshp: tuple containing image dimensions
    @param maxpoolshp: tuple containing shape of area to max pool over

    @output out1: symbolic result (2D tensor)
    @output out2: logical shape of the output
    """
    N = numpy
    poolsize = N.int64(N.prod(maxpoolshp))

    # imgshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
    # in the first case, default nfeatures to 1
    if N.size(imgshp)==2:
        imgshp = (1,)+imgshp

    # construct indices and index pointers for sparse matrix, which, when multiplied
    # with input images will generate a stack of image patches
    indices, indptr, spmat_shape, sptype, outshp = \
            convolution_indices.conv_eval(imgshp, maxpoolshp, maxpoolshp, mode='valid')

    print 'XXXXXXXXXXXXXXXX MAX POOLING LAYER XXXXXXXXXXXXXXXXXXXX'
    print 'imgshp = ', imgshp
    print 'maxpoolshp = ', maxpoolshp
    print 'outshp = ', outshp

    # build sparse matrix, then generate stack of image patches
    csc = theano.sparse.CSM(sptype)(N.ones(indices.size), indices, indptr, spmat_shape)
    patches = sparse.structured_dot(csc, images.T).T

    pshape = tensor.stack(images.shape[0]*\
                            tensor.as_tensor(N.prod(outshp)),
                          tensor.as_tensor(imgshp[0]),
                          tensor.as_tensor(poolsize))
    patch_stack = tensor.reshape(patches, pshape, ndim=3);

    out1 = tensor.max(patch_stack, axis=2)

    pshape = tensor.stack(images.shape[0],
                          tensor.as_tensor(N.prod(outshp)),
                          tensor.as_tensor(imgshp[0]))
    out2 = tensor.reshape(out1, pshape, ndim=3);

    out3 = tensor.DimShuffle((False,)*3, (0,2,1))(out2)

    return tensor.flatten(out3,2), outshp
class ConvolutionIndices(Op):
    """This generates a sparse matrix M, which generates a stack of image patches
       when computing the dot product of M with image patch. Convolution is then
       simply the dot product of (img x M) and the kernels.
    """

    @staticmethod
    def sparse_eval(inshp, kshp, nkern, (dx,dy)=(1,1), mode='valid'):
        # STALE
        return convolution_indices.evaluate(inshp,kshp,(dx,dy),nkern,mode=mode,ws=False)

    @staticmethod
    def conv_eval(IR, IC, KR, KC, C, subsample=(1,1), mode='valid'):
        raise NotImplementedError('TODO: fix broken method')
        #return convolution_indices.evaluate(IR, IC, KR, KC, C, (dx,dy), mode=mode, ws=True)

    # img_shape and ker_shape are (height,width)
    @staticmethod
    def evaluate(imshp,kshp, (dx,dy)=(1,1), nkern=1, mode='valid', ws=True):
        """Build a sparse matrix which can be used for performing...
        * convolution: in this case, the dot product of this matrix with the input
          images will generate a stack of images patches. Convolution is then a
          tensordot operation of the filters and the patch stack.
        * sparse local connections: in this case, the sparse matrix allows us to operate
          the weight matrix as if it were fully-connected. The structured-dot with the
          input image gives the output for the following layer.

        @param ker_shape: shape of kernel to apply (smaller than image)
        @param img_shape: shape of input images
        @param mode: 'valid' generates output only when kernel and image overlap
            full' full convolution obtained by zero-padding the input
        @param ws: True if weight sharing, false otherwise
        @param (dx,dy): offset parameter. In the case of no weight sharing, gives the
            pixel offset between two receptive fields. With weight sharing gives the
            offset between the top-left pixels of the generated patches

        @rtype: tuple(indices, indptr, logical_shape, sp_type, out_img_shp)
        @returns: the structure of a sparse matrix, and the logical dimensions of the image
        which will be the result of filtering.
        """
        N = numpy

        # inshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
        # in the first case, default nfeatures to 1
        if N.size(imshp)==2:
            inshp = (1,)+imshp

        inshp = N.array(imshp)
        kshp  = N.array(kshp)
        ksize = N.prod(kshp)

        kern = ksize-1 - N.arange(ksize)

        # size of output image if doing proper convolution (mode='full',dx=dy=0)
        # outshp is the actual output shape given the parameters
        fulloutshp = inshp[1:] + kshp - 1
        s = -1 if mode=='valid' else 1
        outshp = N.int64(N.ceil((inshp[1:] + s*kshp - s*1) \
                 /N.array([dy,dx], dtype='float')))
        if any(outshp <= 0):
            err = 'Invalid kernel', kshp,'and/or step size',(dx,dy),\
                  'for given input shape', inshp
            raise ValueError(err)

        outsize = N.prod(outshp)
        insize = N.prod(inshp)

        # range of output units over which to iterate
        lbound = N.array([kshp[0]-1,kshp[1]-1]) if mode=='valid' else N.zeros(2)
        ubound = lbound + (inshp[1:]-kshp+1) if mode=='valid' else fulloutshp

        # coordinates of image in "fulloutshp" coordinates
        topleft  = N.array([kshp[0]-1,kshp[1]-1])
        botright = topleft + inshp[1:] # bound when counting the receptive field

        # sparse matrix specifics...
        spmatshp = (outsize*N.prod(kshp)*inshp[0],insize) if ws else\
                   (nkern*outsize,insize)
        spmat = scipy_sparse.lil_matrix(spmatshp)

        # loop over output image pixels
        z,zz = 0,0

        # incremented every time we write something to the sparse matrix
        # this is used to track the ordering of filter tap coefficient in sparse
        # column ordering
        tapi, ntaps = 0, 0

        # Note: looping over the number of kernels could've been done more efficiently
        # as the last step (when writing to spmat). However, this messes up the ordering
        # of the column values (order in which you write the values determines how the
        # vectorized data will get used later one)

        for fmapi in range(inshp[0]): # loop over input features
            for n in range(nkern): # loop over number of kernels (nkern=1 for weight sharing)

                # FOR EACH OUTPUT PIXEL...
                for oy in N.arange(lbound[0],ubound[0],dy): # loop over output image height
                    for ox in N.arange(lbound[1],ubound[1],dx): # loop over output image width

                        l = 0 # kern[l] is filter value to apply at (oj,oi) for (iy,ix)

                        # ... ITERATE OVER INPUT UNITS IN RECEPTIVE FIELD
                        for ky in oy+N.arange(kshp[0]):
                            for kx in ox+N.arange(kshp[1]):

                                # verify if we are still within image boundaries. Equivalent to
                                # zero-padding of the input image
                                if all((ky,kx) >= topleft) and all((ky,kx) < botright):

                                    # convert to "valid" input space coords
                                    # used to determine column index to write to in sparse mat
                                    iy,ix = N.array((ky,kx)) - topleft
                                    # determine raster-index of input pixel...
                                    col = iy*inshp[2]+ix +\
                                          fmapi*N.prod(inshp[1:]) # taking into account multiple input features

                                    # convert oy,ox values to output space coordinates
                                    (y,x) = (oy,ox) if mode=='full' else (oy,ox) - topleft
                                    (y,x) = N.array([y,x]) / (dy,dx) # taking into account step size
                                    # convert to row index of sparse matrix
                                    row = (y*outshp[1]+x)*inshp[0]*ksize + l + fmapi*ksize if ws else\
                                          y*outshp[1] + x

                                    # Store something at that location in sparse matrix.
                                    # The written value is only useful for the sparse case. It
                                    # will determine the way kernel taps are mapped onto
                                    # the sparse columns (idea of kernel map)
                                    spmat[row + n*outsize, col] = tapi + 1   # n*... only for sparse

                                    # total number of active taps (used for kmap)
                                    ntaps += 1

                                tapi += 1 # absolute tap index (total number of taps)
                                l+=1 # move on to next filter tap l=(l+1)%ksize

        if spmat.format != 'csc':
            spmat = spmat.tocsc().ensure_sorted_indices()
        else:
            # BUG ALERT: scipy0.6 has bug where data and indices are written in reverse column
            # ordering. Explicit call to ensure_sorted_indices removes this problem
            spmat = spmat.ensure_sorted_indices()

        if ws:
            kmap = None
        else:
            kmap = N.zeros(ntaps, dtype='int')
            k=0
            #print 'TEMPORARY BUGFIX: REMOVE !!!'
            for j in xrange(spmat.shape[1]):
                for i_idx in xrange(spmat.indptr[j], spmat.indptr[j+1]):
                    if spmat.data[i_idx] != 0:
                        kmap[k] = spmat.data[i_idx] -1 # this is == spmat[i,j] - 1
                        k+=1

        # when in valid mode, it is more efficient to store in sparse row
        # TODO: need to implement structured dot for csr matrix
        assert spmat.format == 'csc'
        sptype = 'csc'
        #sptype = 'csr' if mode=='valid' else 'csc'
        if 0 and mode=='valid':
            spmat = spmat.tocsr()

        rval = (spmat.indices[:spmat.size],
                spmat.indptr, spmatshp, sptype, outshp)
        rval += (kmap,) if kmap!=None else ()

        return rval

    def perform(self, node, (inshp, kshp),\
                (out_indices, out_indptr, spmat_shape)):
        indices, indptr, spmatshp, outshp = self.evaluate(inshp, kshp)
        out_indices[0] = indices
        out_indptr[0] = indptr
        spmat_shape[0] = numpy.asarray(spmatshp)

convolution_indices = ConvolutionIndices()

def applySparseFilter(kerns, kshp, nkern, images, imgshp, step=(1,1), bias=None, mode='valid'):
    """
     === Input / Output conventions===
    "images" is assumed to be a matrix of shape batch_size x img_size, where the second
    dimension represents each image in raster order

    Output feature map will have shape:
       batch_size x number of kernels * output_size
    IMPORTANT: note that this means that each feature map is contiguous in memory.
               The memory layout will therefore be:
               [ <feature_map_0> <feature_map_1> ... <feature_map_n>],
               where <feature_map> represents a "feature map" in raster order
    Note that the concept of feature map doesn't really apply to sparse filters without
    weight sharing. Basically, nkern=1 will generate one output img/feature map,
    nkern=2 a second feature map, etc.

    kerns is a 1D tensor, and assume to be of shape:
       nkern * N.prod(outshp) x N.prod(kshp)
    Each filter is applied seperately to consecutive output pixels.

    @param kerns: nkern*outsize*ksize vector containing kernels
    @param kshp: tuple containing actual dimensions of kernel (not symbolic)
    @param nkern: number of kernels to apply at each pixel in the input image.
                  nkern=1 will apply a single unique filter for each input pixel.
    @param images: bsize x imgsize matrix containing images on which to apply filters
    @param imgshp: tuple containing actual image dimensions (not symbolic)
    @param step: determines number of pixels between adjacent receptive fields
                 (tuple containing dx,dy values)
    @param mode: 'full', 'valid' see CSM.evaluate function for details
    @output out1: symbolic result
    @output out2: logical shape of the output img (nkern,height,width)
                  (after dot product, not of the sparse matrix!)
    """

    # inshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
    # in the first case, default nfeatures to 1
    if numpy.size(imgshp)==2:
        imgshp = (1,)+imgshp

    # construct indices and index pointers for sparse matrix
    indices, indptr, spmat_shape, sptype, outshp, kmap = \
        convolution_indices.sparse_eval(imgshp, kshp, nkern, step, mode)

    # build a sparse weight matrix
    sparsew = theano.sparse.CSM(sptype, kmap)(kerns, indices, indptr, spmat_shape)
    output =  sparse.structured_dot(sparsew, images.T).T
    if bias is not None:
        output += bias

    return output, numpy.hstack((nkern,outshp))



