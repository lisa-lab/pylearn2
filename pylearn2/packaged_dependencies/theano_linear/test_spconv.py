from theano.compat.six.moves import xrange
activate_test_spconv = 0
if activate_test_spconv:
    import sys
    from theano import function, Mode
    from theano.gof import OpWiseCLinker
    import theano, numpy
    import theano.tensor as T
    import theano.sparse
    import scipy.sparse

    from scipy.signal import convolve2d
    import scipy.sparse as sparse
    import numpy
    import numpy as N
    #from theano.sparse.sandbox import spconv as sp

    import unittest
    import time
    sp = None

    def test_convolution():
        print('\n\n*************************************************')
        print('           TEST CONVOLUTION')
        print('*************************************************')

        # fixed parameters
        channels=3
        bsize = 10     # batch size
        imshp = (32,32)
        kshp = (8,8)
        nkern = 32
        subsample_amounts = ((1,1),(2,2),(3,3),(4,4))
        convmodes = ('full','valid')

        ishp4_channel_major = (bsize, channels) + imshp
        kshp4_channel_major = (nkern, channels) + kshp
        ishp4_channel_minor = (bsize,) + imshp + (channels,)
        kshp4_channel_minor = (nkern,) + kshp + (channels,)

        # symbolic stuff
        kerns = T.tensor4()
        imgs = T.tensor4()
        rng = N.random.RandomState(3423489)
        kern_data = rng.rand(*kshp4_channel_major).astype(kerns.dtype)+1
        img_data = rng.rand(*ishp4_channel_major).astype(imgs.dtype)+1

        # re-arrange these random-images so that the channel data is the minor
        # dimension: (batch rows cols channels)
        kern_data_minor = kern_data.transpose([0,2,3,1]).copy()
        img_data_minor = img_data.transpose([0,2,3,1]).copy()

        assert img_data_minor.shape == (bsize,)+imshp + (channels,)

        for conv_mode in convmodes:
            for subsample in subsample_amounts:
                #print 'Subsample', subsample,
                de_output = theano.tensor.nnet.conv2d(imgs, kerns,
                        ishp4_channel_major,
                        kshp4_channel_major,
                        border_mode=conv_mode,
                        subsample=subsample)

                f_d = function([kerns, imgs], de_output, profile='DENSE')

                t0 = time.time()
                for i in range(5):
                    rval_d = f_d(kern_data, img_data)
                t_d = time.time() - t0
                #print "Conv2D", t_d,
                use_channel_major_ordering = 0
                if use_channel_major_ordering: # sparse with channel_major ordering
                    sp_output, outshp  = sp.conv2d(imgs, kerns,
                            ishp4_channel_major,
                            kshp4_channel_major,
                            subsample=subsample,
                            border_mode=conv_mode)
                    f_s = function([kerns, imgs], sp_output,
                            profile='MAJOR')

                    t0 = time.time()
                    for i in range(5):
                        rval_s = f_s(kern_data, img_data)
                        assert rval_s.size == rval_d.size, (rval_s.shape, rval_d.shape)

                    # put rval_s into channel-submajor format
                    rval_s_major = rval_s.transpose([0,3,1,2])
                    assert numpy.allclose(rval_s_major, rval_d)
                    t_s_major = time.time() - t0
                    #print "spconv_major", t_s_major, 'ratio', t_d / t_s_major

                use_channel_minor_ordering = 1
                if use_channel_minor_ordering: # sparse with channel_minor ordering
                    sp_output, outshp  = sp.conv2d_channel_minor(imgs, kerns,
                            ishp4_channel_minor,
                            kshp4_channel_minor,
                            subsample=subsample,
                            border_mode=conv_mode)
                    f_s = function([kerns, imgs], sp_output,
                            profile='MINOR')

                    t0 = time.time()
                    for i in range(5):
                        rval_s = f_s(kern_data_minor, img_data_minor)
                        assert rval_s.size == rval_d.size, (rval_s.shape, rval_d.shape)

                    # put rval_s into channel-submajor format
                    rval_s_major = rval_s.transpose([0,3,1,2])
                    assert rval_s_major.shape == rval_d.shape
                    assert numpy.allclose(rval_s_major, rval_d)
                    t_s_minor = time.time() - t0
                    #print "spconv_minor", t_s_minor, 'ratio', t_d / t_s_minor
                    #assert rval_d.shape == rval_s.shape

    def test_sparse():

        print('\n\n*************************************************')
        print('           TEST SPARSE')
        print('*************************************************')

        # fixed parameters
        bsize = 10     # batch size
        imshp = (28,28)
        kshp = (5,5)
        nkern = 1 # per output pixel
        ssizes = ((1,1),(2,2))
        convmodes = ('full','valid',)

        # symbolic stuff
        bias = T.dvector()
        kerns = T.dvector()
        input = T.dmatrix()
        rng = N.random.RandomState(3423489)

        import theano.gof as gof
        #Mode(optimizer='fast_run', linker=gof.OpWiseCLinker(allow_gc=False)),):
        ntot, ttot = 0,0
        for conv_mode in convmodes:
            for ss in ssizes:

                output, outshp = sp.applySparseFilter(kerns, kshp,\
                        nkern, input, imshp, ss, bias=bias, mode=conv_mode)
                f = function([kerns, bias, input], output)

                # build actual input images
                img2d = N.arange(bsize*N.prod(imshp)).reshape((bsize,)+imshp)
                img1d = img2d.reshape(bsize,-1)
                zeropad_img = N.zeros((bsize,\
                                       img2d.shape[1]+2*(kshp[0]-1),\
                                       img2d.shape[2]+2*(kshp[1]-1)))
                zeropad_img[:, kshp[0]-1:kshp[0]-1+img2d.shape[1],
                               kshp[1]-1:kshp[1]-1+img2d.shape[2]] = img2d

                # build kernel matrix -- flatten it for theano stuff
                filters = N.arange(N.prod(outshp)*N.prod(kshp)).\
                            reshape(nkern,N.prod(outshp[1:]),N.prod(kshp))
                spfilt = filters.flatten()
                biasvals = N.arange(N.prod(outshp))

                # compute output by hand
                ntime1 = time.time()
                refout = N.zeros((bsize,nkern,outshp[1],outshp[2]))
                patch = N.zeros((kshp[0],kshp[1]))
                for b in xrange(bsize):
                    for k in xrange(nkern):
                        pixi = 0 # pixel index in raster order
                        for j in xrange(outshp[1]):
                            for i in xrange(outshp[2]):
                                n = j * ss[0]
                                m = i * ss[1]
                                patch = zeropad_img[b,n:n+kshp[0],m:m+kshp[1]]
                                refout[b,k,j,i] = N.dot(filters[k,pixi,:],\
                                                        patch.flatten())
                                pixi += 1
                refout = refout.reshape(bsize,-1) + biasvals
                ntot += time.time() - ntime1
                # need to flatten images
                ttime1 = time.time()
                out1 = f(spfilt, biasvals, img1d)
                ttot += time.time() - ttime1
                temp = refout - out1
                assert (temp < 1e-10).all()
                # test downward propagation
                vis = T.grad(output, input, output)
                downprop = function([kerns,output], vis)
                temp1 = time.time()
                for zz in range(100):
                    visval = downprop(spfilt,out1)
                indices, indptr, spmat_shape, sptype, outshp, kmap = \
                        sp.convolution_indices.sparse_eval(imshp,kshp,nkern,ss,conv_mode)
                spmat = sparse.csc_matrix((spfilt[kmap],indices,indptr),spmat_shape)
                visref = N.dot(out1,spmat.todense())
                assert N.all(visref==visval)

            print('**** Sparse Profiling Results ****')
            print('Numpy processing time: ', ntot)
            print('Theano processing time: ', ttot)
        #profmode.print_summary()


    def test_maxpool():
        # generate flatted images
        maxpoolshps = ((2,2),(3,3),(4,4),(5,5),(6,6))
        imval = N.random.rand(4,5,10,10)

        images = T.dmatrix()
        for maxpoolshp in maxpoolshps:

            # symbolic stuff
            output, outshp = sp.max_pool(images, imval.shape[1:], maxpoolshp)
            f = function([images,],[output,])
            output_val = f(imval.reshape(imval.shape[0],-1))

            # numeric verification
            my_output_val = N.zeros((imval.shape[0], imval.shape[1],
                                     imval.shape[2]/maxpoolshp[0],
                                     imval.shape[3]/maxpoolshp[1]))
            assert N.prod(my_output_val.shape[1:]) == N.prod(N.r_[imval.shape[1],outshp])

            for n in range(imval.shape[0]):
                for k in range(imval.shape[1]):
                    for i in range(imval.shape[2]/maxpoolshp[0]):
                        for j in range(imval.shape[3]/maxpoolshp[1]):
                            ii,jj = i*maxpoolshp[0], j*maxpoolshp[1]
                            patch = imval[n,k,ii:ii+maxpoolshp[0],jj:jj+maxpoolshp[1]]
                            my_output_val[n,k,i,j] = N.max(patch)
            my_output_val = my_output_val.reshape(imval.shape[0],-1)
            assert N.all(output_val == my_output_val)

            def mp(input):
                output, outshp = sp.max_pool(input, imval.shape[1:], maxpoolshp)
                return output
            T.verify_grad(None, mp, [imval.reshape(imval.shape[0],-1)])


