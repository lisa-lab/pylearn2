"""

An implementation of probabilistic max-pooling, based on

TODO writme


"""

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import theano.tensor as T
import numpy as np
from theano import config
from theano import function
import time
from pylearn2.utils import sharedX
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams

def max_pool_python(z, pool_shape):

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    assert zr % r == 0
    assert zc % c == 0

    h = np.zeros(z.shape, dtype = z.dtype)
    p = np.zeros( (batch_size, zr /r, zc /c, ch), dtype = z.dtype)

    for u in xrange(0,zr,r):
        for l in xrange(0,zc,c):
            pt = np.exp(z[:,u:u+r,l:l+c,:])
            denom = pt.sum(axis=1).sum(axis=1) + 1.
            p[:,u/r,l/c,:] = 1. - 1. / denom
            for i in xrange(batch_size):
                for j in xrange(ch):
                    pt[i,:,:,j] /= denom[i,j]
            h[:,u:u+r,l:l+c,:] = pt

    return p, h


def max_pool_raw_graph(z, pool_shape):
    #random max pooling implemented with set_subtensor
    #could also do this using the stuff in theano.sandbox.neighbours
    #might want to benchmark the two approaches, see how each does on speed/memory
    #on cpu and gpu
    #this method is not numerically stable, use max_pool instead

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    zpart = []

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            zpart[i].append( z[:,i:zr:r,j:zc:c,:] )

    pt = []

    for i in xrange(r):
        pt.append( [ T.exp(z_ij) for z_ij in zpart[i] ] )

    denom = 1.

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]

    p = 1. - 1. / denom

    hpart = []
    for i in xrange(r):
        hpart.append( [ pt_ij / denom for pt_ij in pt[i] ] )

    h = T.alloc(0., batch_size, zr, zc, ch)

    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:,i:zr:r,j:zc:c,:],hpart[i][j])

    return p, h

def max_pool_stable_graph(z, pool_shape, top_down = None, theano_rng = None):
    """

    top_down: like z, but applied to the pooling units
              I think I handle things slightly different than Honglak does
              Need to check this, but I think he adds top_down to all the
              detector unit inputs.
              Instead, I subtract it from the energy of the "off" unit
    """
    #random max pooling implemented with set_subtensor
    #could also do this using the stuff in theano.sandbox.neighbours
    #might want to benchmark the two approaches, see how each does on speed/memory
    #on cpu and gpu
    #note: actually theano.sandbox.neighbours is probably a bad idea. it treats
    #the images as being one channel, and emits all channels and positions into
    #a 2D array. so I'd need to index each channel separately and join the channels
    #back together, with a reshape. I expect joining num_channels is more expensive
    #then incsubtensoring pool_rows*pool_cols, simply because we tend to have small
    #pooling regions and a lot of channels, but I guess this worth testing.
    #actually I might be able to do it fast with reshape-see galatea/cond/neighbs.py
    #however, at some point the grad for this was broken. check that calling grad
    #on images2neibs doesn't raise an exception before sinking too much time
    #into this.
    #here I stabilized the softplus with 4 calls to T.maximum and 5 elemwise
    #subs. this is 10% slower than the unstable version, and the gradient
    #is 40% slower. on GPU both the forward prop and backprop are more like
    #100% slower!
    #might want to dry doing a reshape, a T.nnet.softplus, and a reshape
    #instead
    #another way to implement the stabilization is with the max pooling operator
    #(you'd still need to do maximum with 0)


    #timing hack
    #return T.nnet.sigmoid(z[:,0:z.shape[1]/pool_shape[0],0:z.shape[2]/pool_shape[1],:]), T.nnet.sigmoid(z)

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    if top_down is None:
        t = 0.
    else:
        t = - top_down

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:,i:zr:r,j:zc:c,:]
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i,j)
            zpart[i].append( cur_part )
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,'+cur_part.name+')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max('+cur_part.name+','+mx.name+')'
                mx = T.maximum(mx,cur_part)
                mx.name = mx_name
    mx.name = 'local_max('+z_name+')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append( cur_pt )

    off_pt = T.exp(t - mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    off_prob = off_pt / denom
    p = 1. - off_prob
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append( [ pt_ij / denom for pt_ij in pt[i] ] )

    h = T.alloc(0., batch_size, zr, zc, ch)

    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:,i:zr:r,j:zc:c,:],hpart[i][j])

    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h
    else:
        events = []
        for i in xrange(r):
            for j in xrange(c):
                events.append(hpart[i][j])
        events.append(off_prob)

        events = [ event.dimshuffle(0,1,2,3,'x') for event in events ]

        events = tuple(events)

        stacked_events = T.concatenate( events, axis = 4)

        batch_size, rows, cols, channels, outcomes = stacked_events.shape
        reshaped_events = stacked_events.reshape((batch_size * rows * cols * channels, outcomes))

        multinomial = theano_rng.multinomial(pvals = reshaped_events, dtype = p.dtype)

        reshaped_multinomial = multinomial.reshape((batch_size, rows, cols, channels, outcomes))

        h_sample = T.alloc(0., batch_size, zr, zc, ch)

        idx = 0
        for i in xrange(r):
            for j in xrange(c):
                h_sample = T.set_subtensor(h_sample[:,i:zr:r,j:zc:c,:],
                        reshaped_multinomial[:,:,:,:,idx])
                idx += 1

        p_sample = 1 - reshaped_multinomial[:,:,:,:,-1]

        return p, h, p_sample, h_sample

def max_pool_softmax_with_bias_op(z, pool_shape):
    #random max pooling implemented with set_subtensor
    #could also do this using the stuff in theano.sandbox.neighbours
    #might want to benchmark the two approaches, see how each does on speed/memory
    #on cpu and gpu
    #note: actually theano.sandbox.neighbours is probably a bad idea. it treats
    #the images as being one channel, and emits all channels and positions into
    #a 2D array. so I'd need to index each channel separately and join the channels
    #back together, with a reshape. I expect joining num_channels is more expensive
    #then incsubtensoring pool_rows*pool_cols, simply because we tend to have small
    #pooling regions and a lot of channels, but I guess this worth testing.
    #actually I might be able to do it fast with reshape-see galatea/cond/neighbs.py
    #however, at some point the grad for this was broken. check that calling grad
    #on images2neibs doesn't raise an exception before sinking too much time
    #into this.
    #here I stabilized the softplus with 4 calls to T.maximum and 5 elemwise
    #subs. this is 10% slower than the unstable version, and the gradient
    #is 40% slower. on GPU both the forward prop and backprop are more like
    #100% slower!
    #might want to dry doing a reshape, a T.nnet.softplus, and a reshape
    #instead
    #another way to implement the stabilization is with the max pooling operator
    #(you'd still need to do maximum with 0)


    #timing hack
    #return T.nnet.sigmoid(z[:,0:z.shape[1]/pool_shape[0],0:z.shape[2]/pool_shape[1],:]), T.nnet.sigmoid(z)

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    flat_z = []

    for i in xrange(r):
        for j in xrange(c):
            cur_part = z[:,i:zr:r,j:zc:c,:]
            assert cur_part.ndim == 4
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i,j)
            flat_z.append( cur_part.dimshuffle(0,1,2,3,'x') )

    flat_z.append(T.zeros_like(flat_z[-1]))

    stacked_z = T.concatenate( flat_z, axis = 4)

    batch_size, rows, cols, channels, outcomes = stacked_z.shape
    reshaped_z = stacked_z.reshape((batch_size * rows * cols * channels, outcomes))

    dist = T.nnet.softmax_with_bias(reshaped_z,T.zeros_like(reshaped_z[0,:]))

    dist = dist.reshape((batch_size, rows, cols, channels, outcomes))

    p = 1. - dist[:,:,:,:,-1]
    p.name = 'p(%s)' % z_name

    h = T.alloc(0., batch_size, zr, zc, ch)

    idx = 0
    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:,i:zr:r,j:zc:c,:],
                    dist[:,:,:,:,idx])
            idx += 1

    h.name = 'h(%s)' % z_name

    return p, h

def max_pool_softmax_op(z, pool_shape):
    #random max pooling implemented with set_subtensor
    #could also do this using the stuff in theano.sandbox.neighbours
    #might want to benchmark the two approaches, see how each does on speed/memory
    #on cpu and gpu
    #note: actually theano.sandbox.neighbours is probably a bad idea. it treats
    #the images as being one channel, and emits all channels and positions into
    #a 2D array. so I'd need to index each channel separately and join the channels
    #back together, with a reshape. I expect joining num_channels is more expensive
    #then incsubtensoring pool_rows*pool_cols, simply because we tend to have small
    #pooling regions and a lot of channels, but I guess this worth testing.
    #actually I might be able to do it fast with reshape-see galatea/cond/neighbs.py
    #however, at some point the grad for this was broken. check that calling grad
    #on images2neibs doesn't raise an exception before sinking too much time
    #into this.
    #here I stabilized the softplus with 4 calls to T.maximum and 5 elemwise
    #subs. this is 10% slower than the unstable version, and the gradient
    #is 40% slower. on GPU both the forward prop and backprop are more like
    #100% slower!
    #might want to dry doing a reshape, a T.nnet.softplus, and a reshape
    #instead
    #another way to implement the stabilization is with the max pooling operator
    #(you'd still need to do maximum with 0)


    #timing hack
    #return T.nnet.sigmoid(z[:,0:z.shape[1]/pool_shape[0],0:z.shape[2]/pool_shape[1],:]), T.nnet.sigmoid(z)

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    flat_z = []

    for i in xrange(r):
        for j in xrange(c):
            cur_part = z[:,i:zr:r,j:zc:c,:]
            assert cur_part.ndim == 4
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i,j)
            flat_z.append( cur_part.dimshuffle(0,1,2,3,'x') )

    flat_z.append(T.zeros_like(flat_z[-1]))

    stacked_z = T.concatenate( flat_z, axis = 4)

    batch_size, rows, cols, channels, outcomes = stacked_z.shape
    reshaped_z = stacked_z.reshape((batch_size * rows * cols * channels, outcomes))

    dist = T.nnet.softmax(reshaped_z)

    dist = dist.reshape((batch_size, rows, cols, channels, outcomes))

    p = 1. - dist[:,:,:,:,len(flat_z)-1]
    p.name = 'p(%s)' % z_name

    h = T.alloc(0., batch_size, zr, zc, ch)

    idx = 0
    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:,i:zr:r,j:zc:c,:],
                    dist[:,:,:,:,idx])
            idx += 1

    h.name = 'h(%s)' % z_name

    return p, h

max_pool = max_pool_stable_graph

def check_correctness(f):
    print 'checking correctness of',f
    rng = np.random.RandomState([2012,7,19])
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX) * 2. - 3.

    p_np, h_np = max_pool_python( zv, (pool_rows, pool_cols) )

    z_th = T.TensorType( broadcastable=(False,False,False,False), dtype = config.floatX)()
    z_th.name = 'z_th'

    p_th, h_th = f( z_th, (pool_rows, pool_cols) )

    func = function([z_th],[p_th,h_th])

    pv, hv = func(zv)

    assert p_np.shape == pv.shape
    assert h_np.shape == hv.shape
    if not np.allclose(h_np,hv):
        print (h_np.min(),h_np.max())
        print (hv.min(),hv.max())
        assert False
    assert np.allclose(p_np,pv)
    print 'Correct'

def check_sample_correctishness(f):
    print 'checking correctness of',f
    rng = np.random.RandomState([2012,7,19])
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX) * 2. - 3.

    z_th = T.TensorType( broadcastable=(False,False,False,False), dtype = config.floatX)()
    z_th.name = 'z_th'

    theano_rng = MRG_RandomStreams(rng.randint(2147462579))
    p_th, h_th, p_sth, h_sth = f( z_th, (pool_rows, pool_cols), theano_rng )

    prob_func = function([z_th],[p_th,h_th])
    pv, hv = prob_func(zv)

    sample_func = function([z_th],[p_sth, h_sth])

    acc_p = 0. * pv
    acc_h = 0. * hv

    # make sure the test gets good coverage, ie, that it includes many different
    # activation probs for both detector and pooling layer
    buckets = 10
    bucket_width = 1. / float(buckets)
    for i in xrange(buckets):
        lower_lim = i * bucket_width
        upper_lim = (i+1) * bucket_width

        assert np.any( (pv >= lower_lim) * (pv < upper_lim) )
        assert np.any( (hv >= lower_lim) * (hv < upper_lim) )

    assert upper_lim == 1.


    for i in xrange(10000):
        ps, hs = sample_func(zv)

        assert ps.shape == pv.shape
        assert hs.shape == hv.shape

        acc_p += ps
        acc_h += hs

    est_p = acc_p / float(i+1)
    est_h = acc_h / float(i+1)

    pd = np.abs(est_p-pv)
    hd = np.abs(est_h-hv)

    """
    # plot maps of the estimation error, this is to see if it has some spatial pattern
    # this is useful for detecting bugs like not handling the border correctly, etc.
    from pylearn2.gui.patch_viewer import PatchViewer

    pv = PatchViewer((pd.shape[0],pd.shape[3]),(pd.shape[1],pd.shape[2]),is_color = False)
    for i in xrange(pd.shape[0]):
        for j in xrange(pd.shape[3]):
            pv.add_patch( (pd[i,:,:,j] / pd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()

    pv = PatchViewer((hd.shape[0],hd.shape[3]),(hd.shape[1],hd.shape[2]),is_color = False)
    for i in xrange(hd.shape[0]):
        for j in xrange(hd.shape[3]):
            pv.add_patch( (hd[i,:,:,j] / hd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()
    """

    """
    plot expectation to estimate versus error in estimation
    expect bigger errors for values closer to 0.5

    from matplotlib import pyplot as plt

    #nelem = reduce( lambda x, y : x*y, pd.shape)
    #plt.scatter( pv.reshape(nelem), pd.reshape(nelem))
    #plt.show()

    nelem = reduce( lambda x, y : x*y, hd.shape)
    plt.scatter( hv.reshape(nelem), hd.reshape(nelem))
    plt.show()
    """

    # don't really know how tight this should be
    # but you can try to pose an equivalent problem
    # and implement it in another way
    # using a numpy implementation in softmax_acc.py
    # I got a max error of .17
    assert max(pd.max(), hd.max()) < .17

    # Do exhaustive checks on just the last sample
    assert np.all( (ps ==0) + (ps == 1) )
    assert np.all( (hs == 0) + (hs == 1) )

    for k in xrange(batch_size):
        for i in xrange(ps.shape[1]):
            for j in xrange(ps.shape[2]):
                for l in xrange(channels):
                    p = ps[k,i,j,l]
                    h = hs[k,i*pool_rows:(i+1)*pool_rows,j*pool_cols:(j+1)*pool_cols,l]
                    assert h.shape == (pool_rows, pool_cols)
                    assert p == h.max()


    print 'Correctish (cant tell if samples are perfectly "correct")'

def profile(f):
    print 'profiling ',f
    rng = np.random.RandomState([2012,7,19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU transfer during test
    p_shared = sharedX(zv[:,0:rows:pool_rows,0:cols:pool_cols,:])
    h_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f( z_shared, (pool_rows, pool_cols) )

    func = function([],updates = { p_shared : p_th, h_shared : h_th} )

    print 'warming up'
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        print t2 - t1
        results.append(t2-t1)
    print 'final: ',sum(results)/float(trials)

def profile_samples(f):
    print 'profiling samples',f
    rng = np.random.RandomState([2012,7,19])
    theano_rng = MRG_RandomStreams(rng.randint(2147462579))
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU transfer during test
    p_shared = sharedX(zv[:,0:rows:pool_rows,0:cols:pool_cols,:])
    h_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th, ps_th, hs_th = f( z_shared, (pool_rows, pool_cols), theano_rng )

    func = function([],updates = { p_shared : ps_th, h_shared : hs_th} )

    print 'warming up'
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        print t2 - t1
        results.append(t2-t1)
    print 'final: ',sum(results)/float(trials)

def profile_grad(f):
    print 'profiling gradient of ',f
    rng = np.random.RandomState([2012,7,19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU transfer during test
    grad_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f( z_shared, (pool_rows, pool_cols) )

    func = function([],updates = { grad_shared : T.grad(p_th.sum() +  h_th.sum(), z_shared)} )

    print 'warming up'
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        print t2 - t1
        results.append(t2-t1)
    print 'final: ',sum(results)/float(trials)

if __name__ == '__main__':
    check_correctness(max_pool_softmax_op)
    check_correctness(max_pool_softmax_with_bias_op)
    check_correctness(max_pool_raw_graph)
    check_correctness(max_pool_stable_graph)
    check_sample_correctishness(max_pool_stable_graph)
    #profile(max_pool_raw_graph)
    #profile(max_pool_stable_graph)
    #profile_samples(max_pool_stable_graph)
    #profile(max_pool_softmax_op)
    #profile(max_pool_softmax_with_bias_op)
    #profile_grad(max_pool_raw_graph)
    #profile_grad(max_pool_stable_graph)
    #profile_grad(max_pool_softmax_op)
    #profile_grad(max_pool_softmax_with_bias_op)






def max_pool_stable_graph_bc01(z, pool_shape, top_down = None, theano_rng = None):
    """
        copy-paste of max_pool_stable_graph, then edited to be formatted as (batch idx, channel, row, col)
        rather than (batch_idx, row, col, channel)
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'


    batch_size, ch, zr, zc = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    if top_down is None:
        t = 0.
    else:
        t = - top_down
        t.name = 'neg_top_down'

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:,:,i:zr:r,j:zc:c]
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i,j)
            zpart[i].append( cur_part )
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,'+cur_part.name+')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max('+cur_part.name+','+mx.name+')'
                mx = T.maximum(mx,cur_part)
                mx.name = mx_name
    mx.name = 'local_max('+z_name+')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append( cur_pt )

    off_pt = T.exp(t - mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    off_prob = off_pt / denom
    p = 1. - off_prob
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append( [ pt_ij / denom for pt_ij in pt[i] ] )

    h = T.alloc(0., batch_size, ch, zr, zc)

    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:,:,i:zr:r,j:zc:c],hpart[i][j])

    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h
    else:
        events = []
        for i in xrange(r):
            for j in xrange(c):
                events.append(hpart[i][j])
        events.append(off_prob)

        events = [ event.dimshuffle(0,1,2,3,'x') for event in events ]

        events = tuple(events)

        stacked_events = T.concatenate( events, axis = 4)

        batch_size, channels, rows, cols, outcomes = stacked_events.shape
        reshaped_events = stacked_events.reshape((batch_size * rows * cols * channels, outcomes))

        multinomial = theano_rng.multinomial(pvals = reshaped_events, dtype = p.dtype)

        reshaped_multinomial = multinomial.reshape((batch_size, channels, rows, cols, outcomes))

        h_sample = T.alloc(0., batch_size, ch, zr, zc)

        idx = 0
        for i in xrange(r):
            for j in xrange(c):
                h_sample = T.set_subtensor(h_sample[:,:,i:zr:r,j:zc:c],
                        reshaped_multinomial[:,:,:,:,idx])
                idx += 1

        p_sample = 1 - reshaped_multinomial[:,:,:,:,-1]

        return p, h, p_sample, h_sample
