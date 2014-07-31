from functools import wraps
import numpy as np
import scipy

from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX
from theano import config, scan
from theano.compat.python2x import OrderedDict
from theano import config, scan, tensor


class RecursiveConvolutionalLayer(Layer):
    """
        (Binary) Recursive Convolutional Layer
    """
    def __init__(self, dim, layer_name, irange, indices, activation = 'rect', conv_mode = 'conv'):
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self
        super(RecursiveConvolutionalLayer, self).__init__()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Rconv layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space
        self.output_space = VectorSpace(dim=self.dim)

        # Left weight matrix
        self.rng = self.mlp.rng
        W_hh = self.rng.uniform(-self.irange, self.irange, (self.dim, self.dim))
        W_hh,_,_ = scipy.linalg.svd(W_hh)
        self.W_hh = sharedX(0.9 * W_hh)
        self.W_hh.name = self.layer_name + '_W'
        
        # Right weight matrix
        U_hh = self.rng.uniform(-self.irange, self.irange, (self.dim, self.dim))
        U_hh,_,_ = scipy.linalg.svd(U_hh)
        self.U_hh = sharedX(0.9 * U_hh)
        self.U_hh.name = self.layer_name + '_U'
        
        # Bias
        self.b_hh = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b')
        
        # gaters
        self.GW_hh = self.rng.uniform(-self.irange, self.irange, (self.dim, 3))
        self.GU_hh = self.rng.uniform(-self.irange, self.irange, (self.dim, 3))
        self.GW_hh, self.GU_hh = sharedX(self.GW_hh), sharedX(self.GU_hh)
        self.GW_hh.name, self.GU_hh.name = [ self.layer_name + '_' + param for param in ['GW', 'GU'] ]
        self.Gb_hh = sharedX(np.zeros((3,)), name=self.layer_name + '_Gb')
        
        self._params = [self.W_hh, self.U_hh, self.b_hh, self.GW_hh, self.GU_hh, self.Gb_hh]

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None,
                                      targets=None):
        W, U, b, GW, GU, Gb = self._params
        sq_W = tensor.sqr(W)
        sq_U = tensor.sqr(U)
        sq_GW = tensor.sqr(GW)
        sq_GU = tensor.sqr(GU)
        row_norms = tensor.sqrt(sq_W.sum(axis=1))
        col_norms = tensor.sqrt(sq_W.sum(axis=0))
        u_row_norms = tensor.sqrt(sq_U.sum(axis=1))
        u_col_norms = tensor.sqrt(sq_U.sum(axis=0))
        gw_row_norms = tensor.sqrt(sq_GW.sum(axis=1))
        gw_col_norms = tensor.sqrt(sq_GW.sum(axis=0))
        gu_row_norms = tensor.sqrt(sq_GU.sum(axis=1))
        gu_col_norms = tensor.sqrt(sq_GU.sum(axis=0))

        rval = OrderedDict([('W_row_norms_min',  row_norms.min()),
                            ('W_row_norms_mean', row_norms.mean()),
                            ('W_row_norms_max',  row_norms.max()),
                            ('W_col_norms_min',  col_norms.min()),
                            ('W_col_norms_mean', col_norms.mean()),
                            ('W_col_norms_max',  col_norms.max()),
                            ('U_row_norms_min', u_row_norms.min()),
                            ('U_row_norms_mean', u_row_norms.mean()),
                            ('U_row_norms_max', u_row_norms.max()),
                            ('U_col_norms_min', u_col_norms.min()),
                            ('U_col_norms_mean', u_col_norms.mean()),
                            ('U_col_norms_max', u_col_norms.max()),
                            ('GW_row_norms_min', gw_row_norms.min()),
                            ('GW_row_norms_mean', gw_row_norms.mean()),
                            ('GW_row_norms_max', gw_row_norms.max()),
                            ('GW_col_norms_min', gw_col_norms.min()),
                            ('GW_col_norms_mean', gw_col_norms.mean()),
                            ('GW_col_norms_max', gw_col_norms.max()),
                            ('GU_row_norms_min', gu_row_norms.min()),
                            ('GU_row_norms_mean', gu_row_norms.mean()),
                            ('GU_row_norms_max', gu_row_norms.max()),
                            ('GU_col_norms_min', gu_col_norms.min()),
                            ('GU_col_norms_mean', gu_col_norms.mean()),
                            ('GU_col_norms_max', gu_col_norms.max())])

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=0)
            mean = state.mean(axis=0)
            mn = state.min(axis=0)
            rg = mx - mn

            rval['range_x_max_u'] = rg.max()
            rval['range_x_mean_u'] = rg.mean()
            rval['range_x_min_u'] = rg.min()

            rval['max_x_max_u'] = mx.max()
            rval['max_x_mean_u'] = mx.mean()
            rval['max_x_min_u'] = mx.min()

            rval['mean_x_max_u'] = mean.max()
            rval['mean_x_mean_u'] = mean.mean()
            rval['mean_x_min_u'] = mean.min()

            rval['min_x_max_u'] = mn.max()
            rval['min_x_mean_u'] = mn.mean()
            rval['min_x_min_u'] = mn.min()

        return rval

    def fprop(self, state_below):
        state_below, mask = state_below
        
        nsteps = state_below.shape[0]
        batch_size = state_below.shape[1]
        #mask = tensor.alloc(1., nsteps, batch_size)

        W_hh = self.W_hh
        U_hh = self.U_hh
        b_hh = self.b_hh
        
        GW_hh = self.GW_hh
        GU_hh = self.GU_hh
        Gb_hh = self.Gb_hh

        if state_below.ndim == 3:
            b_hh = b_hh.dimshuffle('x','x',0)
        else:
            b_hh = b_hh.dimshuffle('x',0)

        def _level_fprop(mask_t, n_iter, prev_level):
            lower_level = prev_level

            prev_shifted = tensor.zeros_like(prev_level)
            prev_shifted = tensor.set_subtensor(prev_shifted[1:], prev_level[:-1])
            lower_shifted = prev_shifted

            prev_shifted = tensor.dot(prev_shifted, U_hh)
            prev_level = tensor.dot(prev_level, W_hh)
            if self.activation == 'rect':
                new_act = tensor.maximum(0., prev_level + prev_shifted + b_hh)
            elif self.activation == 'tanh':
                new_act = tensor.tanh(prev_level + prev_shifted + b_hh)
            
            gater = tensor.dot(lower_shifted, GU_hh) + \
                    tensor.dot(lower_level, GW_hh) + Gb_hh
            if prev_level.ndim == 3:
                gater_shape = gater.shape
                gater = gater.reshape((gater_shape[0] * gater_shape[1], 3))
            gater = tensor.nnet.softmax(gater)
            if prev_level.ndim == 3:
                gater = gater.reshape((gater_shape[0], gater_shape[1], 3))

            if prev_level.ndim == 3:
                gater_new = gater[:,:,0].dimshuffle(0,1,'x')
                gater_left = gater[:,:,1].dimshuffle(0,1,'x')
                gater_right = gater[:,:,2].dimshuffle(0,1,'x')
            else:
                gater_new = gater[:,0].dimshuffle(0,'x')
                gater_left = gater[:,1].dimshuffle(0,'x')
                gater_right = gater[:,2].dimshuffle(0,'x')

            act = new_act * gater_new + \
                    lower_shifted * gater_left + \
                    lower_level * gater_right
            
            if self.conv_mode == 'deconv':
                new_act = tensor.zeros_like(act)
                new_act = tensor.set_subtensor(new_act[:n_iter],act[:n_iter])
                act = new_act
 
            if prev_level.ndim == 3:
                mask_t = mask_t.dimshuffle('x',0,'x')
            else:
                mask_t = mask_t.dimshuffle('x', 0)
            new_level = tensor.switch(mask_t, act, lower_level)
            new_iter = n_iter + 1

            return new_iter, new_level

        rval, updates = scan(_level_fprop,
                        sequences = [mask[1:]],
                        outputs_info = [tensor.constant(2), state_below],
                        name='layer_%s'%self.layer_name,
                        n_steps = nsteps-1)

        seqlens = tensor.cast(mask.sum(axis=0), 'int64')-1
        roots = rval[-1][-1] # take new_level from [new_iter, new_level] then take last step of scan
    
        if self.conv_mode == 'conv':
            if state_below.ndim == 3:
                def _grab_root(seqlen,one_sample,prev_sample):
                    return one_sample[seqlen]

                roots, updates = scan(_grab_root,
                        sequences = [seqlens, roots.dimshuffle(1,0,2)],
                        outputs_info = [tensor.alloc(0., self.dim)],
                        name='grab_root_%s'%self.layer_name)
                #roots = roots.dimshuffle('x', 0, 1)
            else:
                roots = roots[seqlens]


        # Note that roots has only a single timestep
        new_h = roots
        self.out = roots
        self.rval = roots
        self.updates =updates

        return self.out

    def get_params(self):
        return self._params

