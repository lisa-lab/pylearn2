import numpy
import theano
import theano.sparse
from theano import tensor
from pylearn2.autoencoder import DenoisingAutoencoder
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
from theano.sparse.sandbox.sp2 import sampling_dot

from pylearn2.expr.basic import theano_norms

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

class Linear(object):
    def __call__(self, X_before_activation):
        # X_before_activation is linear inputs of hidden units, dense
        return X_before_activation
        
class Rectify(object):
    def __call__(self, X_before_activation):
        # X_before_activation is linear inputs of hidden units, dense
        return X_before_activation * (X_before_activation > 0)

class SparseDenoisingAutoencoder(DenoisingAutoencoder):
    """
    denoising autoencoder working with only sparse inputs and efficient reconstruction sampling

    Based on: 
    Y. Dauphin, X. Glorot, Y. Bengio.
    Large-Scale Learning of Embeddings with Reconstruction Sampling.
    In Proceedings of the 28th International Conference on Machine Learning (ICML 2011).
    """
    def __init__(self, corruptor, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001):

        # sampling dot only supports tied weights
        assert tied_weights == True
        
        self.names_to_del = set()
        
        super(SparseDenoisingAutoencoder, self).__init__(corruptor,
                                    nvis, nhid, act_enc, act_dec,
                                    tied_weights=tied_weights, irange=irange, rng=rng)

        # this step is crucial to save loads of space because w_prime is never used in
        # training the sparse da.
        del self.w_prime
        
        self.input_space = VectorSpace(nvis, sparse=True)
        
    def get_params(self):
        # this is needed because sgd complains when not w_prime is not used in grad
        # so delete w_prime from the params list
        params = super(SparseDenoisingAutoencoder, self).get_params()
        return params[0:3]
        
    def encode(self, inputs):
        if (isinstance(inputs, theano.sparse.basic.SparseVariable)):
            return self._hidden_activation(inputs)
        else:
            raise TypeError
            #return [self.encode(v) for v in inputs]

    def decode(self, hiddens, pattern):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        hiddens : tensor_like or list of tensor_likes
        Theano symbolic (or list thereof) representing the input
        minibatch(es) to be encoded. Assumed to be 2-tensors, with the
        first dimension indexing training examples and the second indexing
        data dimensions.

        pattern: dense matrix, the same shape of the minibatch inputs
        0/1 like matrix specifying how to reconstruct inputs. 
        
        Returns
        -------
        decoded : tensor_like or list of tensor_like
        Theano symbolic (or list thereof) representing the corresponding
        minibatch(es) after decoding.
        """
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
            if isinstance(hiddens, tensor.Variable):
                pattern = theano.sparse.csr_from_dense(pattern)
                return act_dec(self.visbias + theano.sparse.dense_from_sparse(sampling_dot(hiddens, self.weights, pattern)))
            else:
                return [self.decode(v, pattern) for v in hiddens]
    
    def reconstruct(self, inputs, pattern):
        """
        Parameters
        ----------
        inputs : theano sparse variable

        pattern: binary dense matrix specifying which part of inputs should be reconstructed
        """
        # corrupt the inputs
        inputs_dense = theano.sparse.dense_from_sparse(inputs)
        corrupted = self.corruptor(inputs_dense)
        inputs = theano.sparse.csc_from_dense(corrupted)

        return self.decode(self.encode(inputs), pattern)
        
    def reconstruct_without_dec_acti(self, inputs, pattern):
        # return results before applying the decoding activation function
        inputs_dense = theano.sparse.dense_from_sparse(inputs)
        corrupted = self.corruptor(inputs_dense)
        inputs = theano.sparse.csc_from_dense(corrupted)
        
        hiddens = self.encode(inputs)
        
        outputs = self.visbias + sampling_dot.sampling_dot(hiddens, self.weights, pattern)

        return outputs
        
    def _hidden_input(self, x):
        """
        Given a single minibatch, computes the input to the
        activation nonlinearity without applying it.
        
        Parameters
        ----------
        x : theano sparse variable 
        Theano symbolic representing the corrupted input minibatch.
        
        Returns
        -------
        y : tensor_like
        (Symbolic) input flowing into the hidden layer nonlinearity.
        """
                
        return self.hidbias + theano.sparse.dot(x, self.weights)
        
    def get_monitoring_channels(self, V):
        
        vb, hb, weights = self.get_params()
        norms = theano_norms(weights)
        return {'W_min': tensor.min(weights),
                'W_max': tensor.max(weights),
                'W_norm_mean': tensor.mean(norms),
                'bias_hid_min' : tensor.min(hb),
                'bias_hid_mean' : tensor.mean(hb),
                'bias_hid_max' : tensor.max(hb),
                'bias_vis_min' : tensor.min(vb),
                'bias_vis_mean' : tensor.mean(vb),
                'bias_vis_max': tensor.max(vb),
        }
        