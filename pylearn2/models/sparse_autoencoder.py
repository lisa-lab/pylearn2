import theano
import theano.sparse
from theano import tensor
from pylearn2.models.autoencoder import DenoisingAutoencoder
from pylearn2.space import VectorSpace
from theano.sparse.sandbox.sp2 import sampling_dot

from pylearn2.expr.basic import theano_norms

class SparseDenoisingAutoencoder(DenoisingAutoencoder):
    """
    Denoising autoencoder working with only sparse inputs and efficient
    reconstruction sampling

    Parameters
    ----------
    corruptor : WRITEME
    nvis : WRITEME
    nhid : WRITEME
    act_enc : WRITEME
    act_dec : WRITEME
    tied_weights : WRITEME
    irange : WRITEME
    rng : WRITEME

    References
    ----------
    Y. Dauphin, X. Glorot, Y. Bengio. Large-Scale Learning of Embeddings with
    Reconstruction Sampling. In Proceedings of the 28th International
    Conference on Machine Learning (ICML 2011).
    """
    
    def __init__(self):
        raise NotImplementedError(
        'This class has been deprecated since 2012.'\
        'In Feb, 2015, all historical codes are hence removed.')
        
