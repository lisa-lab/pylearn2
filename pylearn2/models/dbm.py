__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2011, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"


import time
from pylearn2.models import Model
from theano import config, function, shared
import theano.tensor as T
import numpy as np
import warnings
from theano.gof.op import get_debug_values, debug_error_message
from pylearn2.utils import make_name, sharedX, as_floatX
from pylearn2.expr.information_theory import entropy_binary_vector
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.models.rbm import RBM
from pylearn2.expr.nnet import sigmoid_numpy

warnings.warn('s3c changing the recursion limit')
import sys
sys.setrecursionlimit(50000)

from pylearn2.models.s3c import numpy_norms
from pylearn2.models.s3c import theano_norms
from pylearn2.models.s3c import full_min
from pylearn2.models.s3c import full_max
from theano.printing import min_informative_str
from theano.printing import Print
from pylearn2.space import VectorSpace
try:
    from scipy import io
except ImportError:
    warnings.warn("Could not import scipy")

class Sampler:
    def __init__(self, theano_rng, kind = 'binomial'):
        self.theano_rng = theano_rng
        self.kind = kind

    def __call__(self, P):
        kind = self.kind
        if kind == 'binomial':
            return self.theano_rng.binomial(size = P.shape, n = 1, p = P, dtype = P.dtype)
        elif kind == 'multinomial':
            return self.theano_rng.multinomial( n = 1, pvals = P, dtype = P.dtype)
        else:
            raise ValueError("Unrecognized sampling kind: "+kind)

class DBM(Model):

    def __init__(self, rbms,
                    use_cd = False,
                        negative_chains = 0,
                       inference_procedure = None,
                       monitor_params = False,
                       sampling_steps = 5,
                       num_classes = 0,
                       init_beta = None,
                       min_beta = None,
                       print_interval = 10000):
        """
            rbms: list of rbms to stack
                    all rbms must be of type pylearn2.models.rbm, and not a subclass
                    first entry is the visible rbm
                    DBM may destroy these rbms-- it won't delete them,
                    but it may do terrible things to them

                    the DBM parameters will be constructed by taking the visible biases
                    and weights from each RBM. only the topmost RBM will additionally
                    donate its hidden biases.
            negative_chains: the number of negative chains to simulate
            inference_procedure: a pylearn2.models.dbm.InferenceProcedure object
                (if None, assumes the model is not meant to run on its own)
            print_interval: every print_interval examples, print out a status summary
            num_classes: if > 0, makes an extra visible layer attached to the deepest
                        hidden layer. this layer is one-hot and driven by the labels
                        from the data
        """

        self.init_beta = init_beta
        self.min_beta = min_beta

        self.sampling_steps = sampling_steps
        self.monitor_params = monitor_params

        self.use_cd = use_cd
        self.num_classes = num_classes

        if use_cd:
            assert negative_chains == 0
        else:
            assert negative_chains > 0

        warnings.warn("""The DBM class is still under development, and currently mostly
                only supports use as a component of a larger model that remains
                private. Contact Ian Goodfellow if you have questions about the current
                status of this class.""")

        super(DBM,self).__init__()

        self.autonomous = False

        self.rbms = rbms
        self.negative_chains = negative_chains

        self.monitoring_channel_prefix = ""

        if inference_procedure is None:
            self.autonomous = False
            inference_procedure = InferenceProcedure()

        self.inference_procedure = inference_procedure
        self.inference_procedure.register_model(self)
        if self.inference_procedure.autonomous:
            raise NotImplementedError("No such thing as an autonomous DBM yet")

        self.print_interval = print_interval

        #copy parameters from RBM to DBM, ignoring bias_hid of all but last RBM
        self.W = []
        for rbm in self.rbms:
            weights ,= rbm.transformer.get_params()
            self.W.append( weights )

        for i, W in enumerate(self.W):
            W.name = 'dbm_W[%d]' % (i,)
        self.bias_vis = rbms[0].bias_vis
        self.bias_hid = [ rbm.bias_vis for rbm in self.rbms[1:] ]
        self.bias_hid.append(self.rbms[-1].bias_hid)
        for i, bias_hid in enumerate(self.bias_hid):
            bias_hid.name = 'dbm_bias_hid[%d]' % (i,)

        self.reset_rng()

        if self.num_classes > 0:
            self.bias_class = sharedX(np.zeros((self.num_classes,)), name = 'bias_class')
            nhid = self.bias_hid[-1].get_value().shape[0]
            self.W_class = sharedX( .001 * self.rng.randn(nhid, self.num_classes),
                                    name = 'W_class')

        self.redo_everything()


    def get_weights(self):
        x = raw_input('which weights?')
        assert x in ['0','1']
        if x == '0':
            return self.W[0].get_value()
        return np.dot(self.W[0].get_value(),self.W[1].get_value())

    def get_input_space(self):
        return self.rbms[0].get_input_space()

    def get_output_space(self):
        return VectorSpace(self.num_classes)

    def reset_rng(self):
        self.rng = np.random.RandomState([1,2,3])

    def redo_everything(self):
        """ compiles learn_func if necessary
            makes new negative chains
            does not reset weights or biases
            TODO: figure out how to make the semantics of this cleaner / more in line with other models
        """

        #compile learn_func if necessary
        if self.autonomous:
            self.redo_theano()

        #make the negative chains
        if not self.use_cd:
            self.V_chains = self.make_chains(self.bias_vis)
            self.V_chains.name = 'dbm_V_chains'

            self.H_chains = [ self.make_chains(bias_hid) for bias_hid in self.bias_hid ]
            for i, H_chain in enumerate(self.H_chains):
                H_chain.name = 'dbm_H[%d]_chain' % i

            if self.num_classes > 0:
                P = np.zeros((self.negative_chains, self.num_classes)) \
                        + T.nnet.softmax( self.bias_class )
                temp_theano_rng = RandomStreams(87)
                sample_from = Sampler(temp_theano_rng, 'multinomial')
                values = function([],sample_from(P))()
                self.Y_chains = sharedX(values, 'Y_chains')
            else:
                self.Y_chains = None

        if hasattr(self, 'init_beta') and self.init_beta is not None:
            self.beta = sharedX( np.zeros( self.bias_vis.get_value().shape) + self.init_beta, name = 'beta')


    def make_chains(self, bias):
        """ make the shared variable representing a layer of
            the network for all negative chains

            for now units are initialized randomly based on their
            biases only
            """

        assert not self.use_cd

        b = bias.get_value(borrow=True)

        nhid ,= b.shape

        shape = (self.negative_chains, nhid)

        driver = self.rng.uniform(0.0, 1.0, shape)

        thresh = sigmoid_numpy(b)

        value = driver < thresh

        return sharedX(value)

    def set_monitoring_channel_prefix(self, prefix):
        self.monitoring_channel_prefix = prefix

    def get_monitoring_channels(self, V, Y =  None):

        try:
            self.compile_mode()

            rval = {}

            #from_ip = self.inference_procedure.get_monitoring_channels(V, self)

            #rval.update(from_ip)

            if self.monitor_params:
                for param in self.get_params():
                    rval[param.name + '_min'] = full_min(param)
                    rval[param.name + '_mean'] = T.mean(param)
                    rval[param.name + '_max'] = full_max(param)

                    if 'W' in param.name:
                        norms = theano_norms(param)

                        rval[param.name + '_norms_min' ]= T.min(norms)
                        rval[param.name + '_norms_mean'] = T.mean(norms)
                        rval[param.name + '_norms_max'] = T.max(norms)

            new_rval = {}
            for key in rval:
                new_rval[self.monitoring_channel_prefix+key] = rval[key]

            rval = new_rval

            return rval
        finally:
            self.deploy_mode()

    def compile_mode(self):
        """ If any shared variables need to have batch-size dependent sizes,
        sets them all to the sizes used for interactive debugging during graph construction """
        pass

    def deploy_mode(self):
        """ If any shared variables need to have batch-size dependent sizes, sets them all to their runtime sizes """
        pass


    def print_status(self):
        print ""
        bv = self.bias_vis.get_value(borrow=True)
        print "bias_vis: ",(bv.min(),bv.mean(),bv.max())
        for i in xrange(len(self.W)):
            W = self.W[i].get_value(borrow=True)
            print "W[%d]"%i,(W.min(),W.mean(),W.max())
            norms = numpy_norms(W)
            print " norms: ",(norms.min(),norms.mean(),norms.max())
            bh = self.bias_hid[i].get_value(borrow=True)
            print "bias_hid[%d]"%i,(bh.min(),bh.mean(),bh.max())
        if self.num_classes > 0:
            W = self.W_class.get_value(borrow=True)
            print "W_class",(W.min(),W.mean(),W.max())
            norms = numpy_norms(W)
            print " norms: ",(norms.min(),norms.mean(),norms.max())
            bc = self.bias_class.get_value(borrow=True)
            print "bias_class",(bc.min(),bc.mean(),bc.max())


    def get_sampling_updates(self):

        assert not self.use_cd

        ip = self.inference_procedure

        rval = {}

        rval[self.V_chains] = self.V_chains
        for H_chains in self.H_chains:
            rval[H_chains] = H_chains
        if self.num_classes > 0:
            rval[self.Y_chains] = self.Y_chains

        driver = RandomStreams(17)
        sample_from = Sampler(driver)
        sample_multi = Sampler(driver,'multinomial')


        warnings.warn("""TODO: update the sampler
                to sample all odd and then all even layers
                simultaneously""")

        for i in xrange(self.sampling_steps):
            #sample the visible units
            V_prob = ip.infer_H_hat_one_sided(
                    other_H_hat = rval[self.H_chains[0]],
                    W = self.W[0].T, b = self.bias_vis)

            V_sample = sample_from(V_prob)

            rval[self.V_chains] = V_sample

            #sample the first hidden layer unless this is also the last hidden layer)
            if len(self.H_chains) > 1:
                prob = ip.infer_H_hat_two_sided(H_hat_below = rval[self.V_chains], H_hat_above = rval[self.H_chains[1]], W_below = self.W[0], W_above = self.W[1], b = self.bias_hid[0])

                sample = sample_from(prob)

                rval[self.H_chains[0]] = sample

            #sample the intermediate hidden layers
            for i in xrange(1,len(self.H_chains)-1):
                prob = ip.infer_H_hat_two_sided(H_hat_below = rval[self.H_chains[i-1]], H_hat_above = rval[self.H_chains[i+1]],
                                                W_below = self.W[i], W_above = self.W[i+1], b = self.bias_hid[i])
                sample = sample_from(prob)

                rval[self.H_chains[i-1]] = sample

            #sample the last hidden layer
            if len(self.H_chains) > 1:
                ipt = rval[self.H_chains[-2]]
            else:
                ipt = rval[self.V_chains]

            if self.num_classes == 0:
                prob = ip.infer_H_hat_one_sided(other_H_hat = ipt, W = self.W[-1], b = self.bias_hid[-1])
            else:
                prob = ip.infer_H_hat_two_sided(H_hat_below = ipt, H_hat_above = rval[self.Y_chains],
                        W_below = self.W[-1], W_above = self.W_class, b = self.bias_hid[-1])

            sample = sample_from(prob)

            rval[self.H_chains[-1]] = sample

            #sample the class labels
            if self.num_classes > 0:
                prob = ip.infer_Y_hat( H_hat = rval[self.H_chains[-1]] )
                sample = sample_multi(prob)
                rval[self.Y_chains] = sample

        return rval

    def rao_blackwellize(self, V_sample, H_samples, Y_sample = None):
        """ Returns a new H_samples list with the the even-numbered
        layers of hidden samples replaced by activation probabilities """


        #TODO: update this so user can control whether
        #to integrate out odd or even layers
        #for now always integrates out even hidden layers
        #because this makes sure it always integrates out
        #something even when the DBM is just an RBM

        ip = self.inference_procedure

        rval_H = [ H_sample for H_sample in H_samples ]

        assert (Y_sample is None) == (self.num_classes == 0)

        for i in xrange(0,len(rval_H),2):
            #Special case for layer 0--it is attached to the input
            if i ==  0:
                #Only do this case if it is not the last layer. Otherwise the last layer special case will catch it
                if len(self.H_chains) > 1:
                    rval_H[i] = ip.infer_H_hat_two_sided(H_hat_below = V_sample, H_hat_above = rval_H[H_samples[i+1]],
                            W_below = self.W[0], W_above = self.W[1], b = self.bias_hid[0])

            if i > 0 and i < len(rval_H) - 1:
                #Case for intermediate layers
                rval_H[i] = ip.infer_H_hat_two_sided(H_hat_below = H_samples[i-1], H_hat_above = H_samples[i+1],
                        W_below = self.W[i], W_above = self.W[i+1], b = self.bias_hid[i])

            if i == len(rval_H) - 1:
                #case for final layer
                if len(rval_H) > 1:
                    assert H_samples[-2] is H_samples[i-1]
                    ipt = H_samples[-2]
                else:
                    ipt = V_sample

                assert self.bias_hid[i] is self.bias_hid[-1]

                if Y_sample is None:
                    rval_H[i] = ip.infer_H_hat_one_sided(other_H_hat = ipt, W = self.W[-1], b = self.bias_hid[-1])
                else:
                    rval_H[i] = ip.infer_H_hat_two_sided(H_hat_below = ipt, W_below = self.W[-1], b = self.bias_hid[-1],
                            W_above = self.W_class, H_hat_above = Y_sample)

        if Y_sample is None:
            rval_Y = None
        else:
            if len(rval_H) % 2 == 0:
                rval_Y = ip.infer_Y_hat(H_hat = H_samples[-1])
            else:
                rval_Y = Y_sample

        return rval_H, rval_Y

    def get_cd_neg_phase_grads(self, V, H_hat, Y = None):

        assert self.use_cd
        assert not hasattr(self, 'V_chains')
        assert len(H_hat) == len(self.rbms)
        assert (Y is None) == (self.num_classes == 0)

        driver = RandomStreams(42)
        sample_from = Sampler(driver)

        H_samples = []

        for H_hat_elem in H_hat:
            H_samples.append(sample_from(H_hat_elem))

        ip = self.inference_procedure

        #leave the top layer as-is: a sample from Q
        #use it as the starting point to resample everything else
        # (the lower hidden layers, V, and, if applicable, Y)

        if Y is None:
            Y_sample = None
        else:
            Y_hat = ip.infer_class(H_hat = H_samples[-1])
            sample_multinomial = Sampler(driver,'multinomial')
            Y_sample = sample_multinomial(Y_hat)

        for i in xrange(len(H_samples)-2,-1,-1):
            if i > 0:
                H_hat_below = H_samples[i-1]
            else:
                H_hat_below = V
            prob = ip.infer_H_hat_two_sided( H_hat_below = H_hat_below, H_hat_above = H_samples[i+1],
                    W_below = self.W[i], W_above = self.W[i+1], b = self.bias_hid[i])
            H_samples[i] = sample_from(prob)

        V_sample = sample_from(ip.infer_H_hat_one_sided(other_H_hat = H_hat[0], W = self.W[0].T, b = self.bias_vis))

        return self.get_neg_phase_grads_from_samples(V_sample, H_samples, Y_sample)

    def get_neg_phase_grads(self):
        """ returns a dictionary mapping from parameters to negative phase gradients
            (assuming you're doing gradient ascent on negative variational free energy)
        """

        assert not self.use_cd

        return self.get_neg_phase_grads_from_samples(self.V_chains, self.H_chains, self.Y_chains)

    def get_neg_phase_grads_from_samples(self, V_sample, H_samples, Y_sample = None):

        if hasattr(self, 'V_chains'):
            # theta must be updated using samples that were generated using gibbs sampling
            # on theta
            # if we use the shared variable itself, then these samples were generated using
            # the *previous* value of theta
            assert V_sample is not self.V_chains

        H_rao_blackwell, Y_rao_blackwell = self.rao_blackwellize(V_sample, H_samples, Y_sample)

        obj = self.expected_energy(V_hat = V_sample, H_hat = H_rao_blackwell, Y_hat = Y_rao_blackwell)

        constants = list(set(H_rao_blackwell).union([V_sample]))

        if Y_rao_blackwell is not None:
            constants = constants.union(Y_rao_blackwell)

        params = self.get_params()

        grads = T.grad(obj, params, consider_constant = constants)

        rval = {}

        for param, grad in zip(params, grads):
            rval[param] = grad

        assert self.bias_vis in rval

        return rval


    def get_params(self):
        rval = set([self.bias_vis])
        if self.bias_vis.name is None:
            warnings.warn('whoa, for some reason bias_vis was unnamed')
            self.bias_vis.name = 'dbm_bias_vis'

        assert len(self.W) == len(self.bias_hid)

        for i in xrange(len(self.W)):
            rval = rval.union(set([ self.W[i], self.bias_hid[i]]))
            assert self.W[i].name is not None
            assert self.bias_hid[i].name is not None

        if self.num_classes > 0:
            rval = rval.union([self.W_class, self.bias_class])

        rval = list(rval)

        assert self.bias_hid[0] in rval

        if hasattr(self,'beta'):
            rval.append(self.beta)

        return rval

    def make_learn_func(self, V):
        """
        V: a symbolic design matrix
        """

        raise NotImplementedError("Not yet supported-- current project does not require DBM to learn on its own")

        """
        #E step
        hidden_obs = self.e_step.variational_inference(V)

        stats = SufficientStatistics.from_observations(needed_stats = self.m_step.needed_stats(),
                V = V, **hidden_obs)

        H_hat = hidden_obs['H_hat']
        S_hat = hidden_obs['S_hat']

        learning_updates = self.m_step.get_updates(self, stats, H_hat, S_hat)


        self.censor_updates(learning_updates)


        print "compiling function..."
        t1 = time.time()
        rval = function([V], updates = learning_updates)
        t2 = time.time()
        print "... compilation took "+str(t2-t1)+" seconds"
        print "graph size: ",len(rval.maker.env.toposort())

        return rval
        """

    def censor_updates(self, updates):

        for rbm in self.rbms:
            rbm.censor_updates(updates)

        if hasattr(self,'beta') and self.beta in updates:
            #todo--censorship cache, etc.
            min_beta = 1e-4
            if hasattr(self,'min_beta') and self.min_beta is not None:
                min_beta = self.min_beta
            updates[self.beta] = T.clip(updates[self.beta],min_beta,1e6)

    def random_design_matrix(self, batch_size, theano_rng):
        raise NotImplementedError()

        #return V_sample

    def expected_energy(self, V_hat, H_hat, Y_hat = None, no_v_bias = False):
        """ expected energy of the model under the mean field distribution
            defined by V_hat and H_hat
            alternately, could be expectation of the energy function across
            a batch of examples, where every element of V_hat and H_hat is
            a binary observation
            if no_v_bias is True, ignores the contribution from biases on visible units
        """

        assert (Y_hat is None) == (self.num_classes == 0)

        V_name = make_name(V_hat, 'anon_V_hat')
        assert isinstance(H_hat, (list,tuple))

        H_names = []
        for i in xrange(len(H_hat)):
            H_names.append( make_name(H_hat[i], 'anon_H_hat[%d]' %(i,) ))

        m = V_hat.shape[0]
        m.name = V_name + '.shape[0]'

        assert len(H_hat) == len(self.rbms)

        v = T.mean(V_hat, axis=0)

        if no_v_bias:
            v_bias_contrib = 0.
        else:
            v_bias_contrib = T.dot(v, self.bias_vis)

        #exp_vh = T.dot(V_hat.T,H_hat[0]) / m

        #v_weights_contrib = T.sum(self.W[0] * exp_vh)

        v_weights_contrib = (T.dot(V_hat, self.W[0]) * H_hat[0]).sum(axis=1).mean()

        v_weights_contrib.name = 'v_weights_contrib('+V_name+','+H_names[0]+')'

        total = v_bias_contrib + v_weights_contrib

        for i in xrange(len(H_hat) - 1):
            lower_H = H_hat[i]
            low = T.mean(lower_H, axis = 0)
            higher_H = H_hat[i+1]
            #exp_lh = T.dot(lower_H.T, higher_H) / m
            lower_bias = self.bias_hid[i]
            W = self.W[i+1]

            lower_bias_contrib = T.dot(low, lower_bias)

            #weights_contrib = T.sum( W * exp_lh) / m
            weights_contrib = (T.dot(lower_H, W) * higher_H).sum(axis=1).mean()

            total = total + lower_bias_contrib + weights_contrib

        highest_bias_contrib = T.dot(T.mean(H_hat[-1],axis=0), self.bias_hid[-1])

        total = total + highest_bias_contrib

        assert len(total.type.broadcastable) == 0

        if Y_hat is not None:
            weights_contrib = (T.dot(H_hat[-1], self.W_class) * Y_hat).sum(axis=1).mean()
            bias_contrib = T.dot(T.mean(Y_hat,axis=0), self.bias_class)
            total = total + weights_contrib + bias_contrib

        rval =  - total

        #rval.name = 'dbm_expected_energy('+V_name+','+str(H_names)+')'

        return rval



    def expected_energy_batch(self, V_hat, H_hat, Y_hat = None, no_v_bias = False):
        """ expected energy of the model under the mean field distribution
            defined by V_hat and H_hat
            alternately, could be expectation of the energy function across
            a batch of examples, where every element of V_hat and H_hat is
            a binary observation
            if no_v_bias is True, ignores the contribution from biases on visible units
        """

        warnings.warn("TODO: write unit test verifying expected_energy_batch/m = expected_energy")

        assert (Y_hat is None) == (self.num_classes == 0)

        V_name = make_name(V_hat, 'anon_V_hat')
        assert isinstance(H_hat, (list,tuple))

        H_names = []
        for i in xrange(len(H_hat)):
            H_names.append( make_name(H_hat[i], 'anon_H_hat[%d]' %(i,) ))

        assert len(H_hat) == len(self.rbms)

        if no_v_bias:
            v_bias_contrib = 0.
        else:
            v_bias_contrib = T.dot(V_hat, self.bias_vis)


        assert len(V_hat.type.broadcastable) == 2
        assert len(self.W[0].type.broadcastable) == 2
        assert len(H_hat[0].type.broadcastable) == 2

        interm1 = T.dot(V_hat, self.W[0])
        assert len(interm1.type.broadcastable) == 2
        interm2 = interm1 * H_hat[0]
        assert len(interm2.type.broadcastable) == 2

        v_weights_contrib = interm2.sum(axis=1)

        v_weights_contrib.name = 'v_weights_contrib('+V_name+','+H_names[0]+')'
        assert len(v_weights_contrib.type.broadcastable) == 1

        total = v_bias_contrib + v_weights_contrib

        for i in xrange(len(H_hat) - 1):
            lower_H = H_hat[i]
            higher_H = H_hat[i+1]
            #exp_lh = T.dot(lower_H.T, higher_H) / m
            lower_bias = self.bias_hid[i]
            W = self.W[i+1]

            lower_bias_contrib = T.dot(lower_H, lower_bias)

            #weights_contrib = T.sum( W * exp_lh) / m
            weights_contrib = (T.dot(lower_H, W) * higher_H).sum(axis=1)

            cur_contrib = lower_bias_contrib + weights_contrib
            assert len(cur_contrib.type.broadcastable) == 1
            total = total + cur_contrib

        highest_bias_contrib = T.dot(H_hat[-1], self.bias_hid[-1])

        total = total + highest_bias_contrib

        if Y_hat is not None:
            weights_contrib = (T.dot(H_hat[-1], self.W_class) * Y_hat).sum(axis=1)
            assert weights_contrib.ndim == 1
            bias_contrib = T.dot(Y_hat, self.bias_class)
            assert bias_contrib.ndim == 1
            total = total + weights_contrib + bias_contrib

        assert len(total.type.broadcastable) == 1

        rval =  - total

        #rval.name = 'dbm_expected_energy('+V_name+','+str(H_names)+')'

        return rval



    def entropy_h(self, H_hat):
        """ entropy of the hidden layers under the mean field distribution
        defined by H_hat """

        for Hv in get_debug_values(H_hat[0]):
            assert Hv.min() >= 0.0
            assert Hv.max() <= 1.0

        total = entropy_binary_vector(H_hat[0])

        for H in H_hat[1:]:

            for Hv in get_debug_values(H):
                assert Hv.min() >= 0.0
                assert Hv.max() <= 1.0

            total += entropy_binary_vector(H)

        return total

    def redo_theano(self):
        raise NotImplementedError("Not yet supported-- current project does not require DBM to learn on its own")
        try:
            self.compile_mode()
            init_names = dir(self)

            V = T.matrix(name='V')
            V.tag.test_value = np.cast[config.floatX](self.rng.uniform(0.,1.,(self.test_batch_size,self.nvis)) > 0.5)

            self.learn_func = self.make_learn_func(V)

            final_names = dir(self)

            self.register_names_to_del([name for name in final_names if name not in init_names])
        finally:
            self.deploy_mode()

    def train_batch(self, dataset, batch_size):
        #TODO [for IG, not LY]: always uses exhaustive iteration, regardless of how the dataset is configured.
        #clean this up a bit

        warnings.warn(" This method isn't tested yet")

        def make_iterator():
            self.iterator = dataset.iterator(
                    mode = 'sequential',
                    batch_size = batch_size,
                    targets = self.dbm.num_classes > 0)

        if self.iterator is None:
            self.batch_size = batch_size
            self.dataset = dataset
            self.register_names_to_del(['dataset','iterator'])
            make_iterator()
        else:
            assert dataset is self.dataset
            assert batch_size == self.batch_size
        if self.num_classes > 0:
            try:
                X, Y = self.iterator.next()
            except StopIteration:
                print 'Finished a dataset-epoch'
                make_iterator()
                X, Y = self.iterator.next()
        else:
            Y = None
            try:
                X = self.iterator.next()
            except StopIteration:
                print 'Finished a dataset-epoch'
                make_iterator()
                X = self.iterator.next()

        self.learn_mini_batch(X,Y)
        return True

    def learn_mini_batch(self, X):
        raise NotImplementedError("Not yet supported-- current project does not require DBM to learn on its own")

        self.learn_func(X)

        if self.monitor.examples_seen % self.print_interval == 0:
            self.print_status()


    def get_weights_format(self):
        return self.rbms[0].get_weights_format()

class InferenceProcedure:
    """

        Variational inference

    """

    def get_monitoring_channels(self):
        raise NotImplementedError()

    def __init__(self, layer_schedule = None, monitor_kl = False):
        self.autonomous = False
        self.model = None
        self.monitor_kl = monitor_kl
        self.layer_schedule = layer_schedule

    def register_model(self, model):
        self.model = model

    def truncated_KL(self, V, obs, Y = None, no_v_bias = False):
        """ KL divergence between variation and true posterior, dropping terms that don't
            depend on the variational parameters

            if no_v_bias is True, ignores the contribution of the visible biases to the expected energy
            """

        """
            D_KL ( Q(h ) || P(h | v) ) =  - sum_h Q(h) log P(h | v) + sum_h Q(h) log Q(h)
                                       = -sum_h Q(h) log P( h, v) + sum_h Q(h) log P(v) + sum_h Q(h) log Q(h)
            <truncated version>        = -sum_h Q(h) log P( h, v) + sum_h Q(h) log Q(h)
                                       = -sum_h Q(h) log exp( -E (h,v)) + sum_h Q(h) log Z + sum_H Q(h) log Q(h)
            <truncated version>        = sum_h Q(h) E(h, v) + sum_h Q(h) log Q(h)

            this comment was written before adding support for Y
        """

        H_hat = obs['H_hat']

        for Hv in get_debug_values(H_hat):
            assert Hv.min() >= 0.0
            assert Hv.max() <= 1.0

        entropy_term = - self.model.entropy_h(H_hat = H_hat)
        assert len(entropy_term.type.broadcastable) == 1
        energy_term = self.model.expected_energy_batch(V_hat = V, H_hat = H_hat, Y_hat = Y, no_v_bias = no_v_bias)
        assert len(energy_term.type.broadcastable) == 1

        KL = entropy_term + energy_term

        return KL

    def infer_H_hat_two_sided(self, H_hat_below, W_below, H_hat_above, W_above, b):

        bottom_up = T.dot(H_hat_below, W_below)
        top_down =  T.dot(H_hat_above, W_above.T)
        total = bottom_up + top_down + b

        H_hat = T.nnet.sigmoid(total)

        return H_hat

    def infer_H_hat_one_sided(self, other_H_hat, W, b):
        """ W should be arranged such that other_H_hat.shape[1] == W.shape[0] """

        if W is self.model.W[-1]:
            assert self.model.num_classes == 0
            #shouldn't be using one-sided inference when there is also top-down influence from
            #labels

        dot = T.dot(other_H_hat, W)
        presigmoid = dot + b

        H_hat = T.nnet.sigmoid(presigmoid)

        return H_hat

    def infer_Y_hat(self, H_hat):

        dot = T.dot(H_hat, self.model.W_class)
        presoftmax = dot + self.model.bias_class

        Y_hat = T.nnet.softmax(presoftmax)

        return Y_hat

    def infer(self, V, return_history = False):
        """

            return_history: if True:
                                returns a list of dictionaries with
                                showing the history of the variational
                                parameters
                                throughout fixed point updates
                            if False:
                                returns a dictionary containing the final
                                variational parameters
        """

        if self.model.num_classes > 0:
            raise NotImplementedError("This inference procedure doesn't support using a class variable as part of the DBM yet")

        H   =    self.init_H_hat(V)

        def check_H(my_H, my_V):
            if my_H.dtype != config.floatX:
                raise AssertionError('my_H.dtype should be config.floatX, but they are '
                        ' %s and %s, respectively' % (my_H.dtype, config.floatX))

            allowed_v_types = ['float32']

            if config.floatX == 'float64':
                allowed_v_types.append('float64')

            assert my_V.dtype in allowed_v_types

            if config.compute_test_value != 'off':
                from theano.gof.op import PureOp
                Hv = PureOp._get_test_value(my_H)

                Vv = my_V.tag.test_value

                assert Hv.shape[0] == Vv.shape[0]

        for H_elem in H:
            check_H(H_elem, V)

        def make_dict():

            return {
                    'H_hat' : H,
                    }

        history = [ make_dict() ]

        for layer in self.layer_schedule:

            assert H[layer] is not None

            if layer == 0:
                ipt = V
            else:
                ipt = H[layer-1]

            if layer == len(self.model.W) -1:
                H[layer] = self.infer_H_hat_one_sided(other_H_hat = ipt, W = self.model.W[layer], b = self.model.bias_hid[layer])
                assert H[layer] is not None
            else:

                H[layer] = self.infer_H_hat_two_sided(H_hat_below = ipt, H_hat_above = H[layer+1],
                        W_below = self.model.W[layer], W_above = self.model.W[layer+1],
                        b = self.model.bias_hid[layer])
                assert H[layer] is not None

            check_H(H[layer],V)

            history.append(make_dict())

        if return_history:
            return history
        else:
            return history[-1]

    def init_H_hat(self, V):
        """ Returns a list of matrices of hidden units, with same batch size as V
            For now hidden unit values are initialized by taking the sigmoid of their
            bias """

        H_hat = []

        for b in self.model.bias_hid:
            value = T.nnet.sigmoid(b)

            mat = T.alloc(value, V.shape[0], value.shape[0])

            H_hat.append(mat)

        return H_hat

    def init_Y_hat(self, V):
        value = T.nnet.sigmoid(self.model.bias_class)
        mat = T.alloc(value, V.shape[0], value.shape[0])

        return mat

def load_matlab_dbm(path, num_chains = 1):
    """ Loads a two layer DBM stored in the format used by Ruslan Salakhutdinov's
    matlab demo"""

    d = io.loadmat(path)

    for key in d:
        try:
            d[key] = np.cast[config.floatX](d[key])
        except:
            pass

    visbiases = d['visbiases']
    assert len(visbiases.shape) == 2
    assert visbiases.shape[0] == 1
    visbiases = visbiases[0,:]

    hidbiases = d['hidbiases']
    assert len(hidbiases.shape) == 2
    assert hidbiases.shape[0] == 1
    hidbiases = hidbiases[0,:]

    penbiases = d['penbiases']
    assert len(penbiases.shape) == 2
    assert penbiases.shape[0] == 1
    penbiases = penbiases[0,:]

    vishid = d['vishid']
    hidpen = d['hidpen']

    D ,= visbiases.shape
    N1 ,= hidbiases.shape
    N2 ,= penbiases.shape

    assert vishid.shape == (D,N1)
    assert hidpen.shape == (N1,N2)

    rbms = [ RBM( nvis = D, nhid = N1),
            RBM( nvis = N1, nhid = N2) ]

    dbm = DBM(rbms, negative_chains = num_chains)

    dbm.bias_vis.set_value(visbiases)
    dbm.bias_hid[0].set_value(hidbiases)
    dbm.bias_hid[1].set_value(penbiases)
    dbm.W[0].set_value(vishid)
    dbm.W[1].set_value(hidpen)

    return dbm

