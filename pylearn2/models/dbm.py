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

warnings.warn('s3c changing the recursion limit')
import sys
sys.setrecursionlimit(50000)

from pylearn2.models.s3c import numpy_norms
from pylearn2.models.s3c import theano_norms
from pylearn2.models.s3c import full_min
from pylearn2.models.s3c import full_max
from theano.printing import min_informative_str
from theano.printing import Print

warnings.warn("""
TODO/NOTES
The sampler ought to store the state of all but the topmost hidden layer
learning updates will be based on marginalizing out this topmost layer
to reduce noise a bit
each round of negative phase sampling should start by sampling the topmost
layer, then sampling downward from there
actually, couldn't I marginalize out all the odd or all the even layers?
""")

class Sampler:
    def __init__(self, theano_rng):
        self.theano_rng = theano_rng

    def __call__(self, P):
        return self.theano_rng.binomial(size = P.shape, n = 1, p = P, dtype = P.dtype)

class DBM(Model):

    def __init__(self, rbms,
                    use_cd = False,
                        negative_chains = 0,
                       inference_procedure = None,
                       monitor_params = False,
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

        """

        self.sampling_steps = 5
        self.monitor_params = monitor_params

        self.use_cd = use_cd

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
        self.autonomous = False
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

        self.redo_everything()

    def reset_rng(self):
        self.rng = np.random.RandomState([1,2,3])

    def redo_everything(self):
        """ compiles learn_func if necessary
            makes new negative chains
            does not reset weights or biases
        """

        #compile learn_func if necessary
        if self.autonomous:
            self.redo_theano()

        #make the negative chains
        if not self.use_cd:
            self.V_chains = self.make_chains(self.bias_vis)

            self.H_chains = [ self.make_chains(bias_hid) for bias_hid in self.bias_hid ]


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

        thresh = 1./(1.+np.exp(-b))

        value = driver < thresh

        return sharedX(value)

    def set_monitoring_channel_prefix(self, prefix):
        self.monitoring_channel_prefix = prefix

    def get_monitoring_channels(self, V):

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


    def get_sampling_updates(self):

        assert not self.use_cd

        ip = self.inference_procedure

        rval = {}

        rval[self.V_chains] = self.V_chains
        for H_chains in self.H_chains:
            rval[H_chains] = H_chains

        sample_from = Sampler(RandomStreams(17))

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

            prob = ip.infer_H_hat_one_sided(other_H_hat = ipt, W = self.W[-1], b = self.bias_hid[-1])

            sample = sample_from(prob)

            rval[self.H_chains[-1]] = sample

        return rval

    def rao_blackwellize(self, V_sample, H_samples):
        """ Returns a new H_samples list with the the even-numbered
        layers of hidden samples replaced by activation probabilities """


        #TODO: update this so user can control whether
        #to integrate out odd or even layers
        #for now always integrates out even hidden layers
        #because this makes sure it always integrates out
        #something even when the DBM is just an RBM

        ip = self.inference_procedure

        rval = [ H_sample for H_sample in H_samples ]

        for i in xrange(0,len(rval),2):
            #Special case for layer 0--it is attached to the input
            if i ==  0:
                #Only do this case if it is not the last layer. Otherwise the last layer special case will catch it
                if len(self.H_chains) > 1:
                    assert False #debugging check that this doesn't execute in a small model. feel free to remove
                    rval[i] = ip.infer_H_hat_two_sided(H_hat_below = V_sample, H_hat_above = rval[H_samples[i+1]],
                            W_below = self.W[0], W_above = self.W[1], b = self.bias_hid[0])

            if i > 0 and i < len(rval) - 1:
                assert False #debugging check that this doesn't execute in a small model. feel free to remove
                #Case for intermediate layers
                rval[i] = ip.infer_H_hat_two_sided(H_hat_below = H_samples[i-1], H_hat_above = H_samples[i+1],
                        W_below = self.W[i], W_above = self.W[i+1], b = self.bias_hid[i])

            if i == len(rval) - 1:
                #case for final layer

                if len(rval) > 1:
                    assert False #debugging check that this doesn't execute in a small model. feel free to remove
                    assert H_samples[-2] is H_samples[i-1]
                    ipt = H_samples[-2]
                else:
                    ipt = V_sample

                assert self.bias_hid[i] is self.bias_hid[-1]

                rval[i] = ip.infer_H_hat_one_sided(other_H_hat = ipt, W = self.W[-1], b = self.bias_hid[-1])

        return rval


    def get_cd_neg_phase_grads(self, V, H_hat):

        assert self.use_cd
        assert not hasattr(self, 'V_chains')

        assert len(H_hat) == len(self.rbms)

        sample_from = Sampler(RandomStreams(17))

        H_samples = []

        for H_hat_elem in H_hat:
            H_samples.append(sample_from(H_hat_elem))

        ip = self.inference_procedure

        for i in xrange(len(H_hat)-1,-1,-1):

            if i > 0:
                P = ip.infer_H_hat_two_sided(H_hat_below = H_samples[i-1], H_hat_above = H_samples[i+1],
                        W_below = self.W[i], W_above = self.W[i+1], b = self.bias_hid[i])
                H_samples[i] = sample_from(P)

        if len(H_hat) > 1:
            H_hat[0] = sample_from(ip.infer_H_hat_two_sided(H_hat_below = V, H_hat_above = H_samples[1],
                    W_below = self.W[0], W_above = self.W[1], b = self.bias_hid[0]))

        V_sample = sample_from(ip.infer_H_hat_one_sided(other_H_hat = H_hat[0], W = self.W[0].T, b = self.bias_vis))

        return self.get_neg_phase_grads_from_samples(V_sample, H_samples)

    def get_neg_phase_grads(self):
        """ returns a dictionary mapping from parameters to negative phase gradients
            (assuming you're doing gradient ascent on variational free energy)
        """

        assert not self.use_cd

        return self.get_neg_phase_grads_from_samples(self.V_chains, self.H_chains)

    def get_neg_phase_grads_from_samples(self, V_sample, H_samples):

        H_rao_blackwell = self.rao_blackwellize(V_sample, H_samples)

        obj = self.expected_energy(V_hat = V_sample, H_hat = H_rao_blackwell)

        constants = list(set(H_rao_blackwell).union([V_sample]))

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

        rval = list(rval)

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

    def random_design_matrix(self, batch_size, theano_rng):
        raise NotImplementedError()

        #return V_sample

    def expected_energy(self, V_hat, H_hat, no_v_bias = False):
        """ expected energy of the model under the mean field distribution
            defined by V_hat and H_hat
            alternately, could be expectation of the energy function across
            a batch of examples, where every element of V_hat and H_hat is
            a binary observation
            if no_v_bias is True, ignores the contribution from biases on visible units
        """


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

        rval =  - total

        #rval.name = 'dbm_expected_energy('+V_name+','+str(H_names)+')'

        return rval



    def expected_energy_batch(self, V_hat, H_hat, no_v_bias = False):
        """ expected energy of the model under the mean field distribution
            defined by V_hat and H_hat
            alternately, could be expectation of the energy function across
            a batch of examples, where every element of V_hat and H_hat is
            a binary observation
            if no_v_bias is True, ignores the contribution from biases on visible units
        """

        warnings.warn("TODO: write unit test verifying expected_energy_batch/m = expected_energy")

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

    def learn(self, dataset, batch_size):
        self.learn_mini_batch(dataset.get_batch_design(batch_size))

    def learn_mini_batch(self, X):

        self.learn_func(X)

        if self.monitor.examples_seen % self.print_interval == 0:
            self.print_status()


    def get_weights_format(self):
        return self.rbms[0].get_weights_format()

class InferenceProcedure:
    """

        Variational inference

        """

    def get_monitoring_channels(self, V, model):

        rval = {}

        if self.monitor_kl or self.monitor_em_functional:
            obs_history = self.infer(V, return_history = True)

            for i in xrange(1, 2 + len(self.h_new_coeff_schedule)):
                obs = obs_history[i-1]
                if self.monitor_kl:
                    rval['trunc_KL_'+str(i)] = self.truncated_KL(V, model, obs).mean()
                if self.monitor_em_functional:
                    rval['em_functional_'+str(i)] = self.em_functional(V, model, obs).mean()

        return rval


    def __init__(self, monitor_kl = False):
        self.autonomous = False
        self.model = None
        self.monitor_kl = monitor_kl
        #for the current project, DBM need not implement its own inference, so the constructor
        #doesn't need an update schedule, etc.
        #note: can't do monitor_em_functional since Z is not tractable


    def register_model(self, model):
        self.model = model

    def truncated_KL(self, V, obs, no_v_bias = False):
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
        """

        H_hat = obs['H_hat']

        for Hv in get_debug_values(H_hat):
            assert Hv.min() >= 0.0
            assert Hv.max() <= 1.0

        entropy_term = - self.model.entropy_h(H_hat = H_hat)
        assert len(entropy_term.type.broadcastable) == 1
        energy_term = self.model.expected_energy_batch(V_hat = V, H_hat = H_hat, no_v_bias = no_v_bias)
        assert len(energy_term.type.broadcastable) == 1

        KL = entropy_term + energy_term

        return KL


    def infer_H_hat_two_sided(self, H_hat_below, W_below, H_hat_above, W_above, b):

        bottom_up = T.dot(H_hat_below, W_below)
        top_down =  T.dot(H_hat_above, W_above.T)
        total = bottom_up + top_down + b

        H_hat = T.nnet.sigmoid(total)

    def infer_H_hat_one_sided(self, other_H_hat, W, b):
        """ W should be arranged such that other_H_hat.shape[1] == W.shape[0] """

        dot = T.dot(other_H_hat, W)
        presigmoid = dot + b

        H_hat = T.nnet.sigmoid(presigmoid)

        return H_hat

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


        raise NotImplementedError("This method is not implemented yet. The code in this file is just copy-pasted from S3C")

        #NOTE: I don't think this method needs to be implemented for the current project

        """
        alpha = self.model.alpha


        var_s0_hat = 1. / alpha
        var_s1_hat = self.var_s1_hat()


        H   =    self.init_H_hat(V)
        Mu1 =    self.init_S_hat(V)

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

        check_H(H,V)

        def make_dict():

            return {
                    'H_hat' : H,
                    'S_hat' : Mu1,
                    'var_s0_hat' : var_s0_hat,
                    'var_s1_hat': var_s1_hat,
                    }

        history = [ make_dict() ]

        for new_H_coeff, new_S_coeff in zip(self.h_new_coeff_schedule, self.s_new_coeff_schedule):

            new_Mu1 = self.infer_S_hat(V, H, Mu1)

            if self.clip_reflections:
                clipped_Mu1 = reflection_clip(Mu1 = Mu1, new_Mu1 = new_Mu1, rho = self.rho)
            else:
                clipped_Mu1 = new_Mu1
            Mu1 = self.damp(old = Mu1, new = clipped_Mu1, new_coeff = new_S_coeff)
            new_H = self.infer_H_hat(V, H, Mu1)

            H = self.damp(old = H, new = new_H, new_coeff = new_H_coeff)

            check_H(H,V)

            history.append(make_dict())

        if return_history:
            return history
        else:
            return history[-1]
        """

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



