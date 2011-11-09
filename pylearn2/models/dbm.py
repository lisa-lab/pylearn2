__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2011, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"


import time
from pylearn2.models.model import Model
from theano import config, function, shared
import theano.tensor as T
import numpy as np
import warnings
from theano.gof.op import get_debug_values, debug_error_message
from pylearn2.utils import make_name, sharedX, as_floatX

warnings.warn('s3c changing the recursion limit')
import sys
sys.setrecursionlimit(50000)

class SufficientStatistics:
    """ The SufficientStatistics class computes several sufficient
        statistics of a minibatch of examples / variational parameters.
        This is mostly for convenience since several expressions are
        easy to express in terms of these same sufficient statistics.
        Also, re-using the same expression for the sufficient statistics
        in multiple code locations can reduce theano compilation time.
        The current version of the S3C code no longer supports features
        like decaying sufficient statistics since these were not found
        to be particularly beneficial relative to the burden of computing
        the O(nhid^2) second moment matrix. The current version of the code
        merely computes the sufficient statistics apart from the second
        moment matrix as a notational convenience. Expressions that most
        naturally are expressed in terms of the second moment matrix
        are now written with a different order of operations that
        avoids O(nhid^2) operations but whose dependence on the dataset
        cannot be expressed in terms only of sufficient statistics."""


    def __init__(self, d):
        self. d = {}
        for key in d:
            self.d[key] = d[key]

    @classmethod
    def from_observations(self, needed_stats, V, H_hat, S_hat, var_s0_hat, var_s1_hat):
        """
            returns a SufficientStatistics

            needed_stats: a set of string names of the statistics to include

            V: a num_examples x nvis matrix of input examples
            H_hat: a num_examples x nhid matrix of \hat{h} variational parameters
            S_hat: variational parameters for expectation of s given h=1
            var_s0_hat: variational parameters for variance of s given h=0
                        (only a vector of length nhid, since this is the same for
                        all inputs)
            var_s1_hat: variational parameters for variance of s given h=1
                        (again, a vector of length nhid)
        """

        m = T.cast(V.shape[0],config.floatX)

        H_name = make_name(H_hat, 'anon_H_hat')
        Mu1_name = make_name(S_hat, 'anon_S_hat')

        #mean_h
        assert H_hat.dtype == config.floatX
        mean_h = T.mean(H_hat, axis=0)
        assert H_hat.dtype == mean_h.dtype
        assert mean_h.dtype == config.floatX
        mean_h.name = 'mean_h('+H_name+')'

        #mean_v
        mean_v = T.mean(V,axis=0)

        #mean_sq_v
        mean_sq_v = T.mean(T.sqr(V),axis=0)

        #mean_s1
        mean_s1 = T.mean(S_hat,axis=0)

        #mean_sq_s
        mean_sq_S = H_hat * (var_s1_hat + T.sqr(S_hat)) + (1. - H_hat)*(var_s0_hat)
        mean_sq_s = T.mean(mean_sq_S,axis=0)

        #mean_hs
        mean_HS = H_hat * S_hat
        mean_hs = T.mean(mean_HS,axis=0)
        mean_hs.name = 'mean_hs(%s,%s)' % (H_name, Mu1_name)
        mean_s = mean_hs #this here refers to the expectation of the s variable, not s_hat
        mean_D_sq_mean_Q_hs = T.mean(T.sqr(mean_HS), axis=0)

        #mean_sq_hs
        mean_sq_HS = H_hat * (var_s1_hat + T.sqr(S_hat))
        mean_sq_hs = T.mean(mean_sq_HS, axis=0)
        mean_sq_hs.name = 'mean_sq_hs(%s,%s)' % (H_name, Mu1_name)

        #mean_sq_mean_hs
        mean_sq_mean_hs = T.mean(T.sqr(mean_HS), axis=0)
        mean_sq_mean_hs.name = 'mean_sq_mean_hs(%s,%s)' % (H_name, Mu1_name)

        #mean_hsv
        sum_hsv = T.dot(mean_HS.T,V)
        sum_hsv.name = 'sum_hsv<from_observations>'
        mean_hsv = sum_hsv / m


        d = {
                    "mean_h"                :   mean_h,
                    "mean_v"                :   mean_v,
                    "mean_sq_v"             :   mean_sq_v,
                    "mean_s"                :   mean_s,
                    "mean_s1"               :   mean_s1,
                    "mean_sq_s"             :   mean_sq_s,
                    "mean_hs"               :   mean_hs,
                    "mean_sq_hs"            :   mean_sq_hs,
                    "mean_sq_mean_hs"       :   mean_sq_mean_hs,
                    "mean_hsv"              :   mean_hsv,
                }


        final_d = {}

        for stat in needed_stats:
            final_d[stat] = d[stat]
            final_d[stat].name = 'observed_'+stat

        return SufficientStatistics(final_d)


warnings.warn("""
TODO/NOTES
The sampler ought to store the state of all but the topmost hidden layer
learning updates will be based on marginalizing out this topmost layer
to reduce noise a bit
each round of negative phase sampling should start by sampling the topmost
layer, then sampling downward from there
when taking params from rbms, each rbm should donate its visible biases
and weights. only the topmost rbm additionally donates its hidden biases
""")

class DBM(Model):

    def __init__(self, rbms,
                       inference_procedure,
                       print_interval = 10000):
        """
            rbms: list of rbms to stack
                    all rbms must be of type pylearn2.models.rbm, and not a subclass
                    first entry is the visible rbm
                    DBM may destroy these rbms-- it won't delete them,
                    but it may do terrible things to them
            inference_procedure: a pylearn2.models.dbm.InferenceProcedure object
            print_interval: every print_interval examples, print out a status summary

        """

        warnings.warn("""The DBM class is still under development, and currently mostly
                only supports use as a component of a larger model that remains
                private. Contact Ian Goodfellow if you have questions about the current
                status of this class.""")

        super(DBM,self).__init__()

        self.rbms = rbms

        self.inference_procedure = inference_procedure
        self.inference_procedure.register_model(self)

        self.print_interval = print_interval

        self.redo_everything()


    def redo_everything(self):

        self.redo_theano()

    def get_monitoring_channels(self, V):
        warnings.warn("DBM doesn't actually return any monitoring channels yet. It has a bunch of S3C code sitting in its get_monitoring_channels but for now it just returns an empty dictionary")

        return {}

        try:
            self.compile_mode()

            rval = self.m_step.get_monitoring_channels(V, self)

            from_e_step = self.e_step.get_monitoring_channels(V, self)

            rval.update(from_e_step)

            monitor_stats = len(self.monitor_stats) > 0

            if monitor_stats:

                obs = self.e_step.variational_inference(V)

                needed_stats = set(self.monitor_stats)


                stats = SufficientStatistics.from_observations( needed_stats = needed_stats,
                                                            V = V, ** obs )


                if monitor_stats:
                    for stat in self.monitor_stats:
                        stat_val = stats.d[stat]

                        rval[stat+'_min'] = T.min(stat_val)
                        rval[stat+'_mean'] = T.mean(stat_val)
                        rval[stat+'_max'] = T.max(stat_val)
                    #end for stat
                #end if monitor_stats
            #end if monitor_stats or monitor_functional


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

    def get_params(self):
        rval = set([])

        for rbm in self.rbms:
            rval = rval.union(set(rbm.get_params))

        rval = list(rval)

        return rval

    def make_learn_func(self, V):
        """
        V: a symbolic design matrix
        """

        raise NotImplementedError("Not yet supported-- current project does not require DBM to learn on its own")

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

    def censor_updates(self, updates):

        for rbm in self.rbms:
            rbm.censor_updates(updates)

    def random_design_matrix(self, batch_size, theano_rng):
        raise NotImplementedError()

        return V_sample

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


    def __init__(self, monitor_kl):
        self.model = None
        self.monitor_kl = monitor_kl
        #for the current project, DBM need not implement its own inference, so the constructor
        #doesn't need an update schedule, etc.
        #note: can't do monitor_em_functional since Z is not tractable


    def register_model(self, model):
        self.model = model

    def truncated_KL(self, V, model, obs):
        """ KL divergence between variation and true posterior, dropping terms that don't
            depend on the variational parameters """

        raise NotImplementedError("This method is not implemented yet. The code in this file is just copy-pasted from S3C")

        H_hat = obs['H_hat']
        var_s0_hat = obs['var_s0_hat']
        var_s1_hat = obs['var_s1_hat']
        S_hat = obs['S_hat']

        entropy_term = - model.entropy_hs(H_hat = H_hat, var_s0_hat = var_s0_hat, var_s1_hat = var_s1_hat)
        energy_term = model.expected_energy_vhs(V, H_hat = H_hat, S_hat = S_hat,
                                        var_s0_hat = var_s0_hat, var_s1_hat = var_s1_hat)

        KL = entropy_term + energy_term

        return KL


    def infer_H_hat_two_sided(self, H_hat_below, W_below, H_hat_above, W_above, b):
        raise NotImplementedError()


    def infer_H_hat_one_sided(self, other_H_hat, W, b):
        raise NotImplementedError()

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


