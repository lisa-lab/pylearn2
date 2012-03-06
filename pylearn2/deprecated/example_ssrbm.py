import numpy
import theano
from theano import tensor
from pylearn2.rbm import (mu_pooled_ssRBM, PersistentCDSampler,
        training_updates)
from pylearn2.optimizer import SGDOptimizer
from pylearn2.rbm_tools import compute_log_z, compute_nll

import utils.debug


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the demo of mu-ssRBM '
                                     'with pooling, plotting '
                                     'results at the end (by default)')
    parser.add_argument('--no-plot', action='store_const',
                        default=False, const=True,
                        required=False, help='Disable plotting of results.')
    args = parser.parse_args()
    data_rng = numpy.random.RandomState(seed=999)
    data = data_rng.normal(size=(500, 20)).astype(theano.config.floatX)

    conf = {
        'nvis': 20,
        'nhid': 30,
        'n_s_per_h': 1, # pooling factor
        'rbm_seed': 1,
        'batch_size': 100,
        'base_lr': 1e-2,
        'anneal_start': None,
        'pcd_steps': 1,
        # Initialization and constraints
        'alpha0': 5,
        'alpha_irange': 0, ##
        'alpha_min': 1,
        'alpha_max': 100,
        'b0': 0,
        'B0': 10, ##
        'B_min': 1.00,
        'B_max': 101.0,
        'Lambda0': 0.00001, ##???
        'Lambda_min': 0.00001,
        'Lambda_max': 10,
        'Lambda_irange': 0,
        'mu0': 1,
        'particles_min': -30,
        'particles_max': 30,
    }
    conf['W_irange'] = 2 / numpy.sqrt(conf['nvis'] * conf['nhid'])

    rng = numpy.random.RandomState(seed=conf.get('rbm_seed', 42))

    rbm = mu_pooled_ssRBM(
            nvis=conf['nvis'],
            nhid=conf['nhid'],
            n_s_per_h=conf['n_s_per_h'],
            batch_size=conf['batch_size'],
            alpha0=conf['alpha0'],
            alpha_irange=conf['alpha_irange'],
            b0=conf['b0'],
            B0=conf['B0'],
            Lambda0=conf['Lambda0'],
            Lambda_irange=conf['Lambda_irange'],
            mu0=conf['mu0'],
            W_irange=conf['W_irange'],
            rng=rng)

    sampler = PersistentCDSampler(
            rbm,
            data[0:conf['batch_size']],
            rng,
            steps=conf['pcd_steps'],
            particles_clip=(conf['particles_min'], conf['particles_max']),
            )
    minibatch = tensor.matrix()

    optimizer = SGDOptimizer(
            rbm,
            conf['base_lr'],
            conf['anneal_start'],
            log_alpha_clip=(numpy.log(conf['alpha_min']), numpy.log(conf['alpha_max'])),
            B_clip=(conf['B_min'], conf['B_max']),
            Lambda_clip=(conf['Lambda_min'], conf['Lambda_max']),
            )
    updates = training_updates(visible_batch=minibatch, model=rbm,
                               sampler=sampler, optimizer=optimizer)

    proxy_cost = rbm.reconstruction_error(minibatch, rng=sampler.s_rng)
    train_fn = theano.function([minibatch], proxy_cost, updates=updates)

    vis = tensor.matrix('vis')
    free_energy_fn = theano.function([vis], rbm.free_energy_given_v(vis))

    utils.debug.setdebug()

    recon = []
    nlls = []
    for j in range(0, 401):
        avg_rec_error = 0

        for i in range(0, 500, 100):
            rec_error = train_fn(data[i:i+100])
            recon.append(rec_error / 100)
            avg_rec_error = (i*avg_rec_error + rec_error) / (i+100)
        print "Epoch %d: avg_rec_error = %f" % (j+1, avg_rec_error)

        if (j%50)==0:
            log_z = compute_log_z(rbm, free_energy_fn)
            nll = compute_nll(rbm, data, log_z, free_energy_fn)
            nlls.append(nll)
            print "Epoch %d: avg_nll = %f" % (j+1, nll)
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.subplot(2, 1, 1)
        plt.plot(range(len(recon)), recon)
        plt.xlim(0, len(recon))
        plt.title('Reconstruction error per minibatch')
        plt.xlabel('minibatch number')
        plt.ylabel('reconstruction error')
        plt.subplot(2, 1, 2)
        plt.plot(range(0, len(nlls) * 50, 50), nlls, '-d')
        plt.xlabel('Epoch')
        plt.ylabel('Average nll per data point')
        plt.show()
