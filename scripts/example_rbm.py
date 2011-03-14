import numpy
import theano
import matplotlib.pyplot as plt
from theano import tensor
from framework.rbm import GaussianBinaryRBM, PersistentCDSampler, \
        training_updates
from framework.optimizer import SGDOptimizer
from framework.rbm_tools import compute_log_z, compute_nll

if __name__ == "__main__":

    data_rng = numpy.random.RandomState(seed=999)
    data = data_rng.normal(size=(500, 20))

    conf = {
        'n_vis': 20,
        'n_hid': 30,
        'rbm_seed': 1,
        'batch_size': 100,
        'base_lr': 0.01,
        'lr_anneal_start': 200,
        'pcd_steps': 1,
    }

    rbm = GaussianBinaryRBM(conf)
    rng = numpy.random.RandomState(seed=conf.get('rbm_seed',42))
    sampler = PersistentCDSampler(conf, rbm, data[0:100], rng)
    minibatch = tensor.matrix()

    optimizer = SGDOptimizer(conf, rbm)
    updates = training_updates(visible_batch=minibatch, model=rbm,
                               sampler=sampler, optimizer=optimizer)
    proxy_cost = rbm.reconstruction_error(minibatch, rng=sampler.s_rng)
    train_fn = theano.function([minibatch], proxy_cost, updates=updates)

    vis = tensor.matrix('vis')
    free_energy_fn = theano.function([vis], rbm.free_energy_given_v(vis))

    recon = []

    for j in range(0, 400):
        avg_rec_error = 0

        for i in range(0, 500, 100):
            rec_error = train_fn(data[i:i+100])
            recon.append(rec_error)
            avg_rec_error = (i*avg_rec_error + rec_error) / (i+100)

        print "Epoch %d: avg_rec_error = %f" % (j+1, avg_rec_error)

        if (j%50)==0:
            log_z = compute_log_z(rbm, free_energy_fn)
            nll = compute_nll(rbm, data, log_z, free_energy_fn)
            print "Epoch %d: avg_nll = %f" % (j+1, nll)

    plt.plot(range(len(recon)), recon)
    plt.show()
