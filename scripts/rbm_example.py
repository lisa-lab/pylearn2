import numpy
import matplotlib.pyplot as plt
from theano import tensor
from rbm import GaussianBinaryRBM, PersistentCDSampler
from optimizer import RBMOptimizer

if __name__ == "__main__":
    data = numpy.random.normal(size=(500, 20))
    conf = {
        'n_vis': 20,
        'n_hid': 30,
        'rbm_seed': 1,
        'batch_size': 100,
        'base_lr': 0.01,
        'lr_anneal_start': 200
    }
    rbm = GaussianBinaryRBM(conf)
    sampler = PersistentCDSampler(conf, rbm, data[0:100], numpy.random)
    minibatch = tensor.dmatrix()
    optimizer = RBMOptimizer(conf, rbm, sampler, minibatch)
    train_fn = optimizer.function(minibatch)
    recon = []
    for j in range(0, 400):
        for i in range(0, 500, 100):
            reconstruction = train_fn(data[i:i+100])
            print "%d: %d: %f" % (j+1,i+1, reconstruction)
            recon.append(reconstruction)
    plt.plot(range(len(recon)), recon)
    plt.show()
