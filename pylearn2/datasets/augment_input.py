from pylearn2.utils import sharedX
from theano import function
import numpy


def augment_input(X, model, mf_steps):

    print '\nAugmenting data...\n'

    i = 0
    init_data = model.visible_layer.space.get_origin_batch(batch_size = 1, dtype = 'float32')

    for x in X[:]:
        init_data[0] = x
        data = sharedX(init_data, name = 'v')
        marginal_posterior = model.mf(V = data, niter = mf_steps)[1]  # mean field inference of second hidden layer (niter: number of mean field updates)
        mp = function([], marginal_posterior)
        mp = mp()[0][0]
        if i == 0:
            final_data = numpy.asarray([numpy.concatenate((mp, x))])
        else:
            final_data = numpy.append(final_data, [numpy.concatenate((mp, x))], axis = 0)

        i += 1

    print 'Data augmentation complete!'

    return final_data