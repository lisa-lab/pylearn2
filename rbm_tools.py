import numpy
import theano

from theano import tensor

def compute_log_z(rbm, free_energy_fn, max_bits=15):

    # pick whether to iterate over visible or hidden states
    if rbm.conf['n_vis'] < rbm.conf['n_hid']:
        width = rbm.conf['n_vis']
        type = 'vis'
    else:
        width = rbm.conf['n_hid']
        type = 'hid'

    # determine in how many steps to compute Z
    block_bits = width if (not max_bits or width < max_bits) else max_bits
    block_size = 2**block_bits

    # allocate storage for 2**block_bits of the 2**width possible configurations
    logz_data_c = numpy.zeros((block_size, width), order='C', dtype=theano.config.floatX)

    # fill in the first block_bits, which will remain fixed for all 2**width configs
    tensor_10D_idx = numpy.ndindex(*([2]*block_bits))
    for i, j in enumerate(tensor_10D_idx):
        logz_data_c[i, -block_bits:] = j
    logz_data = numpy.array(logz_data_c, order='F', dtype=theano.config.floatX)

    # storage for free-energy of all 2**width configurations
    FE = numpy.zeros(2**width, dtype=theano.config.floatX)

    # now loop 2**(width - block_bits) times, filling in the most-significant bits
    for bi, upper_bits in enumerate(numpy.ndindex(*([2]*(width-block_bits)))):
        logz_data[:, :width-block_bits] = upper_bits
        FE[bi*block_size:(bi+1)*block_size] = free_energy_fn(logz_data)

    alpha = numpy.min(FE)
    log_z = numpy.log(numpy.sum(numpy.exp(-FE - alpha))) + alpha

    return log_z


def compute_nll(rbm, data, log_z, free_energy_fn, bufsize=1000, preproc=None):

    i = 0.
    nll = 0

    for i in xrange(0, len(data), bufsize):

        # recast data as floatX and apply preprocessing if required
        x = numpy.array(data[i:i+bufsize, :], dtype=theano.config.floatX)
        if preproc:
            x = preproc(x)
       
        # compute sum of likelihood for current buffer
        x_nll = numpy.sum(-free_energy_fn(x) - log_z)

        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        nll = (i*nll + x_nll) / (i + len(x))

    return nll 
