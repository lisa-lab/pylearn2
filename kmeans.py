"""KMeans as a postprocessing Block subclass."""

import numpy
from framework.base import Block

class KMeans(Block):
    """
    Block that outputs a vector of probabilities that a sample belong to means
    computed during training.
    """

    def __init__(self, k):
        """
        Parameters in conf:

        :type k: int
        :param k: number of clusters.
        """
        self.k = k

    def train(self,X):
        """
        Process kmeans algorithm on the input to localize clusters.
        """
        n,m = X.shape
        k = self.k

        #computing initial clusters
        mu = numpy.zeros((k,m))
        for i in xrange(k):
            mu[i,:] = X[i:n:k,:].mean(axis=0)

        dists = numpy.zeros((n,k))

        killed_on_prev_iter = False
        old_kills = {}

        iter = 0
        while True:
            #if numpy.sum(numpy.isnan(mu)) > 0:
            if True in numpy.isnan(mu):
                print 'nan found'
                return X

            #computing distances
            for i in xrange(k):
                dists[:,i] = numpy.square((X - mu[i,:])).sum(axis=1)

            if iter > 0:
                prev_mmd = mmd

            min_dists = dists.min(axis=1)

            #mean minimum distance:
            mmd = min_dists.mean()

            if iter > 0 and (not killed_on_prev_iter) and abs(mmd-prev_mmd)<.1:
                #converged
                break

            #finding minimum distances
            min_dist_inds = dists.argmin(axis=1)

            #computing means
            i = 0
            blacklist = []
            new_kills = {}
            killed_on_prev_iter = False
            while i < k:
                b = min_dist_inds == i
                if not numpy.any(b):
                    killed_on_prev_iter = True
                    #initializes empty cluster to be the mean of the d data
                    #points farthest from their corresponding means
                    if i in old_kills:
                        d = old_kills[i]-1
                        if d == 0:
                            d = 50
                        new_kills[i] = d
                    else:
                        d = 5
                    mu[i,:] = 0
                    for j in xrange(d):
                        idx = numpy.argmax(min_dists)
                        min_dists[idx] = 0
                        #chose point idx
                        mu[i,:] += X[idx,:]
                        blacklist.append(idx)
                    mu[i,:] /= float(d)
                    #cluster i was empty, reset it to d far out data points
                    #recomputing distances for this cluster
                    dists[:,i] = numpy.square((X-mu[i,:])).sum(axis=1)
                    min_dists = dists.min(axis=1)
                    for idx in blacklist:
                        min_dists[idx] = 0
                    min_dist_inds = dists.argmin(axis=1)
                    #done
                    i += 1
                else:
                    mu[i,:] = numpy.mean( X[b,: ] ,axis=0)
                    if True in numpy.isnan(mu):
                        print 'nan found at i'
                        return X
                    i += 1

            old_kills = new_kills

            iter += 1
        self.mu=mu

    def __call__(self, X):
        """
        Compute for each sample its probability to belong to a cluster.

        :type inputs: numpy.ndarray, shape (n, d)
        :param inputs: matrix of samples
        """
        n,m=X.shape
        k=self.k
        mu=self.mu
        dists = numpy.zeros((n,k))
        for i in xrange(k):
            dists[:,i] = numpy.square((X - mu[i,:])).sum(axis=1)
        return dists/dists.sum(axis=1).reshape(-1,1)

if __name__=='__main__':
    import theano
    from theano import tensor
    from framework.corruption import GaussianCorruptor
    from framework.autoencoder import DenoisingAutoencoder
    from framework.cost import MeanSquaredError
    from framework.optimizer import SGDOptimizer
    # toy labeled data: [x,y,label]*n samples
    n=50
    rng = numpy.random.RandomState(seed=7777777)
    noise = rng.random_sample((n,2))
    class1=numpy.concatenate((  noise*10+numpy.array([-10,-10]),
                                numpy.array([[1]*n]).T),
                                axis=1)
    class2=numpy.concatenate((  noise*10+numpy.array([10,10]),
                                numpy.array([[2]*n]).T),
                                axis=1)
    data=numpy.append(class1,class2,axis=0)
    rng.shuffle(data)
    #labels are just going to be used as visual reference in terminal output
    train_data,train_labels=data[:-10,:-1],data[:-10,-1]
    test_data,test_labels=data[-10:,:-1],data[-10:,-1]
    print train_data.shape
    print test_data.shape

    #train an SDG on it, as in the first part of example_da.py
    conf = {
        'corruption_level': 0.1,
        'nhid': 3,
        'nvis': train_data.shape[1],
        'anneal_start': 100,
        'base_lr': 0.01,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': None,
        #'lr_hb': 0.10,
        #'lr_vb': 0.10,
        'irange': 0.001,
        #note the kmean hyper-parameter here
        'kmeans_k':2
    }
    print '== training =='
    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Allocate a denoising autoencoder with binomial noise corruption.
    corruptor = GaussianCorruptor(corruption_level=conf['corruption_level'])
    da = DenoisingAutoencoder(conf['nvis'], conf['nhid'], corruptor,
                              conf['act_enc'], conf['act_dec'],
                              tied_weights=conf['tied_weights'],
                              irange=conf['irange'])

    # Allocate an optimizer, which tells us how to update our model.
    # TODO: build the cost another way
    cost = MeanSquaredError(da)(minibatch, da.reconstruction(minibatch))
    trainer = SGDOptimizer(da.params(), conf['base_lr'], conf['anneal_start'])

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], cost,
                               updates=trainer.cost_updates(cost))

    # Suppose we want minibatches of size 10
    batchsize = 10

    # Here's a manual training loop. I hope to have some classes that
    # automate this a litle bit.
    for epoch in xrange(10):
        for offset in xrange(0, train_data.shape[0], batchsize):
            minibatch_err = train_fn(train_data[offset:(offset + batchsize)])
            #print "epoch %d, batch %d-%d: %f" % \
                    #(epoch, offset, offset + batchsize - 1, minibatch_err)

    # Suppose you then want to use the representation for something.
    transform = theano.function([minibatch], da([minibatch])[0])

    #then train & apply kmeans as a postprocessing
    kmeans=KMeans(conf['kmeans_k'])
    kmeans.train(transform(train_data))

    print '== testing =='
    output=kmeans(transform(test_data))
    print 'output:',output.shape
    for i,sample in enumerate(test_data):
        print sample,'/',test_labels[i],'->',output[i]

    #print "Transformed data:"
    #print numpy.histogram(transform(data))

