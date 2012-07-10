import numpy as np
from pylearn2.distributions.mnd import MND
from theano import function

def test_seed_same():
    """Verifies that two MNDs initialized with the same
    seed produce the same samples """

    rng = np.random.RandomState([1,2,3])

    #the number in the argument here is the limit on
    #seed value
    seed = rng.randint(2147462579)

    dim = 3

    mu = rng.randn(dim)

    rank = dim

    X = rng.randn(rank,dim)

    cov = np.dot(X.T,X)

    mnd1 = MND( sigma = cov, mu = mu, seed = seed)

    num_samples = 5

    rd1 = mnd1.random_design_matrix(num_samples)
    rd1 = function([],rd1)()

    mnd2 = MND( sigma = cov, mu = mu, seed = seed)

    rd2 = mnd2.random_design_matrix(num_samples)
    rd2 = function([],rd2)()

    assert np.all(rd1 == rd2)


def test_seed_diff():
    """Verifies that two MNDs initialized with different
    seeds produce samples that differ at least somewhat
    (theoretically the samples could match even under
    valid behavior but this is extremely unlikely)"""

    rng = np.random.RandomState([1,2,3])

    #the number in the argument here is the limit on
    #seed value, and we subtract 1 so it will be
    #possible to add 1 to it for the second MND
    seed = rng.randint(2147462579) -1

    dim = 3

    mu = rng.randn(dim)

    rank = dim

    X = rng.randn(rank,dim)

    cov = np.dot(X.T,X)

    mnd1 = MND( sigma = cov, mu = mu, seed = seed)

    num_samples = 5

    rd1 = mnd1.random_design_matrix(num_samples)
    rd1 = function([],rd1)()

    mnd2 = MND( sigma = cov, mu = mu, seed = seed + 1)

    rd2 = mnd2.random_design_matrix(num_samples)
    rd2 = function([],rd2)()

    assert np.any(rd1 != rd2)
