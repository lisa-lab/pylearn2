"""
Tests of expr.evaluation
"""

import numpy as np

from theano.compat.six.moves import xrange

from pylearn2.expr.evaluation import all_pr


def test_all_pr():
    """
    Tests that all_pr matches a hand-obtained solution.
    """

    pos_scores = [-1., 0., 2.]
    neg_scores = [-2., 0., 1.]

    # scores: [2., 1., 0., 0., -1., -2.]
    # labels: [1,  0,  1,  0,  1,   0 ]

    precision = [1., 1., .5, .5, .6, 3. / 6.]
    recall = [0., 1. / 3., 1. / 3., 2. / 3., 1., 1.]

    p, r = all_pr(pos_scores, neg_scores)
    assert len(p) == len(precision)
    assert len(r) == len(recall)

    # The actual function should do exactly the same arithmetic on
    # integers so we should get exactly the same floating point values
    for i in xrange(len(p)):
        assert p[i] == precision[i], (i, p[i], precision[i])
        assert recall[i] == recall[i]
