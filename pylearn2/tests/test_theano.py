from theano import tensor as T

def test_grad():
    """Tests that the theano grad method returns a list if it is passed a list
        and a single variable if it is passed a single variable.
       pylearn2 depends on theano behaving this way but theano developers have
       repeatedly changed it """

    X = T.matrix()
    y = X.sum()

    G = T.grad(y, [X])

    assert isinstance(G,list)

    G = T.grad(y, X)

    assert not isinstance(G,list)
