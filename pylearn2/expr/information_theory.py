import theano.tensor as T

def entropy_binary_vector(P):

    #TODO: replace with actually evaluating 0 log 0 as 0
    #note: can't do 1e-8, 1.-1e-8 rounds to 1.0 in float32
    H_hat = T.clip(P, 1e-7, 1.-1e-7)

    logP = T.log(P)

    logOneMinusP = T.log(1.-P)

    term1 = - T.sum( P * logP , axis=1)
    assert len(term1.type.broadcastable) == 1

    term2 = - T.sum( (1.-P) * logOneMinusP , axis =1 )
    assert len(term2.type.broadcastable) == 1

    rval = term1 + term2

    return rval
