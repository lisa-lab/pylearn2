import theano


class SearchDirection(object):
    """
    Given a set of parameters and a gradient of some cost with respect
    to the parameters, computes the updates dictionary based on a
    transformation of the gradient.

    The transformation can be the identity function (in which case the updates
    represent the standard gradient descent), or a function of the gradient and
    the number of previous updates (which can lead to annealing), or even a
    function of all previous gradients (momentum, adagrad, etc.)
    
    Theano functions computing gradients must apply all of these updates in 
    order for successive calls to the function to be correct.
    """
    def dir_from_grad(self, gradients):
        """
        Computes a transformation of the gradient and returns it in the form
        of a dictionary mapping the parameters to their transformed gradient
        (which is called 'search direction' here).
        
        Parameters
        ----------
        gradients: dictionary 
                   Maps parameters (tensor-like theano variables) to gradients
                   (tensor-like theano variables). The only parameter 
                   dir_from_grad should ever accept.
        Returns
        -------
        direction: dictionary 
                   Maps parameters (tensor-like theano variables) to 
                   transformed gradients (tensor-like theano variables)
        updates:   dictionary
                   Maps persistent variables used in the transformation of the
                   gradient to their updated value
        """
        raise NotImplementedError(str(type(self))+" does not implement "+ \
                                  "dir_from_grad.")


class IdentitySD(SearchDirection):
    """
    Computes the identity transformation on the gradient.
    """
    def __init__(self):
        pass

    def dir_from_grad(self, gradients): 
        return gradients, {}
