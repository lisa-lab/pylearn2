from pylearn2.utils import sharedX
from theano import function
import numpy

"""
This module augments the dataset in order to make it suitable for
DBM discriminative finetuning.
For each example in the dataset, using the provided trained DBM,
it performs n mean-field updates initializing the state of the second
hidden layer of the DBM and augments the example with this state.
It returns a dataset where each example is composed of its previous
value concatenated with the respective initialization of the second
hidden layer of the DBM.
"""

def augment_input(X, model, mf_steps):

    """
    Parameters
    ----------
    X : ndarray, 2-dimensional
        A matrix containing the initial dataset.
    model : DBM
        The DBM model to be finetuned. It is used for
        mean field updates.
    mf_steps : int
        The number of mean field updates.

    Returns
    -------
    final_data : ndarray, 2-dimensional
        The final augmented dataset.

    References
    ----------
    Salakhutdinov Ruslan and Hinton Geoffrey. "An efficient
    procedure for deep boltzmann machines". 2012.
    """

    print("\nAugmenting data...\n")

    i = 0
    init_data = model.visible_layer.space.get_origin_batch(batch_size=1,
                                                           dtype='float32')

    for x in X[:]:
        init_data[0] = x
        data = sharedX(init_data, name='v')
        # mean field inference of second hidden layer
        # (niter: number of mean field updates)
        marginal_posterior = model.mf(V=data, niter=mf_steps)[1]
        mp = function([], marginal_posterior)
        mp = mp()[0][0]
        if i == 0:
            final_data = numpy.asarray([numpy.concatenate((mp, x))])
        else:
            final_data = numpy.append(final_data,
                                      [numpy.concatenate((mp, x))],
                                      axis=0)

        i += 1

    print("Data augmentation complete!")

    return final_data
