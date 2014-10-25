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

Parameters
----------
X : ndarray, 2-dimensional
    a matrix containing the initial dataset
model : DBM
    the DBM model to be finetuned. It is used for
    mean field updates
mf_steps : int
    the number of mean field updates

Returns
-------
final_data : ndarray, 2-dimensional
    the final augmented dataset

References
----------
Salakhutdinov Ruslan and Hinton Geoffrey. "An efficient 
procedure for deep boltzmann machines". 2012.
"""

'''def augment_input(X, model, mf_steps):

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

    return final_data'''

from multiprocessing import Process, Queue

def augment(X, model, mf_steps, q):

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

    q.put(final_data)

def augment_input(X, model, mf_steps):
    
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    q5 = Queue()
    q6 = Queue()
    p1 = Process(target = augment, args = (X, model, mf_steps, q1))
    p2 = Process(target = augment, args = (X, model, mf_steps, q2))
    p3 = Process(target = augment, args = (X, model, mf_steps, q3))
    p4 = Process(target = augment, args = (X, model, mf_steps, q4))
    p5 = Process(target = augment, args = (X, model, mf_steps, q5))
    p6 = Process(target = augment, args = (X, model, mf_steps, q6))
    processes = [p1, p2, p3, p4, p5, p6]
    
    for p in processes:
        p.start()
        
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    
    d1 = q1.get()
    d2 = q2.get()
    d3 = q3.get()
    d4 = q4.get()
    d5 = q5.get()
    d6 = q6.get()
    
    return numpy.concatenate((d1,d2,d3,d4,d5,d6), axis = 0)