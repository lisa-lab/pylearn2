
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def DenseMulticlassSVM(C, kernel = 'linear'):
    """ sklearn does very different things behind the scenes depending
        upon the exact identity of the class you use. the only way to
        get an svm implementation that works with dense data is to use
        the SVC class, which implements one-against-one classification.
        this wrapper uses it to implement one-against-rest classification,
        which generally works better in my experiments.

        To avoid duplicating the training data, use only numpy ndarrays
        whose tags.c_contigous flag is true, and which are in float64 format"""

    estimator = SVC(C = C, kernel = kernel)

    return OneVsRestClassifier(estimator)
