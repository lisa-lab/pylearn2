import warnings

warnings.warn("pylearn2.pca has been moved to pylearn2.models.pca and "
        "will be removed from the library on or after Aug 24, 2014.")

from models.pca import sys
from models.pca import numpy
from models.pca import theano
import scipy
from models.pca import linalg
from models.pca import sparse
from models.pca import eigen_symmetric
from models.pca import csr_matrix
# Theano
from models.pca import tensor
from models.pca import SparseType
from models.pca import structured_dot

# Local imports
from models.pca import Block
from models.pca import sharedX

# Classes
#from models.pca import _PCABase
from models.pca import SparseMatPCA
from models.pca import OnlinePCA
from models.pca import Cov
from models.pca import CovEigPCA
from models.pca import SVDPCA
from models.pca import SparsePCA
from models.pca import PcaOnlineEstimator

# These imports were in if statements
import argparse
from pylearn2.utils import load_data