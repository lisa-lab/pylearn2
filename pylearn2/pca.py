import warnings

warnings.warn("pylearn2.pca has been moved to pylearn2.models.pca and "
    "will be removed from the library on or after Aug 24, 2014.")

# Make sure old import statements still work
from models.pca import SparseMatPCA
from models.pca import OnlinePCA
from models.pca import Cov
from models.pca import CovEigPCA
from models.pca import SVDPCA
from models.pca import SparsePCA
from models.pca import PcaOnlineEstimator
