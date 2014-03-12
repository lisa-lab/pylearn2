import sys
import logging

in_path, out_path = sys.argv[1:]
from scipy import io
import numpy as np

logger = logging.getLogger(__name__)

logger.info('loading')
X = np.load(in_path)
if len(X.shape) > 2:
    logger.info('reshaping')
    X = X.reshape(X.shape[0],X.shape[1] * X.shape[2] * X.shape[3])
assert len(X.shape) == 2
logger.info('saving')
io.savemat(out_path,{'X':X})

if X.shape[1] > 14400:
    logger.info('reloading to make sure it worked')
    logger.info('(matlab format can fail for large arrays)')
    X = io.loadmat(out_path)
    logger.info('success')
