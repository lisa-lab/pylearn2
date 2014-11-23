from __future__ import print_function

import sys
in_path, out_path = sys.argv[1:]
from scipy import io
import numpy as np
print('loading')
X = np.load(in_path)
if len(X.shape) > 2:
    print('reshaping')
    X = X.reshape(X.shape[0],X.shape[1] * X.shape[2] * X.shape[3])
assert len(X.shape) == 2
print('saving')
io.savemat(out_path,{'X':X})

if X.shape[1] > 14400:
    print('reloading to make sure it worked')
    print('(matlab format can fail for large arrays)')
    X = io.loadmat(out_path)
    print('success')
