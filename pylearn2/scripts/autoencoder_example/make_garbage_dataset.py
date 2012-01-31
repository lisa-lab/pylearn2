import numpy as np

X = np.random.normal(size=(50000, 300))
np.save('garbage.npy', X)
