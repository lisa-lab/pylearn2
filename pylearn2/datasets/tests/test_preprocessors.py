import numpy as np
from pylearn2.utils import as_floatX
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import TorontoPreprocessor, \
    CenterPreprocessor, RescalePreprocessor
from pylearn2.datasets.preprocessing import Pipeline


num_examples = 5
num_features = 10


### Apply new preprocessors
rng = np.random.RandomState([1, 2, 3])
X = as_floatX(rng.randn(num_examples, num_features))
dataset = DenseDesignMatrix(X=X)
preprocessors = []

# Apply Center preprocessing
preprocessors.append(CenterPreprocessor())
# Apply Rescale preprocessing
preprocessors.append(RescalePreprocessor())
# Apply Toronto preprocessing
preprocessors.append(TorontoPreprocessor())

pipeline = Pipeline(preprocessors)
pipeline.apply(dataset, True)
A = dataset.X


### Apply the same kind of preprocessors but using the old way, i.e.
### not using preprocessors defined in preprocessing
rng = np.random.RandomState([1, 2, 3])
B = as_floatX(rng.randn(num_examples, num_features))

# Apply Center preprocessing
B -= 127.5

# Apply Rescale preprocessing
B /= 127.5

# Apply Toronto preprocessing
B = B / 255.
B = B - B.mean(axis=0)


# Test that both datasets are the same
assert np.array_equal(A, B)