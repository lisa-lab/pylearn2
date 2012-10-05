"""

LinearTransform classes and convenience methods for creating them.
LinearTransform classes are used to linearly transform between vector
spaces. By instantiating different derived classes the same model can
work by dense matrix multiplication, convolution, etc. without needing
to rewrite any of the model's code.

"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
