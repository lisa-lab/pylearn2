"""

LinearTransform classes and convenience methods for creating them.
LinearTransform classes are used to linearly transform between vector
spaces. By instantiating different derived classes the same model can
work by dense matrix multiplication, convolution, etc. without needing
to rewrite any of the model's code.

"""
