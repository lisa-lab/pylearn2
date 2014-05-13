"""
Tests for the show_examples.py script
"""
import os

from pylearn2.testing.skip import skip_if_no_matplotlib, skip_if_no_data
from pylearn2.scripts.show_examples import show_examples


def test_show_examples():
    """
    Create a YAML file of the MNIST dataset and show examples
    """
    skip_if_no_matplotlib()
    skip_if_no_data()
    with open('temp.yaml', 'w') as f:
        f.write("""
!obj:pylearn2.datasets.mnist.MNIST {
        which_set: 'train'
}
""")
    show_examples('temp.yaml', 28, 28, out='garbage.png')
    os.remove('temp.yaml')
