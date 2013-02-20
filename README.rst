==============================
Pylearn2: A machine learning research library
==============================

Pylearn2 is a library designed to make machine learning research easy.

Pylearn2 has online `documentation <http://deeplearning.net/software/pylearn2/>`_.
If you want to build a local copy of the documentation, run
    python ./doc/scripts/docgen.py

More documentation is available in the form of ipython notebooks in the "tutorials"
directory, and some example scripts in "scripts/train_example."

Pylearn2 was initially developed by David
Warde-Farley, Pascal Lamblin, Ian Goodfellow and others during the winter
2011 offering of `IFT6266 <http://www.iro.umontreal.ca/~pift6266/>`_, and
is now developed by the LISA lab.


Quick start and basic design rules
------------------
- Installation instructions are available `here <http://deeplearning.net/software/pylearn2/#download-and-installation>`_.
- Subscribe to the `pylearn-dev Google group
  <http://groups.google.com/group/pylearn-dev>`_ for important updates. Please write
  to this list for troubleshooting help or any feedback you have about the library,
  even if you're not a Pylearn2 developer.
- Read through the documentation and examples mentioned above.
- Pylearn2 should not force users to commit to the whole library. If someone just wants
  to implement a Model, they should be able to do that and not need to implement
  a TrainingAlgorithm. Try not to write library features that force users to buy into
  the whole library.
- When writing reference implementations to go in the library, maximize code re-usability
  by decomposing your algorithm into a TrainingAlgorithm that trains a Model on a Dataset.
  It will probably do this by minimizing a Cost. In fact, you can probably use an existing
  TrainingAlgorithm.

Highlights
------------------
- pylearn2 was used to set the state of the art on MNIST, CIFAR-10, CIFAR-100, and SVHN.
  See pylearn2.models.maxout or pylearn2/scripts/papers/maxout
- pylearn2 provides a wrapper around Alex Krizhevsky's extremely efficient GPU convolutional
  network library. This wrapper lets you use Theano's symbolic differentiation and other
  capabilities with minimal overhead. See pylearn2.sandbox.cuda_convnet.


