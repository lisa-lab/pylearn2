==============================
Pylearn2: A machine learning research library
==============================

**Warning** :

   This project does not have any current developer. We will continue
   to review pull requests and merge them when appropriate, but do not
   expect new development unless someone decides to work on it.

   There are other machine learning frameworks built on top of Theano that
   could interest you, such as: `Blocks
   <https://blocks.readthedocs.org/en/latest>`_, `Keras
   <http://keras.io>`_ and `Lasagne
   <https://lasagne.readthedocs.org/en/latest>`_.


Pylearn2 is a library designed to make machine learning research easy.

Pylearn2 has online `documentation <http://deeplearning.net/software/pylearn2/>`_.
If you want to build a local copy of the documentation, run

    python ./doc/scripts/docgen.py

More documentation is available in the form of commented examples scripts
and ipython notebooks in the "pylearn2/scripts/tutorials" directory.

Pylearn2 was initially developed by David
Warde-Farley, Pascal Lamblin, Ian Goodfellow and others during the winter
2011 offering of `IFT6266 <http://www.iro.umontreal.ca/~pift6266/>`_, and
is now developed by the LISA lab.


Quick start and basic design rules
------------------
- Installation instructions are available `here <http://deeplearning.net/software/pylearn2/#download-and-installation>`_.
- Subscribe to the `pylearn-users Google group
  <http://groups.google.com/group/pylearn-users>`_ for important updates. Please write
  to this list for general inquiries and support questions.
- Subscribe to the `pylearn-dev Google group
  <http://groups.google.com/group/pylearn-dev>`_ for important development updates. Please write
  to this list if you find any bug or want to contribute to the project.
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
- Pylearn2 was used to set the state of the art on MNIST, CIFAR-10, CIFAR-100, and SVHN.
  See pylearn2.models.maxout or pylearn2/scripts/papers/maxout
- Pylearn2 provides a wrapper around Alex Krizhevsky's extremely efficient GPU convolutional
  network library. This wrapper lets you use Theano's symbolic differentiation and other
  capabilities with minimal overhead. See pylearn2.sandbox.cuda_convnet.

License and Citations
---------------------
Pylearn2 is released under the 3-claused BSD license, so it may be used for commercial purposes.
The license does not require anyone to cite Pylearn2, but if you use Pylearn2 in published research
work we encourage you to cite this article:

- Ian J. Goodfellow, David Warde-Farley, Pascal Lamblin, Vincent Dumoulin,
  Mehdi Mirza, Razvan Pascanu, James Bergstra, Frédéric Bastien, and
  Yoshua Bengio.
  `"Pylearn2: a machine learning research library"
  <http://arxiv.org/abs/1308.4214>`_.
  *arXiv preprint arXiv:1308.4214* (`BibTeX
  <http://www.iro.umontreal.ca/~lisa/publications2/index.php/export/publication/594/bibtex>`_)
