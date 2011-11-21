==============================
Pylearn experimental framework
==============================

This repository contains the experimental framework, developed by David
Warde-Farley, Pascal Lamblin, Ian Goodfellow and others during the winter
2011 offering of `IFT6266 <http://www.iro.umontreal.ca/~pift6266/>`_.

The Pylearn2 `documentations <http://deeplearning.net/software/pylearn2/>`_.

Basic design rules
------------------

- There are `examples <https://github.com/lisa-lab/pylearn/tree/master/pylearn2/scripts/train_example>`_.
  They cover how to create a dataset, how to train and how to inspect the model.

- Models should implement the Block interface.
- Methods on these models should expect Theano expressions as arguments and
  return Theano variables (except in a few select cases where the modules are
  trained once and never, ever fine-tuned, for example the K-Means module)
- Parameters should be Theano shared variables.
- Other than the shared variables, the methods should have no side effects in
  the form of storing intermediate state.
- Penalties and whatnot should be generally kept separate from the actual model
  class, and implemented as subclasses of the "Cost" object. We break this rule
  currently in the case of ContractingAutoencoder because the model is
  intimately tied to the penalty, and just have a contracting_penalty() method
  that returns the relevant bits to be added to the cost function.
- Subscribe to the `pylearn-dev Google group
  <http://groups.google.com/group/pylearn-dev>`_ for important updates.
