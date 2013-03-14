This directory contains a simple example to give you a feeling of how you can
play with pylearn2 by writing your own python scripts to train a deep model.

Note that the best-supported method is to write yaml configuration files (see
scripts/train_example and tutorials/*.ipynb for examples of this method).
 
It is by no means a full-blown demo. 

usage:

make sure pylearn2 can be found in your PYTHONPATH

to train models on toy dataset: python run_deep_trainer.py -d toy

to train models on cifar10 dataset: python run_deep_trainer.py -d cifar10

to train models on mnist dataset: python run_deep_trainer.py -d mnist

To visualize the first-layer weights trained on, for example, mnist:

show_weights.py grbm.pkl

You should be able to see number like filters if you train the first layer GaussianRBM for just a few epochs, provided the layer is of a reasonable size(say, hundreds of hidden units)


To get good classification error you need to player with number of hidden units and MAX_EPOCHS_UNSUPERVISED and MAX_EPOCHS_SUPERVISED


(Visualization of deeper layer weights is not conceptually straightforward)

