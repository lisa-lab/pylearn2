This directory contains a simple example to give you a feeling of how you can
use pylearn2 by writing your own python scripts.

Note that the best-supported method is to write yaml configuration files (see
scripts/train_example and tutorials/*.ipynb for examples of this method).
 
It is by no means a full-blown demo and largely still is work in progress. 

usage:

make sure pylearn2 can be found in your PYTHONPATH

to train models on toy dataset: python run_deep_trainer.py -d toy

to train models on cifar10 dataset: python run_deep_trainer.py -d cifar10

To visualize the first-layer weights trained on cifar10:

epoch 0: show_weights.py cifar10_grbm0_epoch.pkl
epoch 1: show_weights.py cifar10_grbm1_epoch.pkl
etc.

(Visualization of deeper layer weights is not conceptually straightforward)

