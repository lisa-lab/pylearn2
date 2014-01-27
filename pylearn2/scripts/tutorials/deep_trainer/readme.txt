pylearn2 tutorial example: "deep_trainer" by Li Yao

This directory contains a simple example to give you a feeling of how you can
use a few kinds of unsupervised learning to do layerwise pretraining of a
deep model formed by stacking shallow models.

This is not the best-supported tutorial in pylearn2, in part because most of
the active pylearn2 developers are no longer researching stacked layerwise
pretraining.

The recommended usage of pylearn2 is to write YAML configuration files
describing experiments and pass them to the train.py script. This method is
described in most of the other tutorials, such as grbm_smd and the .ipynb
tutorials. This tutorial teaches an alternate method, which is just building
the experiment in your own .py file and using exclusively python interfaces
everywhere. This method is not quite as well supported, and this tutorial
shows you how to make sure the proper steps are taken to make sure datasets
get their correct YAML description when using this method.

You can also read stacked_autoencoders.ipynb, a tutorial on how to do
layerwise pretraining using YAML files. That tutorial only covers autoencoder
pretraining but demonstrates how to use the recommended YAML interface.

usage:

make sure pylearn2 can be found in your PYTHONPATH

to train models on toy dataset: python run_deep_trainer.py -d toy

to train models on cifar10 dataset: python run_deep_trainer.py -d cifar10

to train models on mnist dataset: python run_deep_trainer.py -d mnist

To visualize the first-layer weights trained on, for example, mnist:

show_weights.py grbm.pkl

(Visualization of deeper layer weights is not conceptually straightforward,
so we do not explore that topic in this tutorial)

You should be able to see number-like filters if you train the first layer
GaussianRBM for just a few epochs, provided the layer is of a reasonable
size (say, hundreds of hidden units). Note that by default the hidden layer
is not set to such a size.

To get good classification error you need to play with number of hidden
units and MAX_EPOCHS_UNSUPERVISED and MAX_EPOCHS_SUPERVISED. The defaults
are set to make the code run fast so you can see how all the interfaces
work without waiting for a long time. To get good accuracy, you'll need
to edit the code to make the models bigger and train longer.
