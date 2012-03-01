This is just a simple example to give you a feeling of how to use pylearn2 by writing your own python scripts. Note
that an alternative way is to write yaml scripts (see scripts/train_example for this method).
 
It is by no means a full-blown demo and largely still is work in progress. 

usage:

make sure pylearn2 can be found in your PYTHONPATH

to train models on toy dataset: python run_deep_trainer.py -d toy

to train models on cifar10 dataset: python run_deep_trainer.py -d cifar10

to visualize learned the weights on cifar10 (currently only weights of the first layer is supported due to the 
some complication)

epoch 0: show_weights.py cifar10_grbm0_epoch.pkl
epoch 1: show_weights.py cifar10_grbm1_epoch.pkl
....
well, you get the idea.



