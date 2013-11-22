"""
Tests for the sklearn_api
"""
__authors__ = "Alexandre Lacoste"
__copyright__ = "Copyright 2013, Universite Laval"
__credits__ = ["Alexandre Lacoste","Ian Goodfellow", "David Warde-Farley", "Pascal Lamblin" ]
__license__ = "3-clause BSD"
__maintainer__ = "Alexandre Lacoste"


from pylearn2.train import Train
from pylearn2.models import mlp, maxout
#from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import MonitorBased, EpochCounter
from pylearn2.training_algorithms import sgd 
from pylearn2.costs.mlp.dropout import Dropout as DropoutCost
from pylearn2.datasets import DenseDesignMatrix

import warnings
import numpy as np
import theano

print 'base_compiledir ', theano.config.base_compiledir

import logging

class ClassMap:
    """
    A tool for converting classification datasets having
    {-1,1} (or any other kind) as its set of categories.
    It also contains a reverse mapping for converting back to the
    original class set at prediction time. 
    """
    def __init__(self, y):
        classes = np.unique(y)
        self.min = np.min(classes)
        self._map = np.zeros( np.max(classes)-self.min+1, np.int ) 
        self._invmap = np.zeros(len(classes), np.int)
        for i, c in  enumerate(classes):
            self._map[c-self.min] = i
            self._invmap[i] = c
        
    def map(self, y ): 
        return self._map[y-self.min]
    
    def invmap(self, y ): 
        return self._invmap[y]


#def make_dataset_(x,y):
#    """
#    A utility for converting a pair x,y to 
#    a DenseDesignMatrix with one_hot activation function. 
#    """
#    
#    y = np.asarray(y, dtype=np.int)
#    # convert {-1,1} to {0,1} or other odd class sets
#    classmap = ClassMap(y)
#    y = classmap.map(y)
#    ds= DenseDesignMatrix(X=x, y=y )
#    ds.convert_to_one_hot()
#    return ds, classmap
#
#def make_dataset(x,y):
#    ds= DenseDesignMatrix(X=x, y=y )
#    ds.convert_to_one_hot()
#    return ds

class Classifier:
        
    def _make_dataset(self,x,y):
        y = np.asarray(y, dtype=np.int)
        if not hasattr( self, "classmap" ):
            self.classmap = ClassMap(y)
            
        ds = DenseDesignMatrix(X=x, y=self.classmap.map(y) )
        ds.convert_to_one_hot()
        return ds
    
    def fit(self, x, y=None ):
        ds = self._make_dataset(x, y)
        return self.train( ds )

    def train(self, dataset):
        self._build_model(dataset)
        
        x_variable = theano.tensor.matrix()
        y = self.model.fprop(x_variable)
        self.fprop = theano.function([x_variable], y)
        
        train = Train( dataset, self.model, self.algorithm )
        logging.getLogger("pylearn2").setLevel(logging.WARNING)
        train.main_loop()
        logging.getLogger("pylearn2").setLevel(logging.INFO)
    
        return self
    
    def _build_model(self, dataset):
        raise NotImplemented('Please, override this function.')
    
    def predict_proba(self, x):
        return self.fprop(x)
        
    def predict(self, x ):
        y_prob = self.predict_proba(x)
        idx = np.argmax(y_prob,1)
        y = self.classmap.invmap(idx)
        return y

    def set_valid_info(self,x,y):
        """
        Optional. But, if there is no valid_dataset, training will
        stop after a fixed number of iterations. 
        If you already have a valid pylearn2 dataset, you can directly 
        assign the valid_dataset attribute.
        """
        self.valid_dataset = self._make_dataset(x, y)
        
class MaxoutClassifier(Classifier):
    
    def __init__(self, 
        num_units = (100,100),
        num_pieces = 3,
        learning_rate = 0.1, 
        irange = 0.005,
        W_lr_scale = 1.,
        b_lr_scale = 1., 
        max_col_norm = 1.9365):
        
        self.__dict__.update( locals() )
        del self.self
    
    def _broadcast_param(self, param_name, layer ):
        """
        helper function to distinguish between fixed parameter
        or a different parameter for each layer
        """
        param = getattr(self, param_name, None )
        try:
            assert len(self.num_units) == len(param), '%s must have the same length as num_units or be a scalar.'%param_name
            return param[layer]
        except TypeError: # should be raised by len(param) if it is not a list
            return param
    
    def _build_model(self, dataset):
        # is there a more standard way to get this information ?
        n_features = dataset.X.shape[1] 
        n_classes = len(np.unique( dataset.y ) )
        
        layers = []
        for i, num_units in enumerate(self.num_units):
            
            layers.append( maxout.Maxout (
                layer_name= 'h%d'%i,
                num_units= num_units,
                num_pieces= self._broadcast_param('num_pieces', i),
                W_lr_scale= self._broadcast_param('W_lr_scale', i),
                b_lr_scale= self._broadcast_param('b_lr_scale', i),
                irange    = self._broadcast_param('irange', i),
                max_col_norm= self.max_col_norm,
            ))
            
#            print 'layer %d'%i
#            for key, val in layers[-1].__dict__.items():
#                print key, val
#            print
            
        layers.append(  mlp.Softmax (
            max_col_norm= self.max_col_norm,
            layer_name= 'y',
            n_classes= n_classes,
            irange= self.irange,
        ))
        
                
        self.model = mlp.MLP(
            batch_size = 100,
            layers = layers,
            nvis=n_features,
        )
        
        
        try:
            monitoring_dataset= {'valid' : self.valid_dataset}
            monitor = MonitorBased( channel_name= "valid_y_misclass", prop_decrease= 0., N= 100)
            print 'using the valid dataset'
        except AttributeError: 
            warnings.warn('No valid_dataset. Will optimize for 100 epochs')
            monitoring_dataset = None
            monitor = EpochCounter(1000)
        
        self.algorithm = sgd.SGD(
            learning_rate= self.learning_rate,
            init_momentum= .5,
            monitoring_dataset= monitoring_dataset,
            cost= DropoutCost(
                input_include_probs= { 'h0' : .8 },
                input_scales= { 'h0': 1. }
            ),
            termination_criterion= monitor,
            update_callbacks= sgd.ExponentialDecay(
                decay_factor= 1.00004,
                min_lr= .000001
            )
        )
        
        
#    
#class MaxoutConv(Classifier):
#    
#    def __init__(self, 
#        irange = 0.005,
#        w_lr_scale = 0.05,
#        b_lr_scale = 0.05,
#        ):
#        self.__dict__.update( locals() )
#    
#    
#    def _build_model(self, dataset):
#
#        layers= [
#            maxout.MaxoutConvC01B (
#                layer_name= 'h0',
#                pad= 0, # 4, 4
##                tied_b= 1,
#                W_lr_scale= self.w_lr_scale,
#                b_lr_scale= self.b_lr_scale,
#                num_channels= 48, # 64
#                num_pieces= 2,
#                kernel_shape= [8, 8],# [5,5]
#                pool_shape= [4, 4], # [3,3]
#                pool_stride= [2, 2],
#                irange= self.irange,
#                max_kernel_norm= .9,
#            ),
#            maxout.MaxoutConvC01B(
#                layer_name= 'h1',
#                pad= 3,
##                tied_b= 1,
#                W_lr_scale= self.w_lr_scale,
#                b_lr_scale= self.b_lr_scale,
#                num_channels= 48, # 128
#                num_pieces= 2,
#                kernel_shape= [8, 8], # [5,5]
#                pool_shape= [4, 4], # [3,3]
#                pool_stride= [2, 2],
#                irange= self.irange,
#                max_kernel_norm= 1.9365,
#            ),
#            maxout.MaxoutConvC01B(
#                pad= 3,
#                layer_name= 'h2',
##                tied_b= 1,
#                W_lr_scale= self.w_lr_scale,
#                b_lr_scale= self.b_lr_scale,
#                num_channels= 24, # 128
#                num_pieces= 4, # 2
#                kernel_shape= [5, 5],
#                pool_shape= [2, 2], # [3,3]
#                pool_stride= [2, 2],
#                irange= self.irange,
#                max_kernel_norm= 1.9365,
#            ),
##            pylearn2.models.maxout.Maxout(
##                layer_name= 'h3',
##                irange= self.irange,
##                num_units= 240, # 400
##                num_pieces= 5,
##                max_col_norm= 1.9
##            ),
#
#            mlp.Softmax(
#                max_col_norm= 1.9365,
#                layer_name= 'y',
#                n_classes= 10,
#                irange= .005
#            )
#        ]
#    
#        self.model = mlp.MLP(
#            batch_size= 128,
#            layers = layers,
#            input_space= Conv2DSpace(
#                shape= [28, 28],
#                num_channels= 1,
#                axes= ['c', 0, 1, 'b'],
#            ),
#        )
#        
#    
#
#        self.algorithm = sgd.SGD(
#            learning_rate= .05, # 0.1
#            init_momentum= .5,
#            monitoring_dataset= {'valid' : self.valid_dataset},
#            cost= DropoutCost(
#                input_include_probs= { 'h0' : .8 },
#                input_scales= { 'h0': 1. }
#            ),
#            termination_criterion= MonitorBased(
#                channel_name= "valid_y_misclass",
#                prop_decrease= 0.,
#                N= 100
#            ),
#            update_callbacks= sgd.ExponentialDecay(
#                decay_factor= 1.00004,
#                min_lr= .000001
#            )
#        )
#        
#        return self._algorithm


