from pylearn2.costs.cost import Cost
from pylearn2.costs.autoencoder import (MeanSquaredReconstructionError,
                                        MeanBinaryCrossEntropy)

# FIXME: When do we backprop? Do we backprop at the end of execution (how it is
# written now, or do we backprop as soon as we each sample (would have to change
# into generator)

def _class_creator(klass):
    # klass should be a subclass of WalkbackFriendlyCost, but if it quacks like
    # a duck then that's good enough
    assert hasattr(klass, 'cost') and callable(klass.cost)

    class Inner(Cost):
        def expr(self, model, data, walkback=0):
            self.get_data_specs(self, model)[0].validate(data)
            X = data
            return sum(klass.cost(X, reconstructed)
                       for reconstructed in model.get_sample(X, walkback=walkback))

        def get_data_specs(self, model):
            return (model.get_input_space(), model.get_input_source())

    return Inner

MSWalkbackReconstructionError = _class_creator(MeanSquaredReconstructionError)
MBWalkbackCrossEntropy = _class_creator(MeanBinaryCrossEntropy)
