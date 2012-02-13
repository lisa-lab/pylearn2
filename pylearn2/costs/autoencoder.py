from theano import tensor

class MeanSquaredReconstructionError(object):
    def __call__(self, model, X):
        return ((model.reconstruct(X) - X) ** 2).sum(axis=1).mean()


class MeanBinaryCrossEntropy(object):
    def __call__(self, model, X):
        return (
            tensor.xlogx.xlogx(model.reconstruct(X)) +
            tensor.xlogx.xlogx(1 - model.reconstruct(X))
        ).sum(axis=1).mean()


class MeanBinaryCrossEntropyTanh(object):
    def __call__(self, model, X):
        X = (X + 1) / 2.
        return (
            tensor.xlogx.xlogx(model.reconstruct(X)) +
            tensor.xlogx.xlogx(1 - model.reconstruct(X))
        ).sum(axis=1).mean()
