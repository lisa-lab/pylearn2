class MeanSquaredReconstructionError(object):
    def __call__(self, model, X):
        return ((model.reconstruct(X) - X) ** 2).sum(axis=1).mean()
