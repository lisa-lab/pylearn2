def MeanSquaredReconstructionError(object):
    def __call__(self, model, X):
        return ((model.encode(X) - X) ** 2).sum(axis=1).mean()
