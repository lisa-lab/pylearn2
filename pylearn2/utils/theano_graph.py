"""Utility functions that manipulate Theano graphs."""

import theano.tensor as tensor

def is_pure_elemwise(graph, inputs):
    """
    Checks whether a graph is purely elementwise and containing only
    inputs from a given list.

    Parameters
    ----------
    graph : TensorVariable object
        Graph to perform checks against.
    inputs : list
        List of acceptable inputs to the graph.

    Returns
    -------
    elemwise_or_not : bool
        Returns `True` if

        a) everything in the graph is an Elemwise or a DimShuffle
           (DimShuffles are only acceptable to broadcast up constants)
           and
        b) all nodes without an owner appear in `inputs` or are
           constants.

        Returns `False` otherwise.
    """
    allowed_ops = tensor.basic.DimShuffle, tensor.basic.Elemwise
    owner = graph.owner
    op = graph.owner.op if graph.owner is not None else None
    # Ownerless stuff is fine if it's in inputs.
    if owner is None and graph in inputs:
        return True
    # Constants are okay.
    elif owner is None and isinstance(graph, tensor.basic.TensorConstant):
        return True
    # But if it's not a constant and has no owner, it's not.
    elif owner is None and graph not in inputs:
        return False
    # Anything but Elemwise and DimShuffle should be rejected.
    elif op is not None and not isinstance(op, allowed_ops):
        return False
    else:
        if isinstance(graph.owner.op, tensor.basic.DimShuffle):
            shuffled = graph.owner.inputs[0]
            if not isinstance(shuffled, tensor.basic.TensorConstant):
                return False
        for inp in graph.owner.inputs:
            if not is_pure_elemwise(inp, inputs):
                return False
        return True
