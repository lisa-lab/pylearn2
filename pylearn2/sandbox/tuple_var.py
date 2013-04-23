"""
The skeleton of a Theano Variable and Type for representing a tuple.
Not really ready to use yet-- see theano-dev e-mail "TupleType"
"""

__author__ = "Ian Goodfellow"

from theano.gof.graph import Variable
from theano.gof.type import Type
from theano.tensor.basic import hashtype

class TupleType(Type):

    def __init__(self, component_types):
        if not isinstance(component_types, tuple):
            raise TypeError("Expected component_types to be a tuple, got "
                    + str(type(component_types)))
        assert all(isinstance(component, Type) for component in component_types)
        self.__dict__.update(locals())
        del self.self

    def __eq__(self, other):
        return type(self) == type(other) and len(self.component_types) == \
                len(other.component_types) and all(component_type == \
                other_component_type for component_type, other_component_type in \
                zip(self.component_types, other.component_types))

    def __hash__(self):
        return hashtype(self) ^ hash(self.component_types)



class TupleVariable(Variable):

    def __init__(self, component_variables, name = None):

        raise NotImplementedError("This is not safe to use yet, since T.grad won't work right with it.")

        assert isinstance(component_variables, tuple)
        self.__dict__.update(locals())
        del self.self

        assert isinstance(component_variables, tuple)
        self.type = TupleType(tuple(var.type for var in component_variables))

    def __hash__(self):
        return hashtype(self) ^ hash(self.component_variables)

    def __eq__(self, other):
        return type(self) == type(other) and len(self.component_variables) == \
                len(other.component_variables) and all(component_variable == \
                other_component_variable for component_variable, other_component_variable in \
                zip(self.component_variables, other.component_variables))

    def __getitem__(self, index):
        return self.component_variables[index]

    def __len__(self):
        return len(self.component_variables)
