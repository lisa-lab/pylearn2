import warnings
warnings.warn("The pylearn2.supervised_cost module is deprecated."
        "Its name was confusing because it did not actually define"
        "SupervisedCost, which is and was defined in cost.py")

# preserve old import in case anyone was referring to SupervisedCost
# by this location
from pylearn2.costs.cost import SupervisedCost
# import the only class that was defined here, so old code can still
# import it
from pylearn2.costs.cost import CrossEntropy

