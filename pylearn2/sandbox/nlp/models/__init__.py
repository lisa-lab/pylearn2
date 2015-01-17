"""
Sandbox models for natural language processing (NLP)
"""


def as_tuple(list_):
    """
    Some parameters require tuples while YAML can only makes lists.
    Passing the list through this function solves the problem.

    Parameters
    ----------
    list_ : list
        A list to be converted.
    """

    return tuple(list_)
