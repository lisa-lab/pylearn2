"""
Tests for pylearn2.utils.mem functions and classes.
"""


from pylearn2.utils.mem import (
    TypicalMemoryError,
    improve_memory_error_message
)


def test_typical_memory_error():
    """
    A dummy test that instantiates a TypicalMemoryError
    to see if there is no bugs.
    """
    try:
        raise TypicalMemoryError("test")
    except TypicalMemoryError as e:
        pass


def test_improve_memory_error_message():
    """
    Tests that the MemoryError's message is improved correctly
    """

    try:
        improve_memory_error_message(MemoryError(), "test")
    except MemoryError as e:
        # message has been "improved"
        assert len(str(e))

    try:
        improve_memory_error_message(MemoryError("test"), "should not")
    except MemoryError as e:
        assert str(e) == "test"
