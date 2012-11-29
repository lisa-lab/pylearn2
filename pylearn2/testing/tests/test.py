from pylearn2.testing import no_debug_mode

from theano import config

@no_debug_mode
def assert_not_debug_mode():
    assert config.mode != "DEBUG_MODE"

def test_no_debug_mode():
    orig_mode = config.mode
    config.mode = "DEBUG_MODE"
    try:
        # make sure the decorator gets rid of DEBUG_MODE
        assert_not_debug_mode()
    finally:
        # make sure the decorator restores DEBUG_MODE when it's done
        assert config.mode == "DEBUG_MODE"
        config.mode = orig_mode

