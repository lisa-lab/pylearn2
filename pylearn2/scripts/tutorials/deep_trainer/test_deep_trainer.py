"""
A simple unit test of 'run_deep_trainer.py'
"""
from .run_deep_trainer import main


def test_deep_trainer():
    # pass args=[] so we can pass options to nosetests on the command line
    main(args=[])
