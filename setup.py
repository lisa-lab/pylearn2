from setuptools import setup, find_packages

setup(
    name='pylearn2',
    version='0.1dev',
    packages=find_packages(),
    description='A machine learning library build on top of Theano.',
    license='BSD 3-clause license',
    long_description=open('README.rst').read(),
    install_requires=['numpy>=1.5', 'theano'],
    package_data={
        '': ['*.txt', '*.rst', '*.cu', '*.cuh',],
    },
)
