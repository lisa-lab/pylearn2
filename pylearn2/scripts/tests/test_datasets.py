"""
Tests for all datasets scripts
"""

import os
import shutil
import tempfile

from pylearn2.scripts.datasets import (make_cifar100_gcn_whitened,
                                       make_cifar100_patches,
                                       make_cifar100_patches_8x8,
                                       make_cifar100_whitened,
                                       make_cifar10_gcn_whitened,
                                       make_cifar10_whitened,
                                       make_downsampled_stl10,
                                       make_stl10_patches,
                                       make_stl10_patches_8x8,
                                       make_stl10_whitened)

from pylearn2.testing.datasets import ArangeDataset


def dataset_output(dataset):
    try:
        data_dir = tempfile.mkdtemp()
        output_dir = dataset.create_output_dir(data_dir)

        assert os.path.exists(output_dir)
        assert os.path.exists(output_dir + '/README')

        fake_data = ArangeDataset(2)
        dataset.save_dataset(output_dir, fake_data, 'fake')

        assert os.path.exists(output_dir + '/fake.npy')
        assert os.path.exists(output_dir + '/fake.pkl')

    finally:
        shutil.rmtree(data_dir)


def test_cifar100_gcn_whitened_output():
    dataset_output(make_cifar100_gcn_whitened)


def test_cifar100_patches_output():
    dataset_output(make_cifar100_patches)


def test_cifar100_patches_8x8_output():
    dataset_output(make_cifar100_patches_8x8)


def test_cifar100_whitened_output():
    dataset_output(make_cifar100_whitened)


def test_cifar10_gcn_whitened_output():
    dataset_output(make_cifar10_gcn_whitened)


def test_cifar10_whitened_output():
    dataset_output(make_cifar10_whitened)


def test_stl10_downsampled_output():
    dataset_output(make_downsampled_stl10)


def test_stl10_patches_output():
    dataset_output(make_stl10_patches)


def test_stl10_patches_8x8_output():
    dataset_output(make_stl10_patches_8x8)


def test_stl10_whitened_output():
    dataset_output(make_stl10_whitened)
