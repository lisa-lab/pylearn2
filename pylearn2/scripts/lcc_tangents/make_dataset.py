# replicate the preprocessing described in
# Kai Yu's paper Improving LCC with Local Tangents
from pylearn2.utils import serial
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing


def main():
    train = cifar10.CIFAR10(which_set="train", center=True)

    pipeline = preprocessing.Pipeline()
    pipeline.items.append(preprocessing.GlobalContrastNormalization(
        subtract_mean=False, sqrt_bias=0.0, use_std=True))
    pipeline.items.append(preprocessing.PCA(num_components=512))

    test = cifar10.CIFAR10(which_set="test")

    train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    test.apply_preprocessor(preprocessor=pipeline, can_fit=False)

    serial.save('cifar10_preprocessed_train.pkl', train)
    serial.save('cifar10_preprocessed_test.pkl', test)

if __name__ == "__main__":
    main()
