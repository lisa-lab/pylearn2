#replicate the preprocessing described in Kai Yu's paper Improving LCC with Local Tangents
from framework.utils import serial
from framework.datasets import cifar10
from framework.datasets import preprocessing

train = cifar10.CIFAR10(which_set="train")

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.GlobalContrastNormalization(subtract_mean=False,std_bias=0.0))
pipeline.items.append(preprocessing.PCA(num_components=512))

test = cifar10.CIFAR10(which_set="test")

train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)

serial.save('cifar10_preprocessed_train.pkl',train)
serial.save('cifar10_preprocessed_test.pkl',test)


