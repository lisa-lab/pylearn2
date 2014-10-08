import struct
from array import array

def readmnist():

	path = '/home/carlo/workspace/datasets/mnist/'
	
	# read files
	with open(path + 'train-images-idx3-ubyte', "rb") as f:
		f.seek(16)
		train_imgs = array("B", f.read())
		
	with open(path + 't10k-images-idx3-ubyte', "rb") as f:
		f.seek(16)
		test_imgs = array("B", f.read())
		
	with open(path + 'train-labels-idx1-ubyte', "rb") as f:
		f.seek(8)
		train_lbls = array("B", f.read())
		
	with open(path + 't10k-labels-idx1-ubyte', "rb") as f:
		f.seek(8)
		test_lbls = array("B", f.read())
	# write files
	with open(path + 'mnist_train.csv', "wb") as f:
		for i in xrange(len(train_imgs) / 784):
		
			# print pixels values
			for j in xrange(783):
				f.write(str(train_imgs[i * 784 + j]) + ',')
			f.write(str(train_imgs[i * 784 + 783]) + '\n')
			
			# print labels in one-hot configuration
			for j in xrange(train_lbls[i]):
				f.write(str(0) + ',')
			f.write(str(1))
			if train_lbls[i] == 9:
				f.write('\n')
			else:
				f.write(',')
				for j in xrange(train_lbls[i] + 1, 9):
					f.write(str(0) + ',')
				f.write(str(0) + '\n')
				
	# write files
	with open(path + 'mnist_test.csv', "wb") as f:
		for i in xrange(len(test_imgs) / 784):
		
			# print pixels values
			for j in xrange(783):
				f.write(str(test_imgs[i * 784 + j]) + ',')
			f.write(str(test_imgs[i * 784 + 783]) + '\n')
			
			# print labels in one-hot configuration
			for j in xrange(test_lbls[i]):
				f.write(str(0) + ',')
			f.write(str(1))
			if test_lbls[i] == 9:
				f.write('\n')
			else:
				f.write(',')
				for j in xrange(test_lbls[i] + 1, 9):
					f.write(str(0) + ',')
				f.write(str(0) + '\n')
        
if __name__ == "__main__":
	readmnist()
