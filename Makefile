

install : 
	cd pylearn2/utils && python setup.py build_ext --inplace
	cd pylearn2/models && python setup.py build_ext --inplace
	python setup.py install

docs : 
	$(MAKE) -C doc

