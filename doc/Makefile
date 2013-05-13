


all : apidoc doc


apidoc : epydoc_conf.cfg
	epydoc --debug --config $<

doc : scripts/docgen.py
	python $< 

