.PHONY: build test

install:
	pip install .

uninstall:
	pip uninstall pymc3-sklearn

dist:
	python setup.py sdist

build-container: 
	docker build -t drewblasius/pymc3-sklearn .

run-test-container:
	docker run --rm -v ${PWD}:/home/project --workdir /home/project \
		drewblasius/pymc3-sklearn make test

test-container: build-container run-test-container

test: install
	tox

clean:
	rm -rf .tox/
