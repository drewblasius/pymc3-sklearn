.PHONY: build test

install:
	python setup.py install

dist:
	python setup.py sdist

build-container: 
	docker build -t drewblasius/pymc3-sklearn .

run-test-container:
	docker run --rm -v ${PWD}:/home/project --workdir /home/project \
		drewblasius/pymc3-sklearn make test

test-container: build-container run-test-container

test:
	tox

clean:
	rm -rf .tox/
