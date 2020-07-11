.PHONY: install
install:
	pip install .

.PHONY: uninstall
uninstall:
	pip uninstall pymc3-sklearn

.PHONY: build-container
build-container: 
	docker build -t drewblasius/pymc3-sklearn .

.PHONY: run-test-container
run-test-container:
	docker run --rm -v ${PWD}:/home/project --workdir /home/project \
		drewblasius/pymc3-sklearn make test

.PHONY: test-container
test-container: build-container run-test-container

.PHONY: test
test: install
	tox

.PHONY: clean
clean:
	rm -rf .tox/

.PHONY: release
release:
	sleep 1s

.PHONY: dist
dist:
	python setup.py sdist bdist_wheel	
