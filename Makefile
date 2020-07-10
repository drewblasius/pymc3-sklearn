.PHONY: build test

build: 
	docker build -t drewblasius/pymc3-sklearn .

run-test:
	docker run --rm -v ${PWD}:/home/repo drewblasius/pymc3-sklearn tox

test: build run-test
