FROM python:3.7
MAINTAINER Drew Blasius <drew.blasius@gmail.com>
RUN apt-get update -y && apt-get install g++ && pip install virtualenv tox pytest pytest-cov
