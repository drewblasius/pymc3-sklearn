from setuptools import find_packages, setup

setup(
    author="Drew Blasius",
    author_email="drewblasius@gmail.com",
    name="pymc3-sklearn",
    version="0.1",
    packages=find_packages(include=["sk_pymc3", "sk_pymc3.*"]),
    license="MIT License",
    long_description=open("README.rst").read(),
)
