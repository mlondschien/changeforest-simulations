from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

setup(
    name="changeforest_simulations",
    use_scm_version=True,
    description="Simulation studies for changeforest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlondschien/changeforest-simulations",
    author="Malte Londschien",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    install_requires=[],
)
