# ModernDCF

![PyPI version](https://img.shields.io/pypi/v/modern_dcf.svg)
[![Documentation Status](https://readthedocs.org/projects/modern_dcf/badge/?version=latest)](https://modern_dcf.readthedocs.io/en/latest/?version=latest)

This is a rewrite of the Python package pydcf by Damien Robertson with some speed-up to the computation and implimentation of the Gaussian kernel dcf which previously only was used to select a filter size for the sloted dcf function. It is intended to be used within scripts and in Jupyter notebooks, and is not a command line script. This is compatible with Python 3.12. Running the test case over 50k iterations with a tau range of (-200, 200), delta tau of 1.5, and linear detrending is found to be ~45% faster (1.74356 s with the old implimentation against 0.964 s with this implimentation). The new implimentation only uses linear detrending.

* PyPI package: https://pypi.org/project/modern_dcf/
* Free software: MIT License
* Documentation: https://modern_dcf.readthedocs.io.

## Features

* TODO

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
