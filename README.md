## Introduction

Code developed for obtaining the **Master's degree in Engineering Physics**. The code is an arbitrary polynomial chaos toolkit for high dimensional stochastic problems with correlated input variables. The examples and doc folders contain some test-cases as well as a documentation. [Thesis report here](https://hdl.handle.net/2268/293566).

## Installation

First, make sure to work with Python 3 and install the last version of Scipy. Some functionalities may not be available while using older packages. Then, add the main repository folder to your Python path environment variables. Another possibility is to add the path to AcePy in your Python script
```sh
export PYTHONPATH=path-to-acepy
```
```python
from sys import path
path.append('path-to-acepy')
import acepy
```

## Author

* Martin Lacroix
