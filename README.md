# Chaoslib

Code developped for obtaining the **Master's degree in Engineering physics**. The code is a polynomial chaos library for high dimensionnal stochastic problems with correlated input variables. The examples and doc folders contain some test-cases as well as a compilable documentation.

## Use

First, make sure to work with Python 3 and the lastest version of Numpy and Scipy. This library also requires the package sobol_seq to generate the Sobol low-discrepancy sequence. Then, place the chaoslib folder in the site-packages folder of Python to use the library.
```css
C:\ProgramData\WPy64-3770\python-3.7.7.amd64\Lib\site-packages
```
Another possibility is to temporary add the package to your environment variables, this can be done by adding the chaoslib folder to the system path at the begining of your scripts using the following code.
```css
from sys import path
path.append('path-to-chaoslib')
import chaoslib
```

## Author

* Martin Lacroix
