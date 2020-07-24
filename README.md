# Chaoslib

Code developped for obtaining the **Master's degree in Engineering physics**. A part of the thesis consists in the implementation of a polynomial chaos library for high dimensionnal stochastic problems with correlated input variables. The example and doc folders contain some test-cases as well as a compilable documentation.

## Use

First, make sure to work with Python 3 and the last version of Numpy and Scipy. This library also requires the package sobol_seq to generate the Sobol low-discrepancy sequence.
Then, place the chaoslib folder in the site-packages folder of Python.
```css
C:\ProgramData\WPy64-3770\python-3.7.7.amd64\Lib\site-packages
```
Another possibility is to add Chaoslib to your environment variables, this can be done without changing the settings of your computer by temporary adding the chaoslib folder to your path at the begining of your scripts using the following code, adapted to your own system.
```css
import sys
sys.path.append('../../')
import chaoslib
```

## Author

* Martin Lacroix
