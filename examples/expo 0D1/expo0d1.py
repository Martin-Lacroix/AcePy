import numpy as np

# %% Functions

def fun(x):

    expo = np.exp(-abs(x))
    return expo

def sampler(nbrPts):
    return np.random.normal(1,0.5,nbrPts)

def response(point):

    nbrPts = point.shape[0]
    resp = np.array([fun(point[i]) for i in range(nbrPts)]).flatten()
    return resp