import numpy as np

# %% Functions

def fun(t,x):

    expo = np.exp(-t*abs(x))
    return [expo,0.2+expo*1.2]

def sampler(nbrPts):
    return np.random.normal(1,0.5,nbrPts)

def response(point):

    nbrPts = point.shape[0]
    t = np.linspace(0,10,101)
    resp = np.array([fun(t,point[i]) for i in range(nbrPts)])
    return resp