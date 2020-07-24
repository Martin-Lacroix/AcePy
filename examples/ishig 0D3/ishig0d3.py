import numpy as np

# %% Functions

def fun(x1,x2,x3):

    a = 7
    b = 0.1

    shig = np.sin(x1)+a*np.sin(x2)**2+b*x3**4*np.sin(x1)
    return shig

def sampler(nbrPts):

    point = np.zeros((nbrPts,3))
    point[:,0] = np.random.uniform(-np.pi,np.pi,nbrPts)
    point[:,1] = np.random.uniform(-np.pi,np.pi,nbrPts)
    point[:,2] = np.random.uniform(-np.pi,np.pi,nbrPts)
    return point

def response(point):

    nbrPts = point.shape[0]
    resp = np.array([fun(*point[i]) for i in range(nbrPts)])
    return resp
