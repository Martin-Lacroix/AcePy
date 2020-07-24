import numpy as np

# %% Functions

def fun(x1,x2):

    y = x1+x2+x2**2+x1*x2+3
    return y

def sampler(nbrPts):

    std = np.array([1,1])
    mean = np.array([0,0])
    p = np.zeros((2,2))

    p[0,1] = 0.8
    p = np.eye(2)+p+p.T
    cov = np.eye(2)*std
    cov = cov.dot(p).dot(cov)

    point = np.random.multivariate_normal(mean,cov,nbrPts)
    return point

def response(point):

    nbrPts = point.shape[0]
    resp = np.array([fun(*point[i]) for i in range(nbrPts)])
    return resp