import numpy as np

# %% Functions

def fun(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14):

    c = (x3+x7+x6)/2
    i = (x12+x13+x14)/2
    h = x10+x11
    alp = np.arccos(c/np.sqrt(i**2+h**2))-np.arccos(i/np.sqrt(i**2+h**2))
    r2 = ((x13+x14/2)/2-(x7+x6/2)/(2*np.cos(alp)))/np.tan(alp)
    z = r2/np.cos(alp)+((x8+x9/2)/2+x6/4)*np.tan(alp)
    J1 = (x1-x2-z)*np.sin(alp)
    J2 = x6/4*np.cos(alp)
    J3 = (x4+x5)/2*np.cos(alp)
    y = J1+J2+J3

    return y


def sampler(nbrPts):

    p = np.zeros((14,14))
    p[[0,2,2,6,11,11,12],[1,6,7,7,12,13,13]] = 0.8
    std = [0.2,0.04,0.015,0.2,0.06,0.2,0.04,0.05,0.04,0.06,0.06,0.04,0.04,0.04]
    mean = [10.53,0.75,0.643,0.1,0,0,0.72,1.325,0,3.02,0.4,0.72,0.97,0]
    std = np.array(std)/6
    p = np.eye(14)+p+p.T
    p = 2*np.sin(np.pi*p/6)

    cov = np.eye(14)*std
    cov = cov.dot(p).dot(cov)

    point = np.random.multivariate_normal(mean,cov,nbrPts)
    return point

def response(point):

    nbrPts = point.shape[0]
    resp = np.array([fun(*point[i]) for i in range(nbrPts)])
    return resp