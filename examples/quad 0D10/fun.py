import numpy as np

# %% Functions

def fun(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):

    y = (x1*x2)**2-(x3*x4)**2+(x5*x6)**2-(x7*x8)**2+(x9*x10)**2
    return y

def sampler(nbrPts):

    mean = np.array([1,0,1,0,1])
    cor = (np.ones((5,5))-np.eye(5))*0.02
    var = np.eye(5)*0.2**2
    cov = cor+var
    
    norm = np.random.multivariate_normal(mean,cov,nbrPts)

    point = np.zeros((nbrPts,10))
    point[:,0] = norm[:,0]
    point[:,1] = np.random.uniform(0,2,nbrPts)
    point[:,2] = norm[:,1]
    point[:,3] = np.random.uniform(0,2,nbrPts)
    point[:,4] = norm[:,2]
    point[:,5] = np.random.uniform(0,2,nbrPts)
    point[:,6] = norm[:,3]
    point[:,7] = np.random.uniform(0,2,nbrPts)
    point[:,8] = norm[:,4]
    point[:,9] = np.random.uniform(0,2,nbrPts)
    return point

def response(point):

    nbrPts = point.shape[0]
    resp = np.array([fun(*point[i]) for i in range(nbrPts)])
    return resp