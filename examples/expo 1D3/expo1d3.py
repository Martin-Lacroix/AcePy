import numpy as np

# %% Functions

def fun(t,x1,x2,x3):

    def c(x):
        if (x<0.5): return x1
        elif (0.5<=x<0.7): return x2
        else: return x3

    N = t.shape[0]
    u = np.zeros(N)
    u[0] = 0.3

    for i in range(N-1):

        dt = t[i+1]-t[i]
        k1 = -dt*u[i]*c(t[i])
        k2 = -dt*u[i]+k1/2*c(t[i]+dt/2)
        u[i+1] = u[i]+k1+k2

    return u

def sampler(nbrPts):

    point = np.zeros((nbrPts,3))
    point[:,0] = np.random.normal(0.5,0.15,nbrPts)
    point[:,1] = np.random.uniform(0.5,2.5,nbrPts)
    point[:,2] = np.random.normal(0.03,0.07,nbrPts)
    return point

def response(point):

    nbrPts = point.shape[0]
    t = np.linspace(0,1,101)
    resp = np.array([fun(t,*point[i]) for i in range(nbrPts)])
    return resp