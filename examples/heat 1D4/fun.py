import numpy as np

# %% Functions

def fun(x,O,h,k,Tamb):

    if (h<5e-4): h = 5e-4
    if (O>-1): O = -1
    if (O<-50): O = -50

    a = 0.95
    b = 0.95
    L = 70

    gam = np.sqrt((2*(a+b)*h)/(a*b*k))
    C1 = -O/(k*gam)*(np.exp(gam*L)*(h+k*gam))/(np.exp(-gam*L)*(h-k*gam)+np.exp(gam*L)*(h+k*gam))
    C2 = O/(k*gam)+C1
    Ts = C1*np.exp(-gam*x)+C2*np.exp(gam*x)+Tamb
    return Ts

def sampler(nbrPts):

    point = np.zeros((nbrPts,4))
    point[:,0] = np.random.normal(-18,2,nbrPts)
    point[:,1] = np.random.gamma(2,0.001,nbrPts)
    point[:,2] = np.random.uniform(1.5,3,nbrPts)
    point[:,3] = np.random.normal(21,1,nbrPts)
    return point

def response(point):

    nbrPts = point.shape[0]
    x = np.linspace(0,70,101)
    resp = np.array([fun(x,*point[i]) for i in range(nbrPts)])
    return resp