import numpy as np

# %% Functions

def fun(x,O,h):

    if (h<5e-4): h = 5e-4
    if (O>-1): O = -1
    if (O<-50): O = -50

    a = 0.95
    b = 0.95
    L = 70
    k = 2.37
    Tamb = 21.29

    gam = np.sqrt((2*(a+b)*h)/(a*b*k))
    C1 = -O/(k*gam)*(np.exp(gam*L)*(h+k*gam))/(np.exp(-gam*L)*(h-k*gam)+np.exp(gam*L)*(h+k*gam))
    C2 = O/(k*gam)+C1
    Ts = C1*np.exp(-gam*x)+C2*np.exp(gam*x)+Tamb

    return [Ts,Ts*1.1,Ts*1.2]

def sampler(nbrPts):

    point = np.zeros((nbrPts,2))
    point[:,0] = np.random.normal(-18,2,nbrPts)
    point[:,1] = np.random.gamma(2,0.001,nbrPts)

    return point

def response(point):

    resp = []
    nbrPts = point.shape[0]
    x = np.linspace(0,70,101)
    resp = np.array([fun(x,*point[i]) for i in range(nbrPts)])

    return resp