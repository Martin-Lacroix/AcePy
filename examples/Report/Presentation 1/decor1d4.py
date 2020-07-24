from scipy import special
import numpy as np

# %% Functions

def fun(T,A,E,n,F):

    R = 8.314
    tau = 6.1/60

    RT = R*T
    T0 = T[0]
    RT0 = R*T0
    E1 = lambda x: -special.expi(-x)

    C0 = -A*T*np.exp(-E/RT0)+A*E/R*E1(E/RT0)
    beta = (1-(1-n)/tau*(A*T*np.exp(-E/RT)-E1(E/RT)*E*A/R+C0))**(1/(1-n))
    gasProd = F*beta**n*(A/tau)*np.exp(-E/RT)

    return gasProd

def response(point):

    nbrPts = point.shape[0]
    T = np.linspace(300,1400,101)
    point = denorm(point)
    resp = np.array([fun(T,*point[i]) for i in range(nbrPts)])

    return resp

def norm(point):

    R = 8.314
    out = np.zeros(point.shape)
    P = np.array([1,113000,2,0.04])
    T = 800

    out[:,0] = np.log(point[:,0])-point[:,1]/(R*T)
    out[:,1] = point[:,1]/P[1]
    out[:,2] = point[:,2]/P[2]
    out[:,3] = point[:,3]/P[3]

    return out

def denorm(point):

    R = 8.314
    out = np.zeros(point.shape)
    P = np.array([1,113000,2,0.04])
    T = 800

    out[:,0] = np.exp(point[:,0]+point[:,1]*P[1]/(R*T))
    out[:,1] = point[:,1]*P[1]
    out[:,2] = point[:,2]*P[2]
    out[:,3] = point[:,3]*P[3]
    
    return out