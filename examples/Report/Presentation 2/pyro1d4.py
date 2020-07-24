from scipy import special
import numpy as np

# %% Functions

def gas(T,A,E,n,F):

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
    resp = np.array([gas(T,*point[i]) for i in range(nbrPts)])

    return resp