import numpy as np
import acepy as ap
from fun import sampler

# %% Functions

def cholesky(B):
    
    n = B.shape[0]
    L = np.zeros((n,n))
    L[0,0] = np.sqrt(B[0,0])
    
    for j in range(1,n): L[j,0] = B[j,0]/L[0,0]
    
    for i in range(1,n):
        L[i,i] = np.sqrt(B[i,i]-np.sum([L[i,j]**2 for j in range(i)]))
        
        if (i!=n-1):
            for j in range(i+1,n): L[j,i] = (B[j,i]-np.sum([L[i,k]*L[j,k] for k in range(i)]))/L[i,i]
            
    return L

# %% Tests

order = 2
nbrPts = int(50)
point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
V = poly.eval(point)

B = np.dot(V.T,V)
L = cholesky(B)
R = L.T
Q = V.dot(np.linalg.inv(R))

I = np.allclose(np.eye(Q.shape[1]),Q.T.dot(Q))
V = np.allclose(V,Q.dot(R))