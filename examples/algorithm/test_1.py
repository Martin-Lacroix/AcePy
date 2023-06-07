import numpy as np
import acepy as ap
from fun import sampler

# %% Functions

def givens(V,i,j,k):

    v2 = V[j,k]
    v1 = V[i,k]
    r = np.sqrt(v1**2+v2**2)
    sin = -v2/r
    cos = v1/r

    G = np.eye(V.shape[0])
    G[i,j] = -sin
    G[j,i] = sin
    G[j,j] = cos
    G[i,i] = cos
    
    return G

def qr(V):
    
    m,n = V.shape
    Q = np.eye(V.shape[0])
    R = V.copy().astype(float)
    
    for j in range(n):
        for i in range(m-1,j,-1):
            
            v2 = R[i,j]
            v1 = R[i-1,j]
            r = np.sqrt(v1**2+v2**2)
            sin = -v2/r
            cos = v1/r
        
            G = np.array([[cos,-sin],[sin,cos]])
            Q[i-1:i+1] = np.dot(G,Q[i-1:i+1])
            R[i-1:i+1] = np.dot(G,R[i-1:i+1])
            
    Q = np.transpose(Q)
    return Q,R

def sincos(v1,v2):
    
    if v2==0:
        cos = 1
        sin = 0
                
    elif abs(v2)>=abs(v1):
        r = -v1/v2
        sin = 1/np.sqrt(1+r**2)
        cos = sin*r
    else:
        r = -v2/v1
        cos = 1/np.sqrt(1+r**2)
        sin = cos*r
            
    return sin,cos

def rupdate(Q,R,index):
    
    m,n = R.shape
    R = R.copy().astype(float)
    q = Q[index].copy().astype(float)

    for i in range(m-2,-1,-1):
        
        sin,cos = sincos(q[i],q[i+1])
        q[i] = cos*q[i]-sin*q[i+1]
        
        if (i<=n-1):
            
            G = np.array([[cos,-sin],[sin,cos]])
            R[i:i+2,i:] = G.dot(R[i:i+2,i:])
            
    R = R[1:]
    return R

def qupdate(Q,index):
    
    m = Q.shape[0]
    q = Q[index].copy()
    Q = Q.copy().astype(float)
    Q[1:index+1] = Q[0:index]
        
    for i in range(m-2,-1,-1):
    
        sin,cos = sincos(q[i],q[i+1])
        q[i] = cos*q[i]-sin*q[i+1]
        
        GT = np.array([[cos,sin],[-sin,cos]])
        Q[1:,i:i+2] = Q[1:,i:i+2].dot(GT)
        
    Q = Q[1:,1:]
    return Q

# %% Test QR

order = 2
nbrPts = int(50)
point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
V = poly.eval(point)

# Build QR

Q,R = qr(V)
V1 = np.allclose(V,Q.dot(R))
I1 = np.allclose(np.eye(Q.shape[1]),Q.T.dot(Q))

# Update QR

idx = int(nbrPts/2)
V = np.delete(V,idx,axis=0)

R = rupdate(Q,R,idx)
Q = qupdate(Q,idx)
V2 = np.allclose(V,Q.dot(R))
I2 = np.allclose(np.eye(Q.shape[1]),Q.T.dot(Q))

# Null Space

z = Q[:,-1]
zero = np.dot(V.T,z)