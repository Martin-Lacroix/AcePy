from scipy import optimize,linalg
from .tools import timer,start_end
from .proba import Joint
import numpy as np

# %% Tensor Product Quadrature with Recurrence Coefficients

@start_end
def tensquad(order,dist):

    if not isinstance(dist,Joint): dist = Joint(dist)

    nbrPts = order+1
    dim = dist[:].shape[0]
    J = np.zeros((2,nbrPts))
    points = np.zeros((nbrPts,dim))
    weights = np.zeros((nbrPts,dim))

    # Integration points and weights of each variable

    for i in range(dim):

        coef = dist[i].coef(nbrPts)
        J[1] = np.append(np.sqrt(coef[1][1:]),[0])
        J[0] = coef[0]

        val,vec = linalg.eig_banded(J,lower=1)
        weights[:,i] = vec[0,:]**2
        points[:,i] = val.real

    # Places the points and weights in the domain

    point = np.zeros((nbrPts**dim,dim))
    weight = np.zeros((nbrPts**dim,dim))

    for i in range(dim):

        v1 = np.repeat(points[:,i],nbrPts**(dim-1-i))
        v2 = np.repeat(weights[:,i],nbrPts**(dim-1-i))
        weight[:,i] = np.tile(v2,nbrPts**i)
        point[:,i] = np.tile(v1,nbrPts**i)

    weight = np.prod(weight,axis=1)
    point = np.squeeze(point)
    return point,weight

# %% Approximate Fekete Points and Weights

@start_end
def fekquad(point,poly):

    nbrPoly = poly[:].shape[0]

    # Reconditioning of the Vandermonde matrix

    V = poly.eval(point)
    V,R = np.linalg.qr(V)
    m = np.sum(V,axis=0)/V.shape[0]

    # Computes the weights and Fekete points
    
    Q,R,P = linalg.qr(V.T,pivoting=1,mode='economic')
    R = R[:,:nbrPoly]
    q = np.dot(Q.T,m)

    weight = linalg.solve_triangular(R,q)
    index = P[:nbrPoly]

    return index,weight

# %% Positive Quadrature by Node Removal with QR downgrade

@start_end
def nulquad(point,poly,weight=0):

    V = poly.eval(point)
    nbrPts = V.shape[0]
    V,R = np.linalg.qr(V)
    end = nbrPts-V.shape[1]
    index = np.arange(nbrPts)
    if not np.any(weight): weight = np.ones(nbrPts)/nbrPts
    
    # Iteratively updates the quadrature rule

    for i in range(end):

        timer(i,end,'Computing nulquad ')
        if i==0: Q,R = linalg.qr(V,mode='full')
        else: Q,R = linalg.qr_delete(Q,R,idx,overwrite_qr=1,check_finite=0)
        z = Q[:,-1]

        # Selects the coefficient to cancel a weight

        wz = weight/z
        idx = np.argmin(np.abs(wz))
        alp = wz[idx]

        # Updates the weights and the matrix

        weight -= alp*z
        index = np.delete(index,idx)
        weight = np.delete(weight,idx)
    
    return index,weight

# %% Positive Quadrature by Node Removal with Newton-Raphson

@start_end
def newquad(point,poly,weight=0):
    
    def null(Vt,Jinv,z):
        
        for i in range(50):
            F = Vt.dot(z)
            z -= np.dot(Jinv,F)
            z = z/np.linalg.norm(z)
            if np.max(np.abs(F))<1e-16: break

        if i==49: return 0
        else: return z
        
    # First null space and initialization

    V = poly.eval(point)
    nbrPts = V.shape[0]
    V,R = np.linalg.qr(V)
    end = nbrPts-V.shape[1]
    index = np.arange(nbrPts)
    if not np.any(weight): weight = np.ones(nbrPts)/nbrPts
    z = np.ones(nbrPts)
    
     # Iteratively updates the quadrature rule

    for i in range(end):
        
        timer(i,end,'Computing newquad ')

        z = null(V.T,V,z)
        if not np.any(z): break

        # Selects the coefficient to cancel a weight

        wz = weight/z
        idx = np.argmin(np.abs(wz))
        alp = wz[idx]

        # Updates the weights and the matrices

        weight -= alp*z
        z = np.delete(z,idx)
        index = np.delete(index,idx)
        weight = np.delete(weight,idx)
        V,R = linalg.qr_delete(V,R,idx,overwrite_qr=1,check_finite=0)

    return index,weight

# %% Positive Quadrature with Revised Simplex

@start_end
def simquad(point,poly):
    
    # Initialization and Vandermonde matrix
    
    V = poly.eval(point)
    m = np.sum(V,axis=0)/V.shape[0]
    c = np.ones(V.shape[0])
    tol = 1e-20

    # Performs the revised simplex

    x = optimize.linprog(c,A_eq=V.T,b_eq=m,method='revised simplex')
    index = np.argwhere(x['x']>tol).flatten()
    weight = x['x'][index]

    if x['success']: return index,weight
    else: raise Exception('Simplex failure')
