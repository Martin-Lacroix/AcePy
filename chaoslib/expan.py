from .poly import polyrecur
from scipy import integrate
from .tools import printer
from copy import deepcopy
import numpy as np

# %% Polynomial Chaos

class Expansion:
    """Class of polynomial chaos expansion"""

    def __init__(self,coef,poly):

        coef = np.array(coef)
        self.var = np.sum(coef[1:]**2,axis=0)
        self.mean = coef[0]
        
        shape = (poly[:].shape[1],)+coef.shape[1:]
        coef = coef.reshape(poly[:].shape[0],-1)

        self.expo = np.copy(np.atleast_2d(poly.expo))
        self.coef = poly[:].T.dot(coef).reshape(shape).T
        self.dim = self.expo.shape[0]

    # Evaluates the expansion at the points

    def eval(self,point):

        V = 1
        point = np.reshape(np.transpose(point),(self.dim,-1)).T
        for i in range(self.dim): V *= np.power(point[:,i,None],self.expo[i])
        V = np.squeeze(np.dot(self.coef,V.T).T)
        return V

# %% Univariate Expansion

def transfo(invcdf,order,dist):
    """Maps an arbitrary random variable to another distribution"""

    nbrPoly = order+1
    coef = np.zeros(nbrPoly)
    poly = polyrecur(order,dist)

    # Computes polynomial chaos coefficients and model

    for i in range(nbrPoly):

        fun = lambda x: invcdf(x)*poly.eval(dist.invcdf(x))[:,i]
        coef[i] = integrate.quad(fun,0,1)[0]

    expan = Expansion(coef,poly)
    transfo = lambda x: expan.eval(x)
    return transfo

# %% Analysis of Variance

def anova(coef,poly):
    """Computes the first and total order Sobol sensitivity indices"""

    S,ST = [[],[]]
    expo = poly.expo
    dim = expo.shape[0]
    coef = np.array(coef)
    nbrPoly = poly[:].shape[0]
    var = np.sum(coef[1:]**2,axis=0)

    # Computes the first and total Sobol indices

    for i in range(dim):

        order = np.sum(expo,axis=0)
        pIdx = np.array([poly[j].nonzero()[-1][-1] for j in range(nbrPoly)])

        sIdx = np.where(expo[i]-order==0)[0].flatten()[1:]
        index = np.where(np.in1d(pIdx,sIdx))[0]
        S.append(np.sum(coef[index]**2,axis=0)/var)

        sIdx = np.where(expo[i])[0].flatten()
        index = np.where(np.in1d(pIdx,sIdx))[0]
        ST.append(np.sum(coef[index]**2,axis=0)/var)

    S = np.array(S)
    ST = np.array(ST)
    sobol = dict(zip(['S','ST'],[S,ST]))
    return sobol

# %% Analysis of Covariance
    
def ancova(model,point,weight=0):
    """Computes the sensitivity indices by analysis of covariance"""

    printer(0,'Computing ancova ...')
    nbrPts = np.array(point)[...].shape[0]
    if not np.any(weight): weight = np.ones(nbrPts)/nbrPts

    expo = model.expo
    coef = model.coef
    nbrIdx = expo.shape[1]
    index,ST,SS = [[],[],[]]

    model = deepcopy(model)
    resp = model.eval(point)
    difMod = resp-np.dot(resp.T,weight).T
    varMod = np.dot(difMod.T**2,weight).T
    
    # Computes the total and structural indices

    for i in range(1,nbrIdx):

        model.expo = expo[:,i,None]
        model.coef = coef[...,i,None]
        resp = model.eval(point)
        
        dif = resp-np.dot(resp.T,weight).T
        cov = np.dot((dif*difMod).T,weight).T
        var = np.dot(dif.T**2,weight).T
        S = cov/varMod
        
        if not np.allclose(S,0):

            index.append(expo[:,i])
            SS.append(var/varMod)
            ST.append(S)
            
    # Combines the different powers of a same monomial
    
    index,SS,ST = combine(index,SS,ST)
    ancova = dict(zip(['SS','SC','ST'],[SS,ST-SS,ST]))
    printer(1,'Computing ancova 100 %')
    return index,ancova

# %% Combine Power

def combine(index,SS,ST):
    """Combines the indices from different powers of the same monomial"""
    
    index = np.transpose(index)
    index = (index/np.max(index,axis=0)).T
    
    # Normalizes and eliminates the duplicates
    
    minIdx = np.min(index,axis=1)
    idx = np.argwhere(minIdx).flatten()
    index[idx] = (index[idx].T/minIdx[idx]).T
    index = np.rint(index).astype(int)
    
    index,old = np.unique(index,return_inverse=1,axis=0)
    shape = (index.shape[0],)+np.array(SS).shape[1:]
    SS2 = np.zeros(shape)
    ST2 = np.zeros(shape)
    
    # Combines duplicates ancova indices
    
    for i in range(old.shape[0]): SS2[old[i]] += SS[i]
    for i in range(old.shape[0]): ST2[old[i]] += ST[i]
    return index,SS2,ST2