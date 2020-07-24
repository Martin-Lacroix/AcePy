import sys
sys.path.append('../../../')
from sampler import random
from scipy import special
import chaoslib as cl
import numpy as np
import pickle

# %% Functions

def factor(dim,k):
    
    kLen = np.size(k)
    C = special.comb(dim-1,kLen)
    return 1/(dim*C)

def comb(dim):
    
    kList = []
    A = cl.indextens(1,dim,np.inf)
    
    for a in A.T:
        
        k = a-1
        k[np.where(k!=-1)] = np.where(k!=-1)
        k = np.delete(k,np.where(k==-1))
        kList.append(k)
    
    return kList

def select(kList,index):
    
    kSelect = []
    
    for k in kList:
        if not np.isin(index,k): kSelect.append(k)
        
    return kSelect

def E_k(nbrPts,k=[],kVal=[]):
    
    point = random(nbrPts,k,kVal)
    return np.sum(model(point)/nbrPts)

def E2_k(nbrPts,k=[],kVal=[]):
    
    point = random(nbrPts,k,kVal)
    return np.sum(model(point)**2/nbrPts)

def V_k(nbrPts,k=[]):
    
    point = random(nbrPts)
    point = point[:,k]
        
    E2 = np.array([E2_k(nbrPts,k,kVal) for kVal in point])
    E = np.array([E_k(nbrPts,k,kVal) for kVal in point])
    EE2 = np.sum(E2)/nbrPts
    EE = np.sum(E)/nbrPts
    return EE2-EE**2

# %% Code
    

f = open('model.pickle','rb')
mod = pickle.load(f)
model = lambda point: mod.eval(point)[:,0,-1]
f.close()

x = 0
S = 0
nbrPts = 10
kList = comb(8)
kList = select(kList,x)

for k in kList:
    
    kU = np.append(k,x)
    S += factor(8,k)*(V_k(nbrPts,kU)-V_k(nbrPts,k))

