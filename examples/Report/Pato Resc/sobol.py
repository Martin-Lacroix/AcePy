import sys
sys.path.append('../../../')
import chaoslib as cl
import numpy as np

# %% Rescaling and Whitening

def resc(point):
    
    T1 = 1000
    T2 = 1000
    A1 = point[:,1]
    E1 = point[:,2]
    A2 = point[:,5]
    E2 = point[:,6]
    A1 = np.log(A1)-E1/(8.314*T1)
    A2 = np.log(A2)-E2/(8.314*T2)
    E1 = E1/np.mean(E1)
    E2 = E2/np.mean(E2)
    
    ptRes = point.copy()
    ptRes[:,1] = A1
    ptRes[:,2] = E1
    ptRes[:,5] = A2
    ptRes[:,6] = E2
    
    return ptRes

nbrPts = int(1e6)
with np.load("pointALL.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
mapping = cl.Pca(resc(point[index]))

point = mapping.white(resc(point[index]))
cov = np.cov(point,rowvar=0)

# %% Initialisation

nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = mapping.white(resc(point[index]))
resp = resp[index]

# %% Polynomial Chaos

order = 3
poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

# %% Sobol

def anova(coef,poly,i,k):

    S,ST = [[],[]]
    expo = poly.expo
    coef = np.array(coef)
    nbrPoly = poly[:].shape[0]
    var = np.sum(coef[1:]**2,axis=0)

    # Computes the first and total Sobol indices

    order = np.sum(expo,axis=0)
    pIdx = np.array([poly[j].nonzero()[-1][-1] for j in range(nbrPoly)])
    sIdx = np.where(expo[i]-order==0)[0].flatten()[1:]
    indexi = np.where(np.in1d(pIdx,sIdx))[0].tolist()
    pIdx = np.array([poly[j].nonzero()[-1][-1] for j in range(nbrPoly)])
    sIdx = np.where(expo[k]-order==0)[0].flatten()[1:]
    indexk = np.where(np.in1d(pIdx,sIdx))[0].tolist()
    
    index = np.unique(np.array(indexi+indexk))
    S.append(np.sum(coef[index]**2,axis=0)/var)

    sIdx = np.where(expo[i])[0].flatten()
    indexi = np.where(np.in1d(pIdx,sIdx))[0].tolist()
    sIdx = np.where(expo[k])[0].flatten()
    indexk = np.where(np.in1d(pIdx,sIdx))[0].tolist()
    
    index = np.unique(np.array(indexi+indexk))
    ST.append(np.sum(coef[index]**2,axis=0)/var)

    S = np.array(S)
    ST = np.array(ST)
    sobol = dict(zip(['S','ST'],[S,ST]))
    return sobol

coef = coef[:,:,-1]
SAE1 = anova(coef,poly,1,2)
SAE2 = anova(coef,poly,5,6)
S = cl.anova(coef,poly)