import sys
sys.path.append('../../../')
import chaoslib as cl
import numpy as np

# %% Functions

def random(nbrPts,k=[],kVal=[]):
    
    idx = []
    point = []
    kVal = np.array(kVal)
    k = np.array(k)
    
    coStd = 0
    mean1 = np.array([1.2e4,7.1e4])
    cov1 = np.power([[1e3,coStd],[coStd,4e3]],2)
    
    coStd = 0
    mean2 = np.array([5e8,1.7e5])
    cov2 = np.power([[2e7,coStd],[coStd,6e3]],2)
    
    # Uniform sample
    
    if 0 not in k:
        point.append(cl.Uniform(0.1,0.4).random(nbrPts))
        idx.append(0)
        
    else:
        x0 = kVal[np.where(k==0)[0]]
        point.append(np.ones(nbrPts)*x0)
        idx.append(0)
        
        
    if 3 not in k:
        point.append(cl.Normal(3,0.2).random(nbrPts))
        idx.append(3)
        
    else:
        x3 = kVal[np.where(k==3)[0]]
        point.append(np.ones(nbrPts)*x3)
        idx.append(3)
        
    if 4 not in k:
        point.append(cl.Uniform(0.1,0.3).random(nbrPts))
        idx.append(4)
        
    else:
        x4 = kVal[np.where(k==4)[0]]
        point.append(np.ones(nbrPts)*x4)
        idx.append(4)
        
    if 7 not in k:
        point.append(cl.Normal(3,0.2).random(nbrPts))
        idx.append(7)
        
    else:
        x7 = kVal[np.where(k==7)[0]]
        point.append(np.ones(nbrPts)*x7)
        idx.append(7)
        
    # Gaussian x1 and x2
    
    if (1 in k) and (2 in k):
        
        x1 = kVal[np.where(k==1)[0]]
        x2 = kVal[np.where(k==2)[0]]
        point.append(np.ones(nbrPts)*x1)
        point.append(np.ones(nbrPts)*x2)
        idx.append(1)
        idx.append(2)
    
    if (not 1 in k) and (not 2 in k):
        
        pts = np.random.multivariate_normal(mean1,cov1,nbrPts)
        point.append(pts[:,0])
        point.append(pts[:,1])
        idx.append(1)
        idx.append(2)
        
    if (not 1 in k) and (2 in k):
        
        x2 = kVal[np.where(k==2)[0]]
        mu = mean1[0]+cov1[0,1]/cov1[1,1]*(x2-mean1[1])
        var = cov1[0,0]-cov1[0,1]/cov1[1,1]*cov1[1,0]
        std = np.sqrt(var)
        
        point.append(np.random.normal(mu,std,nbrPts))
        point.append(np.ones(nbrPts)*x2)
        idx.append(1)
        idx.append(2)
        
    if (not 2 in k) and (1 in k):
        
        x1 = kVal[np.where(k==1)[0]]
        mu = mean1[1]+cov1[1,0]/cov1[0,0]*(x1-mean1[0])
        var = cov1[1,1]-cov1[1,0]/cov1[0,0]*cov1[0,1]
        std = np.sqrt(var)
        
        point.append(np.random.normal(mu,std,nbrPts))
        point.append(np.ones(nbrPts)*x1)
        idx.append(2)
        idx.append(1)
        
    # Gaussian x5 and x6
    
    if (5 in k) and (6 in k):
        
        x5 = kVal[np.where(k==5)[0]]
        x6 = kVal[np.where(k==6)[0]]
        point.append(np.ones(nbrPts)*x5)
        point.append(np.ones(nbrPts)*x6)
        idx.append(5)
        idx.append(6)
        
    if (not 5 in k) and (not 6 in k):
        
        pts = np.random.multivariate_normal(mean2,cov2,nbrPts)
        point.append(pts[:,0])
        point.append(pts[:,1])
        idx.append(5)
        idx.append(6)
        
    if (not 5 in k) and (6 in k):
        
        x6 = kVal[np.where(k==6)[0]]
        mu = mean2[0]+cov2[0,1]/cov2[1,1]*(x6-mean2[1])
        var = cov2[0,0]-cov2[0,1]/cov2[1,1]*cov2[1,0]
        std = np.sqrt(var)
        
        point.append(np.random.normal(mu,std,nbrPts))
        point.append(np.ones(nbrPts)*x6)
        idx.append(5)
        idx.append(6)
        
    if (not 6 in k) and (5 in k):
        
        x5 = kVal[np.where(k==5)[0]]
        mu = mean2[1]+cov2[1,0]/cov2[0,0]*(x5-mean2[0])
        var = cov2[1,1]-cov2[1,0]/cov2[0,0]*cov2[0,1]
        std = np.sqrt(var)
        
        point.append(np.random.normal(mu,std,nbrPts))
        point.append(np.ones(nbrPts)*x5)
        idx.append(6)
        idx.append(5)
        
    # Returns the sample
    
    k = np.array(idx)
    point = np.vstack(point).T
    idx = np.argsort(k)
    point = point[:,idx]
    k = k[idx]
    
    return point