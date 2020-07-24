import sys
sys.path.append('../../../')
import chaoslib as cl
import numpy as np

# %% Functions

y = lambda x1,x2: 3+x1+x2+x2**2+x1*x2
random = lambda nbrPts: np.random.multivariate_normal(mean,cov,nbrPts)
dist = cl.Normal(0,1)

def random_x1(nbrPts,x1):
    
    mu = mean[1]+cov[1,0]/cov[0,0]*(x1-mean[0])
    var = cov[1,1]-cov[1,0]/cov[0,0]*cov[0,1]
    std = np.sqrt(var)
    
    return np.random.normal(mu,std,nbrPts)

def random_x2(nbrPts,x2):
    
    mu = mean[0]+cov[0,1]/cov[1,1]*(x2-mean[1])
    var = cov[0,0]-cov[0,1]/cov[1,1]*cov[1,0]
    std = np.sqrt(var)
    
    return np.random.normal(mu,std,nbrPts)

def Ey_x2(x2):
    
    point = random_x2(nbrPts,x2)
    return np.sum(y(point,x2)/nbrPts)

def Ey_x1(x1):
    
    point = random_x1(nbrPts,x1)
    return np.sum(y(x1,point)/nbrPts)

# %% Test Statistical Moments

nbrPts = int(1e5)
mean = np.array([0,0])
cov = np.array([[1,0.8],[0.8,1]])

point = random(nbrPts)
Ey = np.sum(y(*point.T)/nbrPts)

point = random(nbrPts)
Vy = np.sum((y(*point.T)-Ey)**2/nbrPts)

print('1')
point = dist.random(nbrPts)
Ey_x1_vec = np.array([Ey_x1(x1) for x1 in point])
EEy_x1 = np.sum(Ey_x1_vec)/nbrPts

print('2')
point = dist.random(nbrPts)
Ey_x2_vec = np.array([Ey_x2(x2) for x2 in point])
EEy_x2 = np.sum(Ey_x2_vec)/nbrPts

E2Ey_x1 = np.sum(Ey_x1_vec**2)/nbrPts
E2Ey_x2 = np.sum(Ey_x2_vec**2)/nbrPts

VEy_x2 = E2Ey_x2-EEy_x2**2
VEy_x1 = E2Ey_x1-EEy_x1**2

S1 = 1/(2*Vy)*(Vy-VEy_x2+VEy_x1)
S2 = 1/(2*Vy)*(Vy-VEy_x1+VEy_x2)
St = S1+S2