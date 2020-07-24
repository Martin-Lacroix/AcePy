import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Functions

def resc(point):
    
    T1 = 2000
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

def Esp1(A,E):
    
    point = joint.sobol(nbrPts)
    point[:,1] = A
    point[:,2] = E
    
    resp = model.eval(point)
    mean = np.mean(resp,axis=0)
    return mean

def Esp2(A,E):
    
    point = joint.sobol(nbrPts)
    point[:,5] = A
    point[:,6] = E
    
    resp = model.eval(point)
    mean = np.mean(resp,axis=0)
    return mean

# %% Whitening

nbrPts = int(1e6)
with np.load("pointALL.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)

point = resc(point[index])
mapping = cl.Pca(point)
point = mapping.white(point)
cov = np.cov(point,rowvar=0)

# %% Marginal Distributions

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

pdf = []
mean = np.mean(point,axis=0)
std = np.std(point,axis=0)
for i in range(8): pdf.append(cl.Normal(mean[i],std[i]).pdf)
joint = cl.Joint([cl.Normal(mean[i],std[i]) for i in range(8)])
x = np.linspace(-10,10,int(1e4))

plt.figure(1)
plt.plot(x,pdf[0](x))
plt.hist(point[:,0],color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_F [-]$")
plt.xlabel("$F_1$ [-]")
plt.xlim([mean[0]-3.5*std[0],mean[0]+3.5*std[0]])

plt.figure(2)
plt.plot(x,pdf[1](x))
plt.hist(point[:,1],color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_A [-]$")
plt.xlabel("$A_1$ [-]")
plt.xlim([mean[1]-3.5*std[1],mean[1]+3.5*std[1]])

plt.figure(3)
plt.plot(x,pdf[2](x))
plt.hist(point[:,2],color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_E [-]$")
plt.xlabel("$E_1$ [-]")
plt.xlim([mean[2]-3.5*std[2],mean[2]+3.5*std[2]])

plt.figure(4)
plt.plot(x,pdf[3](x))
plt.hist(point[:,3],color="w",bins=40,ec="lightgray",density=True)
plt.axes().set_aspect('auto')
plt.ylabel("$f_N [-]$")
plt.xlabel("$n_1$ [-]")
plt.xlim([mean[3]-3.5*std[3],mean[3]+3.5*std[3]])

plt.figure(5)
plt.plot(x,pdf[4](x))
plt.hist(point[:,4],color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_F [-]$")
plt.xlabel("$F_2$ [-]")
plt.xlim([mean[4]-3.5*std[4],mean[4]+3.5*std[4]])

plt.figure(6)
plt.plot(x,pdf[5](x))
plt.hist(point[:,5],color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_A [-]$")
plt.xlabel("$A_2$ [-]")
plt.xlim([mean[5]-3.5*std[5],mean[5]+3.5*std[5]])

plt.figure(7)
plt.plot(x,pdf[6](x))
plt.hist(point[:,6],color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_E [-]$")
plt.xlabel("$E_2$ [-]")
plt.xlim([mean[6]-3.5*std[6],mean[6]+3.5*std[6]])

plt.figure(8)
plt.plot(x,pdf[7](x))
plt.hist(point[:,7],color="w",bins=40,ec="lightgray",density=True)
plt.axes().set_aspect('auto')
plt.ylabel("$f_N [-]$")
plt.xlabel("$n_2$ [-]")
plt.xlim([mean[7]-3.5*std[7],mean[7]+3.5*std[7]])

# %% Polynomial Chaos Pato

nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = mapping.white(resc(point[index]))
resp = resp[index]

order = 4
poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

# %% Polynomial Chaos Expectation

group = 2
nbrPts = int(1e4)
with np.load("pointALL.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = mapping.white(resc(point[index]))

if group == 1: AE = point[:,1:3]
if group== 2: AE = point[:,5:7]

order = 9
poly = cl.gschmidt(order,AE)

nbrFek = poly[:].shape[0]
index,weight = cl.fekquad(AE,poly)
resp = np.zeros((nbrFek,11,121))
print(nbrFek)

poly.trunc(3)
AE = AE[index]
for i in range(nbrFek):
    if group == 1: resp[i] = Esp1(AE[i,0],AE[i,1])
    if group == 2: resp[i] = Esp2(AE[i,0],AE[i,1])
    print(i)

coef = cl.colloc(resp,poly,AE,weight)
modEspAE = cl.Expansion(coef,poly)

# %% Sensitivity Index PCE

varY = model.var
varEspAE = modEspAE.var
index = np.argmax(varY,axis=1)

varYmax = np.zeros(11)
varEspAEmax = np.zeros(11)

for i in range(11):
    varEspAEmax[i] = varEspAE[i,index[i]]
    varYmax[i] = varY[i,index[i]]
    
S = varEspAEmax/varYmax
S[3] = np.nan

# %% Sensitivity Index Monte Carlo

# group = 1
# nbrPts = int(1e3)
# with np.load("pointALL.npz",mmap_mode="r") as file: point = file["pts"]
# index = np.random.choice(point.shape[0],nbrPts,replace=0)
# point = mapping.white(resc(point[index]))

# if group == 1: AE = point[:,1:3]
# if group == 2: AE = point[:,5:7]
# EspAE = np.zeros((nbrPts,11,121))

# for i in range(nbrFek):
#     if group == 1: resp[i] = Esp1(AE[i,0],AE[i,1])
#     if group == 2: resp[i] = Esp2(AE[i,0],AE[i,1])
#     print(nbrPts-i)

# varEspAE = np.var(EspAE,axis=0)
# varY = model.var

# varYmax = np.zeros(11)
# varEspAEmax = np.zeros(11)
# index = np.argmax(varY,axis=1)

# for i in range(11):
#     varEspAEmax[i] = varEspAE[i,index[i]]
#     varYmax[i] = varY[i,index[i]]

# S = varEspAEmax/varYmax
# S[3] = np.nan