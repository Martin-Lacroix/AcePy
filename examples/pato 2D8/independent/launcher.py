import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np
import subprocess
import time
import os

# %% Original Parameters

dist = []
nbrPts = int(1e4)

dist.append(cl.Uniform(0.1,0.4))
dist.append(cl.Normal(1.2e4,1e3))
dist.append(cl.Normal(7.1e4,4e3))
dist.append(cl.Normal(3,0.2))

dist.append(cl.Uniform(0.1,0.3))
dist.append(cl.Normal(5e8,2e7))
dist.append(cl.Normal(1.7e5,6e3))
dist.append(cl.Normal(3,0.2))

dist = cl.Joint(dist)
point = np.abs(dist.random(nbrPts))

oldPath = '/home/orbbe/PATO/PATO_v2.3.0/data/Materials/Composites/TACOT/constantProperties'
massPath = '/home/orbbe/PATO/PATO_v2.3.0/tutorials/1D/Ablation_PCE/output/porousMat/mass'
massLossPath = '/home/orbbe/PATO/PATO_v2.3.0/tutorials/1D/Ablation_PCE/output/porousMat/massLoss'
TaPath = '/home/orbbe/PATO/PATO_v2.3.0/tutorials/1D/Ablation_PCE/output/porousMat/scalar/Ta_plot'
TaSurfPath = '/home/orbbe/PATO/PATO_v2.3.0/tutorials/1D/Ablation_PCE/output/porousMat/scalar/Ta_surfacePatch'
newPath = '/home/orbbe/PATO/PATO_v2.3.0/data/Materials/Composites/TACOT_PCE/constantProperties'
runPath = '/home/orbbe/PATO/PATO_v2.3.0/tutorials/1D/Ablation_PCE/Allrun'

oldF1 = 'F[2][1]         F[2][1]         [0 0 0 0 0 0 0]         0.25;'
oldA1 = 'A[2][1]         A[2][1]         [0 0 -1 0 0 0 0]        12000;'
oldE1 = 'E[2][1]         E[2][1]         [1 2 -2 0 -1 0 0]       71130.89;'
oldm1 = 'm[2][1]         m[2][1]         [0 0 0 0 0 0 0]         3;'

oldF2 = 'F[2][2]         F[2][2]         [0 0 0 0 0 0 0]         0.19;'
oldA2 = 'A[2][2]         A[2][2]         [0 0 -1 0 0 0 0]        4.97777e8;'
oldE2 = 'E[2][2]         E[2][2]         [1 2 -2 0 -1 0 0]       1.69975e5;'
oldm2 = 'm[2][2]         m[2][2]         [0 0 0 0 0 0 0]         3;'

f = open(oldPath,'r')
oldText = f.read()
f.close()

# %% New Parameters

F1 = point[:,0]
A1 = point[:,1]
E1 = point[:,2]
m1 = point[:,3]

F2 = point[:,4]
A2 = point[:,5]
E2 = point[:,6]
m2 = point[:,7]

newF1 = []
newA1 = []
newE1 = []
newm1 = []

newF2 = []
newA2 = []
newE2 = []
newm2 = []

for i in range(nbrPts):
 
    newF1.append('F[2][1]         F[2][1]         [0 0 0 0 0 0 0]         '+str(F1[i])+";")
    newA1.append('A[2][1]         A[2][1]         [0 0 -1 0 0 0 0]        '+str(A1[i])+";")
    newE1.append('E[2][1]         E[2][1]         [1 2 -2 0 -1 0 0]       '+str(E1[i])+";")
    newm1.append('m[2][1]         m[2][1]         [0 0 0 0 0 0 0]         '+str(m1[i])+";")
    
    newF2.append('F[2][2]         F[2][2]         [0 0 0 0 0 0 0]         '+str(F2[i])+";")
    newA2.append('A[2][2]         A[2][2]         [0 0 -1 0 0 0 0]        '+str(A2[i])+";")
    newE2.append('E[2][2]         E[2][2]         [1 2 -2 0 -1 0 0]       '+str(E2[i])+";")
    newm2.append('m[2][2]         m[2][2]         [0 0 0 0 0 0 0]         '+str(m2[i])+";")
    
# %% Algorithm

resp = []

for i in range(nbrPts):
    
    # Update the parameters

    start = time.time()
    f = open(newPath,'w')
    f.truncate()
    
    text = oldText.replace(oldF1,newF1[i])
    text = text.replace(oldA1,newA1[i])
    text = text.replace(oldE1,newE1[i])
    text = text.replace(oldm1,newm1[i])
    
    text = text.replace(oldF2,newF2[i])
    text = text.replace(oldA2,newA2[i])
    text = text.replace(oldE2,newE2[i])
    text = text.replace(oldm2,newm2[i])
    
    f.write(text)
    f.close()
    
    # Runs the algorithm
    
    FNULL = open(os.devnull,'w')
    subprocess.call(runPath,stdout=FNULL,stderr=subprocess.STDOUT)
    massLoss = np.loadtxt(massLossPath,delimiter=" ",skiprows=1)
    mass = np.loadtxt(massPath,delimiter=" ",skiprows=1)
    TaSurf = np.loadtxt(TaSurfPath,skiprows=1)
    Ta = np.loadtxt(TaPath,skiprows=1)
    
    char = mass[:,4]
    mDotg = mass[:,1]
    virgin = mass[:,3]
    wall = TaSurf[:,1]
    TC1 = Ta[:,1]
    TC2 = Ta[:,2]
    TC3 = Ta[:,3]
    TC4 = Ta[:,4]
    TC5 = Ta[:,5]
    TC6 = Ta[:,6]
    TC7 = Ta[:,7]
    
    resp.append([char,mDotg,virgin,wall,TC1,TC2,TC3,TC4,TC5,TC6,TC7])
    
    # Timer and iterations
    
    step = time.time()-start
    hours = (nbrPts-i)*step/3600
    minutes = int(60*(hours-int(hours)))
    hours = int(hours)
    
    timeLeft = str(hours)+' h '+str(minutes)+' min'
    percent = str(round((i+1)/nbrPts*100,2))+' %'
    step = str(round(step,2))+' s' 
    
    print(percent+'\t\t'+step+'\t\t'+timeLeft)
    
# %% Save Results

np.savez_compressed("point.npz",pts=point)
np.savez_compressed("resp.npz",resp=resp)

charMean = np.mean(resp[:,0],axis=0)
mDotgMean = np.mean(resp[:,1],axis=0)
virginMean = np.mean(resp[:,2],axis=0)
wallMean = np.mean(resp[:,3],axis=0)
TC1Mean = np.mean(resp[:,4],axis=0)
TC2Mean = np.mean(resp[:,5],axis=0)
TC3Mean = np.mean(resp[:,6],axis=0)
TC4Mean = np.mean(resp[:,7],axis=0)
TC5Mean = np.mean(resp[:,8],axis=0)
TC6Mean = np.mean(resp[:,9],axis=0)
TC7Mean = np.mean(resp[:,10],axis=0)

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
plt.plot(mDotgMean)
plt.ylabel('Mean')

plt.figure(2)
plt.plot(charMean)
plt.plot(virginMean)
plt.ylabel('Mean')

plt.figure(3)
plt.plot(wallMean,'C0')
plt.plot(TC1Mean,'C1')
plt.plot(TC2Mean,'C2')
plt.plot(TC3Mean,'C3')
plt.plot(TC4Mean,'C4')
plt.plot(TC5Mean,'C5')
plt.plot(TC6Mean,'C6')
plt.plot(TC7Mean,'C7')
plt.ylabel('Mean')
plt.xlabel('Step')