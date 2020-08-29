import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from ishig0d3 import sampler
from scipy import linalg
import chaoslib as cl
import numpy as np
import time

# %% Test QR

nbr = 2
order = 9
ptList = np.array([1e5,2e5,3e5,4e5,5e5]).astype(int)
timeUpdate = np.zeros(ptList.shape[0])
timeInv = np.zeros(ptList.shape[0])

for i in range(ptList.shape[0]):
    
    print(i)
    
    nbrPts = ptList[i]
    point = sampler(ptList[i])
    poly = cl.gschmidt(order,point)
    V = poly.eval(point)
    Jinv = V/nbrPts
    J = V.T
    
    idx = int(nbrPts/2)
    Vinv = np.linalg.pinv(V)
    V2 = np.delete(V,idx,axis=0)
    J2 = V2.T
    
    for j in range(nbr):
    
        # Update QR
        
        start = time.time()
        Jinv2 = cl.invdown(J,Jinv,idx)
        timeUpdate[i] += (time.time()-start)/nbr
        
        # Recompute the QR
        
        start = time.time()
        Jinv2 = linalg.pinv(J2)
        timeInv[i] += (time.time()-start)/nbr
    
# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

grid = [0.9,0.9,0.9]
width = 0.5
size = 8

plt.figure(1)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(ptList,timeInv,'--C0',label='Recompute $J^{-1}$')
plt.plot(ptList,timeInv,'.C0',markersize=size)
plt.ylabel('Time [s]')
plt.xlabel('m [-]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
#plt.savefig("timeInvRedo.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(ptList,timeUpdate,'--C0',label='Update $J^{-1}$')
plt.plot(ptList,timeUpdate,'.C0',markersize=size)
plt.ylabel('Time [s]')
plt.xlabel('m [-]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
#plt.savefig("timeInvUpdate.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(ptList,timeInv/timeUpdate,'--C0')
plt.plot(ptList,timeInv/timeUpdate,'.C0',markersize=size)
plt.ylabel('Speedup [-]')
plt.xlabel('m [-]')
plt.ylim([0,30])
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("speedupInv.pdf",bbox_inches="tight",format="pdf",transparent=True)