import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from ishig0d3 import sampler
from scipy import linalg
import chaoslib as cl
import numpy as np
import time

# %% Test QR

nbr = 20
order = 9
ptList = np.array([1e5,2e5,3e5,4e5,5e5]).astype(int)
timeUpdate = np.zeros(ptList.shape[0])
timeQR = np.zeros(ptList.shape[0])

for i in range(ptList.shape[0]):
    
    print(i)
    
    nbrPts = ptList[i]
    point = sampler(ptList[i])
    poly = cl.gschmidt(order,point)
    V = poly.eval(point)
    
    idx = int(nbrPts/2)
    Q,R = linalg.qr(V,mode='economic')
    V = np.delete(V,idx,axis=0)
    
    for j in range(nbr):
        
        # Update QR
    
        start = time.time()
        Q,R = linalg.qr_delete(Q,R,idx,overwrite_qr=1,check_finite=0)
        timeUpdate[i] += (time.time()-start)/nbr
        
        # Recompute the QR
        
        start = time.time()
        Q,R = linalg.qr(V,mode='economic',overwrite_a=1,check_finite=0)
        timeQR[i] += (time.time()-start)/nbr
    
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
plt.plot(ptList,timeQR,'--C0',label='Recompute QR')
plt.plot(ptList,timeQR,'.C0',markersize=size)
plt.ylabel('Time [s]')
plt.xlabel('m [-]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
#plt.savefig("timeQRredo.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(ptList,timeUpdate,'--C0',label='Update QR')
plt.plot(ptList,timeUpdate,'.C0',markersize=size)
plt.ylabel('Time [s]')
plt.xlabel('m [-]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
#plt.savefig("timeQRupdate.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(ptList,timeQR/timeUpdate,'--C0')
plt.plot(ptList,timeQR/timeUpdate,'.C0',markersize=size)
plt.ylabel('Speedup [-]')
plt.xlabel('m [-]')
plt.ylim([0,50])
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("speedupQR.pdf",bbox_inches="tight",format="pdf",transparent=True)