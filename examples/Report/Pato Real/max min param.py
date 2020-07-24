import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 5
nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

# %% Polynomial Chaos

poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

# %% Max Min

nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

index = np.argmax(point[:,1])
maxA1 = point[index]

index = np.argmin(point[:,2])
minE1 = point[index]

respMax = model.eval(maxA1)
respMin = model.eval(minE1)

# %% Figures

t = np.linspace(0,60,121)
C = ["C3","C1","C2","C4","C5","C6","C7","C8"]
x = [1,2,4,8,12,16,24]
grid = [0.9,0.9,0.9]
width = 0.5

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.plot(t,respMax[1],'C0',label='$(A,E)_1$ max')
plt.plot(t,respMin[1],'--C1',label='$(A,E)_1$ min')
plt.ylabel('$\dot{m}_g$ [kg/m$^2$s]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMaxMg.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.plot(t,1e3*respMax[0],'C0',label='$(A,E)_1$ max')
plt.plot(t,1e3*respMin[0],'--C1',label='$(A,E)_1$ min')
plt.ylabel('$d$ Char [mm]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMaxChar.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3)
plt.plot(t,1e3*respMax[2],'C0',label='$(A,E)_1$ max')
plt.plot(t,1e3*respMin[2],'--C1',label='$(A,E)_1$ min')
plt.ylabel('$d$ Virgin [mm]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMaxVirg.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4)
plt.plot(t,respMax[-1],'C0',label='$(A,E)_1$ max')
plt.plot(t,respMin[-1],'--C1',label='$(A,E)_1$ min')
plt.ylabel('$T$ [K]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMaxT.pdf",bbox_inches="tight",format="pdf",transparent=True)