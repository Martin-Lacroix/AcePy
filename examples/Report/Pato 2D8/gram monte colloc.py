import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Polynomial Chaos

order = 4
nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

# %% Figures Blues

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

mean = np.array([[0.25,1.2e4,7.1e4,3,0.2,5e8,1.7e5,3]]*100)
grid = [0.9,0.9,0.9]
ylim = [0.9,3.8]
width = 0.5

# %% Figure A

point = mean.copy()
A1 = np.linspace(0.9e4,1.5e4,100)
point[:,1] = A1

respMod = model.eval(point)
charMod = 1e3*np.max(respMod[:,0],axis=1)
with np.load('respA1.npz',mmap_mode='r') as file: resp = file['resp']
char = 1e3*np.max(resp[:,0],axis=1)

plt.figure(1)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(A1,char,'C1',label='Ref')
plt.plot(A1,charMod,'--C0',label='PCE')
plt.ylabel('$d$ Char [mm]')
plt.xlabel('$A_1$ [1/s]')
plt.ylim(ylim)
plt.legend(loc='lower right')
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig('patoDomA1.pdf',bbox_inches='tight',format='pdf',transparent=True)

point = mean.copy()
A2 = np.linspace(4.5e8,5.5e8,100)
point[:,5] = A2

respMod = model.eval(point)
charMod = 1e3*np.max(respMod[:,0],axis=1)
with np.load('respA2.npz',mmap_mode='r') as file: resp = file['resp']
char = 1e3*np.max(resp[:,0],axis=1)

plt.figure(2)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(A2,char,'C1',label='Ref')
plt.plot(A2,charMod,'--C0',label='PCE')
plt.ylabel('$d$ Char [mm]')
plt.xlabel('$A_2$ [1/s]')
plt.ylim(ylim)
plt.legend(loc='lower right')
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# %% Figure 3

point = mean.copy()
E1 = np.linspace(6e4,8e4,100)
point[:,2] = E1

respMod = model.eval(point)
charMod = 1e3*np.max(respMod[:,0],axis=1)
with np.load('respE1.npz',mmap_mode='r') as file: resp = file['resp']
char = 1e3*np.max(resp[:,0],axis=1)

plt.figure(3)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(A1,char,'C1',label='Ref')
plt.plot(A1,charMod,'--C0',label='PCE')
plt.ylabel('$d$ Char [mm]')
plt.xlabel('$E_1$ [J/mol]')
plt.ylim(ylim)
plt.legend(loc='lower right')
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig('patoDomE1.pdf',bbox_inches='tight',format='pdf',transparent=True)

point = mean.copy()
E2 = np.linspace(1.5e5,1.9e5,100)
point[:,6] = E2

respMod = model.eval(point)
charMod = 1e3*np.max(respMod[:,0],axis=1)
with np.load('respE2.npz',mmap_mode='r') as file: resp = file['resp']
char = 1e3*np.max(resp[:,0],axis=1)

plt.figure(4)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(A2,char,'C1',label='Ref')
plt.plot(A2,charMod,'--C0',label='PCE')
plt.ylabel('$d$ Char [mm]')
plt.xlabel('$E_2$ [J/mol]')
plt.ylim(ylim)
plt.legend(loc='lower right')
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# %% Figure N

point = mean.copy()
N1 = np.linspace(2.5,3.5,100)
point[:,3] = N1

respMod = model.eval(point)
charMod = 1e3*np.max(respMod[:,0],axis=1)
with np.load('respN1.npz',mmap_mode='r') as file: resp = file['resp']
char = 1e3*np.max(resp[:,0],axis=1)

plt.figure(5)
plt.plot(N1,char,'C1',label='Ref')
plt.plot(N1,charMod,'--C0',label='PCE')
plt.ylabel('$d$ Char [mm]')
plt.xlabel('$n_1$ [-]')
plt.ylim(ylim)
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig('patoDomN1.pdf',bbox_inches='tight',format='pdf',transparent=True)

point = mean.copy()
N2 = np.linspace(2.5,3.5,100)
point[:,7] = N2

respMod = model.eval(point)
charMod = 1e3*np.max(respMod[:,0],axis=1)
with np.load('respN2.npz',mmap_mode='r') as file: resp = file['resp']
char = 1e3*np.max(resp[:,0],axis=1)

plt.figure(6)
plt.plot(N2,char,'C1',label='Ref')
plt.plot(N2,charMod,'--C0',label='PCE')
plt.ylabel('$d$ Char [mm]')
plt.xlabel('$n_2$ [-]')
plt.ylim(ylim)
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig('patoDomN2.pdf',bbox_inches='tight',format='pdf',transparent=True)