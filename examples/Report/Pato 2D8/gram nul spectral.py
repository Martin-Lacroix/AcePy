import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from scipy import interpolate
import chaoslib as cl
import numpy as np
import parula

# %% Initialisation

order = 8
nbrPts = int(1e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

# %% Polynomial Chaos

poly = cl.gschmidt(order,point,trunc=0.85)
print(poly[:].shape[0])

V = poly.eval(point)
m1 = np.sum(V,axis=0)/nbrPts

index,weight = cl.nulquad(point,poly)
point = point[index]
resp = resp[index]

V = poly.eval(point)
V = np.transpose(weight*V.T)
m2 = np.sum(V,axis=0)

poly.trunc(4)

coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

# %% Error

resp = []
nbrPts = int(2e4)

with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]
respMod = model.eval(point)

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)
meanMod = np.mean(respMod,axis=0)
varMod = np.var(respMod,axis=0)

num = np.mean((resp-respMod)**2,axis=0)
den = np.mean(resp**2,axis=0)
error = np.divide(num,den)

# %% Variation

nbrPts = int(2e4)
with np.load("resp.npz",mmap_mode="r") as file: resp = file["resp"]
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

bound = np.sort(respMod,axis=0)
bound = bound[int(0.01*nbrPts):int(0.99*nbrPts)]
down = bound[0]
up = bound[-1]

# %% Figures

parula = parula.colour()
stdMod = np.sqrt(varMod)
std = np.sqrt(var)

t = np.linspace(0,60,121)
x = [1,2,4,8,12,16,24]
grid = [0.9,0.9,0.9]
width = 0.5

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1,figsize=(8,4))
plt.plot(t,std[4],'C0',label='Ref')
for i in range(5,11): plt.plot(t,std[i],'C0')
for i in range(4,11): plt.plot(t,stdMod[i],'--',label=str(x[i-4])+' mm')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel('SD T [K]')
plt.xlabel('Time [s]')
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoTStd.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.fill_between(t,up[1],down[1],color='C0',alpha=0.2)
plt.plot(t,mean[1],'C1',label='Ref')
plt.plot(t,meanMod[1],'--C0',label='PCE')
plt.ylabel('$\dot{m}_g$ [kg/m$^2$s]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMgMean.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.plot(t,std[1],'C1',label='Ref')
plt.plot(t,stdMod[1],'--C0',label='PCE')
plt.ylabel('SD [kg/m$^2$s]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMgStd.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4)
plt.fill_between(t,1e3*up[0],1e3*down[0],color='C0',alpha=0.2)
plt.fill_between(t,1e3*up[2],1e3*down[2],color='C2',alpha=0.2)
plt.plot(t,1e3*mean[0],'C1',label='Ref')
plt.plot(t,1e3*mean[2],'C1')
plt.plot(t,1e3*meanMod[0],'--C0',label='Char')
plt.plot(t,1e3*meanMod[2],'--C2',label='Virg')
plt.ylabel('$d$ [mm]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharVirgMean.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(5)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.plot(t,1e3*std[0],'C1',label='Ref')
plt.plot(t,1e3*std[2],'C1')
plt.plot(t,1e3*stdMod[0],'--C0',label='Char')
plt.plot(t,1e3*stdMod[2],'--C2',label='Virg')
plt.ylabel('SD [mm]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharVirgStd.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% 3D Figure

fig = plt.figure(6,figsize=(12,6))
ax = fig.gca(projection='3d')
ax.set_xlabel('$x$ [mm]',labelpad=20)
ax.set_ylabel('$t$ [s]',labelpad=15)
ax.set_zlabel("Mean $T$ [K]",labelpad=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.zaxis.set_major_locator(plt.MaxNLocator(4))
ax.xaxis.set_pane_color((1,1,1,0))
ax.yaxis.set_pane_color((1,1,1,0))
ax.zaxis.set_pane_color((1,1,1,0))
ax.xaxis._axinfo['grid'] = {'color': grid, 'linewidth': 0.5, 'linestyle': '-'}
ax.yaxis._axinfo['grid'] = {'color': grid, 'linewidth': 0.5, 'linestyle': '-'}
ax.zaxis._axinfo['grid'] = {'color': grid, 'linewidth': 0.5, 'linestyle': '-'}
plt.tight_layout()

T = mean[3:11]
t = np.linspace(0,60,121)
x = [0,1,2,4,8,12,16,24]
fun = interpolate.interp2d(t,x,T,kind='cubic')
t = np.linspace(0,60,60)
x = np.linspace(0,24,60)
T = fun(t,x)
t,x = np.meshgrid(t,x)

ax.plot_surface(x,t,T,rstride=1,cstride=1,cmap=parula)

T = meanMod[3:11]
t = np.linspace(0,60,121)
x = [0,1,2,4,8,12,16,24]
t,x = np.meshgrid(t,x)

ax.scatter(x,t,T,color='white',s=3,alpha=1)
plt.savefig("patoTMean.pdf",bbox_inches="tight",format="pdf",transparent=True)