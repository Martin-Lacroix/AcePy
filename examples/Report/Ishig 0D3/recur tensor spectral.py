import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from ishig0d3 import sampler,response
import chaoslib as cl
import numpy as np

# %% Initialisation

dist = []
model = []
var = np.zeros(3)
mean = np.zeros(3)
S = np.zeros((2,3))
St = np.zeros((2,3))
dist = cl.Joint(3*[cl.Uniform(-np.pi,np.pi)])
orders = np.arange(2,11)

# %% Polynomial Chaos

for order in orders:

    ordQuad = 2*order-1
    point,weight = cl.tensquad(ordQuad,dist)
    resp = response(point)

    poly = cl.polyrecur(order,dist)
    coef = cl.spectral(resp,poly,point,weight)
    model.append(cl.Expansion(coef,poly))
    sobol = cl.anova(coef,poly)
    S[0] = sobol['S']
    St[0] = sobol['ST']

# %% Statistical Moments

a = 7
b = 0.1

var = a**2/8+b*np.pi**4/5+b**2*np.pi**8/18+1/2
S[1] = np.array([0.5*(1+b*np.pi**4/5)**2,a**2/8,0])/var
St[1] = np.array([0.5*(1+b*np.pi**4/5)**2+8*b**2*np.pi**8/225,a**2/8,8*b**2*np.pi**8/225])/var

# %% Error Solution

nbrPts = int(1e5)
point = sampler(nbrPts)
resp = response(point)
error = []

for i in range(len(orders)):

    print('Error',i)
    respMod = model[i].eval(point)
    num = np.mean((resp-respMod)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    error.append(np.divide(num,den))

# %% Error Figures

#sns.set_style('whitegrid')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

grid = [0.9,0.9,0.9]
width = 0.5
size = 8

plt.figure(1,figsize=(8,4))
plt.semilogy(orders,error,'C0--',label='PCE')
plt.semilogy(orders,error,'.C0',markersize=size)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel('SRE [-]')
plt.xlabel('Order [-]')
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig('ishigError.pdf',bbox_inches='tight',format='pdf',transparent=True)