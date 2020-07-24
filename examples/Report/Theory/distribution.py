import sys
sys.path.append('../../../')
from matplotlib.ticker import StrMethodFormatter
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Parameters

order = 20
nbrPts = int(2e6)
grid = [0.9,0.9,0.9]
width = 0.5

plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.unicode_minus"] = 0
plt.rcParams["font.family"] = "DejaVu Sans"

# %% Normal y and Uniform x

xLaw = cl.Uniform(-1,1)
yLaw = cl.Normal(0,1)

y = cl.transfo(yLaw.invcdf,order,xLaw)
xRand = np.random.uniform(-1,1,nbrPts)
xPlot = np.linspace(-3,3,1000)

plt.figure(1)
plt.plot(xPlot,yLaw.pdf(xPlot),"C0",label="Normal")
plt.hist(y(xRand),color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_Y\,(y)$ [-]")
plt.xlabel("$y$ [-]")
plt.xlim(-3,3)

# %% Expo y and Normal x

xLaw = cl.Normal(0,1)
yLaw = cl.Expo(1)

y = cl.transfo(yLaw.invcdf,order,xLaw)
xRand = np.random.normal(0,1,nbrPts)
xPlot = np.linspace(0,10,1000)

plt.figure(2)
plt.plot(xPlot,yLaw.pdf(xPlot),"C0",label="Expo")
plt.hist(y(xRand),color="w",bins=100,ec="lightgray",density=True)
plt.ylabel("$f_Y\,(y)$ [-]")
plt.xlabel("$y$ [-]")
plt.xlim(0,5)

# %% Beta y and Uniform x

xLaw = cl.Expo(1)
yLaw = cl.Beta(0.5,0.5)

y = cl.transfo(yLaw.invcdf,order,xLaw)
xRand = np.random.exponential(1,nbrPts)
xPlot = np.linspace(0,1,int(1e4))

plt.figure(3)
plt.plot(xPlot,yLaw.pdf(xPlot),"C0",label="Beta")
plt.hist(y(xRand),color="w",bins=40,ec="lightgray",density=True)
plt.ylabel("$f_Y\,(y)$ [-]")
plt.xlabel("$y$ [-]")
plt.xlim(0,1)
plt.ylim(0,2.5)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("distBeta.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Gamma y and Uniform x

xLaw = cl.Uniform(-1,1)
yLaw = cl.Gamma(3,2)

y = cl.transfo(yLaw.invcdf,order,xLaw)
xRand = np.random.uniform(-1,1,nbrPts)
xPlot = np.linspace(0,25,1000)

plt.figure(4)
plt.plot(xPlot,yLaw.pdf(xPlot),"C0",label="Gamma")
plt.hist(y(xRand),color="w",bins=40,ec="lightgray",density=True)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.ylabel("$f_Y\,(y)$ [-]")
plt.xlabel("$y$ [-]")
plt.xlim(0,20)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("distGamma.pdf",bbox_inches="tight",format="pdf",transparent=True)