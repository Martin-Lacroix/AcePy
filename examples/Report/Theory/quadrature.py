import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 15
nbrPts = int(1e4)
grid = [0.9,0.9,0.9]
color = "C0"
width = 0.5
size = 7

plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.unicode_minus"] = 0
plt.rcParams["font.family"] = "DejaVu Sans"

def triangle(point):

    if point[0]<=0.5:
        if point[1]/2<point[0]: return 1
    elif point[0]>0.5:
        if point[1]/2<1-point[0]: return 1
    else: return 0

# %% Fekete square

point = np.zeros((nbrPts,2))
point[:,0] = np.random.uniform(0,1,nbrPts)
point[:,1] = np.random.uniform(0,1,nbrPts)

law = [cl.Uniform(0,1),cl.Uniform(0,1)]
poly = cl.polyrecur(order,law)
index,weight = cl.fekquad(point,poly)
point = point[index]

plt.figure(1)
plt.plot(point[:,0],point[:,1],".",color=color,markersize=size)
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("fekSquare.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Fekete triangle

point = np.zeros((nbrPts,2))
point[:,0] = np.random.uniform(0,1,nbrPts)
point[:,1] = np.random.uniform(0,1,nbrPts)

index = []
for i in range(nbrPts):
    if triangle(point[i]): index.append(i)

point = point[index]
law = [cl.Uniform(0,1),cl.Uniform(0,1)]
poly = cl.polyrecur(order,law)
index,weight = cl.fekquad(point,poly)
point = point[index]

plt.figure(2)
plt.plot(point[:,0],point[:,1],".",color=color,markersize=size)
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("fekTriangle.pdf",bbox_inches="tight",format="pdf",transparent=True)