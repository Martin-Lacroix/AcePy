import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Parameters

grid = [0.9,0.9,0.9]
nbrPts = 400
color = "C0"
width = 0.5
size = 7
dim = 2

plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.unicode_minus"] = 0
plt.rcParams["font.family"] = "DejaVu Sans"

sobol = cl.sobol(nbrPts,dim)
halton = cl.halton(nbrPts,dim)
rseq = cl.rseq(nbrPts,dim)

uniform = np.zeros((nbrPts,2))
uniform[:,0] = np.random.uniform(0,1,nbrPts)
uniform[:,1] = np.random.uniform(0,1,nbrPts)

# %% Figures

plt.figure(1)
plt.plot(halton[:,0],halton[:,1],".",color=color,markersize=size,label="Halton")
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-] Halton")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("seqHalton.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.plot(sobol[:,0],sobol[:,1],".",color=color,markersize=size,label="Sobol")
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-] Sobol")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("seqSobol.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3)
plt.plot(rseq[:,0],rseq[:,1],".",color=color,markersize=size,label="R-sequence")
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-] R-sequence")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("seqRseq.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4)
plt.plot(uniform[:,0],uniform[:,1],".",color=color,markersize=size,label="Uniform")
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-] Uniform")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("seqUniform.pdf",bbox_inches="tight",format="pdf",transparent=True)