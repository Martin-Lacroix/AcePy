import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl

# %% Initialisation

order = 7
nbrVar = 2
grid = [0.9,0.9,0.9]
color = "C0"
width = 0.5
size = 12

plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.unicode_minus"] = 0
plt.rcParams["font.family"] = "DejaVu Sans"

# %% Graphes

trunc = 1
expo = cl.indextens(order,nbrVar,trunc)

plt.figure(1)
plt.plot(expo[0],expo[1],".",color=color,markersize=size,label="$q=$"+str(trunc))
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("hyperQ1.pdf",bbox_inches="tight",format="pdf",transparent=True)

trunc = 0.75
expo = cl.indextens(order,nbrVar,trunc)

plt.figure(2)
plt.plot(expo[0],expo[1],".",color=color,markersize=size,label="$q=$"+str(trunc))
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("hyperQ075.pdf",bbox_inches="tight",format="pdf",transparent=True)

trunc = 0.5
expo = cl.indextens(order,nbrVar,trunc)

plt.figure(3)
plt.plot(expo[0],expo[1],".",color=color,markersize=size,label="$q=$"+str(trunc))
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("hyperQ05.pdf",bbox_inches="tight",format="pdf",transparent=True)

trunc = 0.25
expo = cl.indextens(order,nbrVar,trunc)

plt.figure(4)
plt.plot(expo[0],expo[1],".",color=color,markersize=size,label="$q=$"+str(trunc))
plt.ylabel("$x_2$ [-]")
plt.xlabel("$x_1$ [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("hyperQ025.pdf",bbox_inches="tight",format="pdf",transparent=True)