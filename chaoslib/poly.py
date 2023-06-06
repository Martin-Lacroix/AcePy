from scipy import sparse,linalg
from .tools import start_end
from .proba import Joint
import numpy as np

# %% Class of Orthogonal Polynomials Basis

class Polynomial:

    def __init__(self,expo,coef,csr=0):

        if sparse.issparse(coef): self.coef = coef.copy()
        else: self.coef = np.copy(np.atleast_2d(np.transpose(coef)).T)
        if csr: self.coef = sparse.csr_matrix(self.coef)

        self.expo = np.copy(np.atleast_2d(expo))
        self.dim = self.expo.shape[0]

    def __getitem__(self,i): return self.coef[i]
    def __setitem__(self,i,coef): self.coef[i] = coef

    # Computes the Vandermonde matrix by evaluating the basis at the points

    def eval(self,point):

        V = 1
        point = np.reshape(np.transpose(point),(self.dim,-1)).T
        for i in range(self.dim): V *= np.power(point[:,i,None],self.expo[i])
        V = np.transpose(self.coef.dot(V.T))
        return V

    # Truncates the polynomial basis at a lower order

    def trunc(self,order):

        nbrPoly = self.coef.shape[0]
        id1 = np.where(np.sum(self.expo,axis=0)<=order)[0]
        id2 = np.setdiff1d(np.arange(self.expo.shape[1]),id1)
        id2 = np.setdiff1d(range(nbrPoly),self.coef[:,id2].nonzero()[0])

        self.coef = self.coef[id2]
        self.coef = self.coef[:,id1]
        self.expo = self.expo[:,id1]

        idx = np.unique(self.coef.nonzero()[1])
        self.expo = self.expo[:,idx]
        self.coef = self.coef[:,idx]

    # Removes all the polynomials from the basis except index

    def clean(self,idx):

        self.coef = self.coef[idx]
        idx = np.unique(self.coef.nonzero()[1])
        self.expo = self.expo[:,idx]
        self.coef = self.coef[:,idx]

# %% Orthonormal Polynomial Basis with Gram-Schmidt

@start_end
def gschmidt(order,point,weight=0,trunc=1):

    point = np.atleast_2d(np.transpose(point)).T
    if not np.any(weight): weight = 1/point.shape[0]
    dim = point.shape[1]

    # Creates the tensor product of univariate polynomials

    nbrPoly = order+1
    expo = indextens(order,dim,trunc)
    nbrPoly = expo.shape[1]
    coef = sparse.eye(nbrPoly)
    base = Polynomial(expo,coef)

    # Computes modified Gram-Schmidt algorithm

    V = base.eval(point)
    V = np.transpose(np.sqrt(weight)*V.T)
    R = np.linalg.qr(V,'r')

    if V.shape[0]<nbrPoly: raise Exception('Underdetermined system')
    coef = linalg.lapack.dtrtri(R)[0].T
    coef[0,0] = 1

    poly = Polynomial(expo,coef,1)
    return poly

# %% Orthogonal Polynomials with Recurrence Coefficients

@start_end
def polyrecur(order,dist,trunc=1):

    if not isinstance(dist,Joint): dist = Joint(dist)

    nbrPoly = order+1
    dim = dist[:].shape[0]
    expo = np.arange(nbrPoly)
    coef = np.zeros((nbrPoly,nbrPoly))
    norm = np.ones((dim,nbrPoly))

    coef[0,0] = 1
    polyList = []

    # Creates the univariate polynomial basis

    for i in range(dim):

        polyList.append(Polynomial(expo,coef))
        AB = dist[i].coef(nbrPoly)

        for j in range(1,nbrPoly):

            norm[i,j] = norm[i,j-1]*AB[1,j]
            polyList[i][j] = np.roll(polyList[i][j-1],1,axis=0)
            polyList[i][j] -= AB[0,j-1]*polyList[i][j-1]+AB[1,j-1]*polyList[i][j-2]

    # Normalization and tensor product

    for i in range(dim): polyList[i][:] /= np.sqrt(norm[i,:,None])
    poly = tensdot(polyList,order,trunc)
    return poly

# %% Tensor product of Univariate Polynomial Basis

def tensdot(polyList,order,trunc):

    def reshape(poly,expo):

        poly.coef = poly[:][:,expo]
        poly.expo = expo
        return poly

    dim = len(polyList)
    expo = indextens(order,dim,trunc)
    nbrPoly = expo.shape[1]
    coef = np.eye(nbrPoly)

    # Tensor product of the univariate basis

    for i in range(dim): polyList[i] = reshape(polyList[i],expo[i])
    for i in range(nbrPoly): coef[i] = np.prod([polyList[j][expo[j,i]] for j in range(dim)],axis=0)

    poly = Polynomial(expo,coef,1)
    return poly

# %% Multi-Index Matrix of Polynomial Tensor Product

def indextens(order,dim,trunc):

    nbrPoly = order+1
    base = np.arange(nbrPoly)[:,None]
    index = base.copy()

    # Creates the matrix of index

    for i in range(1,dim):

        v1 = np.tile(index,(nbrPoly,1))
        v2 = np.repeat(base,v1.shape[0]/nbrPoly,axis=0)

        index = np.concatenate((v2,v1),axis=1)
        norm = np.sum(index**trunc,axis=1)**(1/trunc)
        index = index[norm.round(6)<=order]
        index = index[np.argsort(np.sum(index,axis=-1))]

    index = np.transpose(index)
    return index
