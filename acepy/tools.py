from pickle import dump
import numpy as np
import sys

# %% Save and Progress Functions

def save(item,name):

    file = open(name+'.pickle','wb')
    dump(item,file)
    file.close()

def timer(step,end,message):

    sys.stdout.write('\r'+message+str(round(100*(step+1)/end))+' %\t')
    sys.stdout.flush()

# %% Start and End Function Decorator

def start_end(func):
    def wrapper(*args,**kwargs):
        
        text = '\rComputing '+func.__name__
        sys.stdout.write(text+' ...\t')
        sys.stdout.flush()

        result = func(*args,**kwargs)

        sys.stdout.write(text+' 100%\t')
        sys.stdout.flush()
        sys.stdout.write('\n')
        
        return result
    return wrapper

# %% Pseudo-Inverse Downgrade for Column Removal

def invdown(A,Ainv,index):

    v1 = A[:,index]
    v2 = Ainv[index]
    alp = np.dot(v2,v1)
    Ainv = np.delete(Ainv,index,axis=0)
    tol = 1-1e-9
    
    if alp<tol: v = np.tensordot(v1/(1-alp),v2,axes=0)
    else: v = -np.tensordot(v2/np.dot(v2,v2),v2,axes=0)
    
    np.fill_diagonal(v,v.diagonal()+1)
    Ainv = np.dot(Ainv,v)
    return Ainv

# %% PCA Whitening for Linearly Separable Variables

class Pca:

    def __init__(self,point):

        self.mean = np.mean(point,axis=0)
        self.std = np.std(point,axis=0,ddof=1)

        # Standardizes and computes the whitening matrix

        point = (point-self.mean)/self.std
        cov = np.cov(point.T)
        val,vec = np.linalg.eig(cov)

        self.A = np.diag(np.sqrt(1/val)).dot(vec.T)
        self.invA = np.linalg.inv(self.A)

    # Mapping from correlated to whitened

    def white(self,point):

        point = np.transpose((point-self.mean)/self.std)
        point = np.transpose(self.A.dot(point))
        return point

    # Mapping from whitened to correlated

    def corr(self,point):

        point = np.dot(self.invA,point.T)
        point = self.std*point.T+self.mean
        return point
    