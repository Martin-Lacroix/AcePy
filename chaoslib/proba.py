from .math import halton,sobol,rseq
from scipy import special
import numpy as np

# %% Uniform Distribution

class Uniform:
    """Class of uniform law with lower and upper boundaries"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.pdf = lambda x: np.array(x)**0/(b-a)
        self.cdf = lambda x: (np.array(x)-a)/(b-a)
        self.invcdf = lambda x: (b-a)*np.array(x)+a
        self.random = lambda x: np.random.uniform(a,b,x)

        self.rseq = lambda x: self.invcdf(rseq(x))
        self.sobol = lambda x: self.invcdf(sobol(x))
        self.halton = lambda x: self.invcdf(halton(x))

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0].fill((b+a)/2)
        coef[1] = ((b-a)*n/2)**2/(4*n**2-1)
        return coef

# %% Normal Distribution

class Normal:
    """Class of normal law with mean and standard deviation"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.invcdf = lambda x: a+np.sqrt(2)*b*special.erfinv(2*np.array(x)-1)
        self.cdf = lambda x: 0.5*(1+special.erf((np.array(x)-a)/(b*np.sqrt(2))))
        self.pdf = lambda x: np.exp(-0.5*((np.array(x)-a)/b)**2)/(b*np.sqrt(2*np.pi))
        self.random = lambda x: np.random.normal(a,b,x)

        self.rseq = lambda x: self.invcdf(rseq(x))
        self.sobol = lambda x: self.invcdf(sobol(x))
        self.halton = lambda x: self.invcdf(halton(x))

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0].fill(a)
        coef[1] = b**2*n
        return coef

# %% Exponential Distribution

class Expo:
    """Class of exponential law with inverse scale"""

    def __init__(self,a):

        self.arg = a
        self.cdf = lambda x: 1-np.exp(-a*np.array(x))
        self.pdf = lambda x: a*np.exp(-a*np.array(x))
        self.invcdf = lambda x: -np.log(1-np.array(x))/a
        self.random = lambda x: np.random.exponential(a,x)

        self.rseq = lambda x: self.invcdf(rseq(x))
        self.sobol = lambda x: self.invcdf(sobol(x))
        self.halton = lambda x: self.invcdf(halton(x))

    def coef(self,nbrCoef):

        a = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = a*(1+2*n)
        coef[1] = (a*n)**2
        return coef

# %% Gamma Distribution

class Gamma:
    """Class of gamma law with shape and scale"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.cdf = lambda x: special.gammainc(a,np.array(x))
        self.invcdf = lambda x: b*special.gammaincinv(a,np.array(x))
        self.pdf = lambda x: x**(a-1)*np.exp(-np.array(x)/b)/(special.gamma(a)*b**a)
        self.random = lambda x: np.random.gamma(a,b,x)

        self.rseq = lambda x: self.invcdf(rseq(x))
        self.sobol = lambda x: self.invcdf(sobol(x))
        self.halton = lambda x: self.invcdf(halton(x))

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = (2*n+a)*b
        coef[1] = (n+a-1)*n*b**2
        return coef

# %% Lognormal Distribution

class Lognorm:
    """Class of lognormal law with mean and variance"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.invcdf = lambda x: np.exp(a+np.sqrt(2)*b*special.erfinv(2*np.array(x)-1))
        self.cdf = lambda x: 0.5*(1+special.erf((np.log(np.array(x))-a)/(b*np.sqrt(2))))
        self.pdf = lambda x: np.exp(-0.5*((np.log(np.array(x))-a)/b)**2)/(np.array(x)*b*np.sqrt(2*np.pi))
        self.random = lambda x: np.random.lognormal(a,b,x)

        self.rseq = lambda x: self.invcdf(rseq(x))
        self.sobol = lambda x: self.invcdf(sobol(x))
        self.halton = lambda x: self.invcdf(halton(x))

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = (np.exp((n+1)*b**2)+np.exp(n*b**2)-1)*np.exp(((2*n-1)*b**2)/2+a)
        coef[1] = (np.exp(n*b**2)-1)*np.exp((3*n-2)*b**2+2*a)
        return coef

# %% Beta Distribution

class Beta:
    """Class of beta law with shape parameters"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.cdf = lambda x: special.betainc(a,b,np.array(x))
        self.invcdf = lambda x: special.betaincinv(a,b,np.array(x))
        self.pdf = lambda x: x**(a-1)*(1-np.array(x))**(b-1)/special.beta(a,b)
        self.random = lambda x: np.random.beta(a,b,x)

        self.rseq = lambda x: self.invcdf(rseq(x))
        self.sobol = lambda x: self.invcdf(sobol(x))
        self.halton = lambda x: self.invcdf(halton(x))

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))

        nab = 2*n+a+b
        B1 = a*b*1./((a+b+1)*(a+b)**2)
        B2 = (n+a-1)*(n+b-1)*n*(n+a+b-2)/((nab-1)*(nab-3)*(nab-2)**2+2*((n==0)+(n==1)))
        coef[0] = ((a-1)**2-(b-1)**2)*0.5/(nab*(nab-2)+(nab==0)+(nab==2))+0.5
        coef[1] = np.where((n==0)+(n==1),B1,B2)
        return coef

# %% Joint Distribution

class Joint:
    """Class of joint probability density function"""

    def __init__(self,dist): self.dist = np.copy(np.atleast_1d(dist))
    def __setitem__(self,i,dist): self.dist[i] = dist
    def __getitem__(self,i): return self.dist[i]

    def pdf(self,point):

        dim = self.dist.shape[0]
        point = np.atleast_2d(point)
        resp = [self.dist[i].pdf(point[:,i]) for i in range(dim)]
        resp = np.squeeze(np.prod(resp,axis=0))
        return resp

    def random(self,nbrPts):

        dim = self.dist.shape[0]
        point = np.transpose([self.dist[i].random(nbrPts) for i in range(dim)])
        return point

    def rseq(self,nbrPts):

        dim = self.dist.shape[0]
        point = rseq(nbrPts,dim)
        point = np.transpose([self.dist[i].invcdf(point[:,i]) for i in range(dim)])
        return point

    def sobol(self,nbrPts):

        dim = self.dist.shape[0]
        point = sobol(nbrPts,dim)
        point = np.transpose([self.dist[i].invcdf(point[:,i]) for i in range(dim)])
        return point

    def halton(self,nbrPts):

        dim = self.dist.shape[0]
        point = halton(nbrPts,dim)
        point = np.transpose([self.dist[i].invcdf(point[:,i]) for i in range(dim)])
        return point