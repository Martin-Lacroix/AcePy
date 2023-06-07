from scipy import special
import numpy as np

# %% Uniform Law with Lower and Upper Boundaries

class Uniform:

    def __init__(self,A,B):

        self.A = A
        self.B = B

    # Uniform probability density function

    def pdf(self,x):
        return np.array(x).fill(1)/(self.B-self.A)
    
    def cdf(self,x):
        return (np.array(x)-self.A)/(self.B-self.A)

    def invcdf(self,x):
        return (self.B-self.A)*np.array(x)+self.A
    
    def random(self,x):
        return np.random.uniform(self.A,self.B,x)

    def coef(self,nbrCoef):

        coef = np.zeros((2,nbrCoef))
        N = np.square(np.arange(nbrCoef))

        # Compute the three term recurrence relation coefficients
        
        coef[0].fill((self.B+self.A)/2)
        coef[1] = N*np.square((self.B-self.A)/2)/(4*N-1)
        return coef

# %% Normal Law with Mean and Standard Deviation

class Normal:

    def __init__(self,A,B):

        self.A = A
        self.B = B

    # Normal probability density function

    def pdf(self,x):
        
        E = np.exp(-np.square((np.array(x)-self.A)/self.B)/2)
        return E/(self.B*np.sqrt(2*np.pi))
    
    def cdf(self,x):

        E = (np.array(x)-self.A)/(np.sqrt(2)*self.B)
        return (1+special.erf(E))/2

    def invcdf(self,x):

        E = special.erfinv(2*np.array(x)-1)
        return self.A+np.sqrt(2)*self.B*E
    
    def random(self,x):
        return np.random.normal(self.A,self.B,x)

    def coef(self,nbrCoef):

        N = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))

        # Compute the three term recurrence relation coefficients

        coef[0].fill(self.A)
        coef[1] = N*np.square(self.B)
        return coef

# %% Exponential Law with Inverse Scale

class Expo:

    def __init__(self,A):

        self.A = A

    # Exponential probability density function

    def pdf(self,x):
        return self.A*np.exp(-self.A*np.array(x))
    
    def cdf(self,x):
        return 1-np.exp(-self.A*np.array(x))

    def invcdf(self,x):
        return -np.log(1-np.array(x))/self.A

    def random(self,x):
        return np.random.exponential(self.A,x)

    def coef(self,nbrCoef):

        N = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))

        # Compute the three term recurrence relation coefficients

        coef[0] = self.A*(1+2*N)
        coef[1] = np.square(self.A*N)
        return coef

# %% Gamma Law with Shape and Scale

class Gamma:

    def __init__(self,A,B):

        self.A = A
        self.B = B

    # Gamma probability density function

    def cdf(self,x):
        return special.gammainc(self.A,np.array(x))

    def invcdf(self,x):
        return self.B*special.gammaincinv(self.A,np.array(x))

    def pdf(self,x):

        E = np.power(x,self.A-1)*np.exp(-np.array(x)/self.B)
        return E/(special.gamma(self.A)*np.power(self.B,self.A))
    
    def random(self,x):
        return np.random.gamma(self.A,self.B,x)

    def coef(self,nbrCoef):

        N = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))

        # Compute the three term recurrence relation coefficients

        coef[0] = (2*N+self.A)*self.B
        coef[1] = (N+self.A-1)*N*np.square(self.B)
        return coef

# %% Lognormal Law with Mean and Variance

class Lognorm:

    def __init__(self,A,B):

        self.A = A
        self.B = B

    # Lognormal probability density function

    def pdf(self,x):

        E = np.exp(-np.square((np.log(np.array(x))-self.A)/self.B)/2)
        return E/(np.array(x)*self.B*np.sqrt(2*np.pi))
    
    def cdf(self,x):

        E = (np.log(np.array(x))-self.A)/(np.sqrt(2)*self.B)
        return 0.5*(1+special.erf(E))
    
    def invcdf(self,x):

        E = special.erfinv(2*np.array(x)-1)
        return np.exp(self.A+np.sqrt(2)*self.B*E)
    
    def random(self,x):
        return np.random.lognormal(self.A,self.B,x)

    def coef(self,nbrCoef):

        B = np.square(self.B)
        N = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))

        # Compute the three term recurrence relation coefficients

        coef[0] = (np.exp(N*B+B)+np.exp(N*B)-1)*np.exp((2*N*B-B)/2+self.A)
        coef[1] = (np.exp(N*B)-1)*np.exp((3*N-2)*B+2*self.A)
        return coef

# %% Beta Law with Shape Parameters

class Beta:

    def __init__(self,A,B):

        self.A = A
        self.B = B

    # Beta probability density function

    def pdf(self,x):

        E = np.power(x,self.A-1)*np.power(1-np.array(x),self.B-1)
        return E/special.beta(self.A,self.B)
    
    def cdf(self,x):
        return special.betainc(self.A,self.B,np.array(x))
    
    def invcdf(self,x):
        return special.betaincinv(self.A,self.B,np.array(x))

    def random(self,x):
        return np.random.beta(self.A,self.B,x)

    def coef(self,nbrCoef):

        N = np.arange(nbrCoef)
        AB = 2*N+self.A+self.B
        coef = np.zeros((2,nbrCoef))

        # Define some temporary variables for the recurrence

        B1 = self.A*self.B/((self.A+self.B+1)*np.power(self.A+self.B,2))
        B2 = (AB-1)*(AB-3)*np.power(AB-2,2)+2*((N==0)+(N==1))
        B3 = (N+self.A-1)*(N+self.B-1)*N*(N+self.A+self.B-2)

        # Compute the three term recurrence relation coefficients

        coef[0] = (np.power(self.A-1,2)-np.power(self.B-1,2))/2
        coef[0] = coef[0]/(AB*(AB-2)+(AB==0)+(AB==2))+0.5
        coef[1] = np.where((N==0)+(N==1),B1,B3/B2)
        return coef

# %% Joint Probability Density Function

class Joint:

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
        point = [self.dist[i].random(nbrPts) for i in range(dim)]
        return np.transpose(point)
    