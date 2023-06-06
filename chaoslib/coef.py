from .tools import timer,printer
import numpy as np

# %% Expansion Coefficients with Spectral Projection

def spectral(resp,poly,point,weight=0):

    printer(0,'Computing coefficients ...')

    V = poly.eval(point)
    resp = np.array(resp)
    if not np.any(weight): weight = 1/V.shape[0]

    # Computes the polynomial chaos coefficients

    V = np.transpose(weight*V.T)
    coef = np.transpose(np.dot(resp.T,V))

    printer(1,'Computing coefficients 100 %')
    return coef

# %% Expansion Coefficients with Least-Square Regression

def colloc(resp,poly,point,weight=0):

    printer(0,'Computing coefficients ...')

    resp = np.array(resp)
    shape = (poly[:].shape[0],)+resp.shape[1:]
    resp = resp.reshape(resp.shape[0],-1)
    V = poly.eval(point)

    # Solves the least squares linear system
    
    coef = square(V,resp,weight)
    coef = coef.reshape(shape)
    
    printer(1,'Computing coefficients 100 %')
    return coef

# %% Expansion Coefficients with Least Angle Regression

def lars(resp,poly,point,weight=0,it=np.inf):
    
    printer(0,'Computing coefficients ...')

    resp = np.array(resp)
    shape = (poly[:].shape[0],)+resp.shape[1:]
    resp = resp.reshape(resp.shape[0],-1)
    V = poly.eval(point)
    end = resp.shape[1]

    # Standardizes V and calls the lars algorithm

    V1  = V[:,1:].copy()
    coef = np.zeros((V.shape[1],end))
    stat = [np.mean(V1,axis=0),np.std(V1,axis=0,ddof=1)]
    V1 = (V1-stat[0])/stat[1]

    for i in range(end):

        coef[:,i] = angle(V1,resp[:,i],stat,it)
        index = np.argwhere(coef[:,i]!=0).flatten()
        coef[index,i] = square(V[:,index],resp[:,i],weight)
        timer(i,end,'Computing coefficients ')

    index = np.argwhere(np.any(coef,axis=1)).flatten()
    coef = coef.reshape(shape)
    
    printer(1,'Computing coefficients 100 %')
    return coef,index

# %% Expansion Coefficients with Least Absolute Shrinkage

def lasso(resp,poly,point,weight=0,it=np.inf):
    
    printer(0,'Computing coefficients ...')

    resp = np.array(resp)
    shape = (poly[:].shape[0],)+resp.shape[1:]
    resp = resp.reshape(resp.shape[0],-1)
    V = poly.eval(point)
    end = resp.shape[1]

    # Standardizes V and calls the lasso algorithm

    V1  = V[:,1:].copy()
    coef = np.zeros((V.shape[1],end))
    stat = [np.mean(V1,axis=0),np.std(V1,axis=0,ddof=1)]
    V1 = (V1-stat[0])/stat[1]

    for i in range(end):

        coef[:,i] = shrink(V1,resp[:,i],stat,it)
        index = np.argwhere(coef[:,i]!=0).flatten()
        coef[index,i] = square(V[:,index],resp[:,i],weight)
        timer(i,end,'Computing coefficients ')

    index = np.argwhere(np.any(coef,axis=1)).flatten()
    coef = coef.reshape(shape)
    
    printer(1,'Computing coefficients 100 %')
    return coef,index

# %% Expansion Coefficients with Least Squares Regression

def square(V,resp,weight):
    
    if np.any(weight):

        Vt = V.T
        v1 = Vt.dot(np.transpose(weight*Vt))
        v2 = Vt.dot(np.transpose(weight*resp.T))
        coef = np.linalg.solve(v1,v2)

    else: coef = np.linalg.lstsq(V,resp,rcond=None)[0]
    return coef

# %% Expansion Coefficients with Least Angle Regression

def angle(V,resp,stat,it):

    nbrPoly = V.shape[1]
    mean = np.mean(resp)
    coef = np.zeros(nbrPoly+1)
    
    # First variable entering the model

    r = resp-mean
    J = np.atleast_1d(np.argmax(np.abs(np.dot(V.T,r))))
    i = 0

    # Performs the lars iterations

    while (i==0 or alp<1) and (i+1<it):

        alp = 1
        d = np.zeros(nbrPoly)
        Vj = V[:,J]

        u1 = np.dot(Vj.T,Vj)
        u2 = np.dot(Vj.T,r)
        d[J] = np.linalg.solve(u1,u2)

        Vd = np.dot(V,d)
        J = np.append(J,-1)
        u1 = np.dot(V[:,J[0]],r)

        for j in range(nbrPoly):

            alp1 = 1
            if not (j in J):

                u2 = np.dot(V[:,j],Vd)
                u3 = np.dot(V[:,j],r)
                alp1 = (u1-u3)/(u1-u2)

                if not (0<alp1<1):

                    alp1 = 1
                    alp2 = (u1+u3)/(u1+u2)
                    if (0<alp2<1): alp1 = alp2

                if alp1<alp:
                    alp = alp1
                    J[-1] = j

        coef[1:] = coef[1:]+alp*d
        r -= alp*Vd
        i += 1

    # Translates coefficient back to original scale

    coef[1:] = coef[1:]/stat[1]
    coef[0] = mean-np.dot(coef[1:],stat[0])
    return coef

# %% Expansion Coefficients with Least Absolute Shrinkage

def shrink(V,resp,stat,it):

    nbrPoly = V.shape[1]
    mean = np.mean(resp)
    coef = np.zeros(nbrPoly+1)
    
    # First variable entering the model

    r = resp-mean
    J = np.atleast_1d(np.argmax(np.abs(np.dot(V.T,r))))
    i = 0

    # Performs the lasso iterations

    while (i==0 or alp<1) and (i+1<it):

        alp = 1
        d = np.zeros(nbrPoly)
        Vj = V[:,J]

        u1 = np.dot(Vj.T,Vj)
        u2 = np.dot(Vj.T,r)
        d[J] = np.linalg.solve(u1,u2)

        Vd = np.dot(V,d)
        J = np.append(J,-1)
        u1 = np.dot(V[:,J[0]],r)

        for j in range(nbrPoly):

            alp1 = 1
            if not (j in J):

                u2 = np.dot(V[:,j],Vd)
                u3 = np.dot(V[:,j],r)
                alp1 = (u1-u3)/(u1-u2)

                if not (0<alp1<1):
                             
                    alp1 = 1
                    alp2 = (u1+u3)/(u1+u2)
                    if (0<alp2<1): alp1 = alp2

                if alp1<alp:
                    alp = alp1
                    J[-1] = j
                    
            elif d[j] != -1:
                
                alp1 = -coef[j+1]/d[j]
                if (0<alp1<alp):
                    alp = alp1
                    J[-1] = -j

        if J[-1]<0: J = J[:-1]
        coef[1:] = coef[1:]+alp*d
        r -= alp*Vd
        i += 1

    # Translates coefficient back to original scale

    coef[1:] = coef[1:]/stat[1]
    coef[0] = mean-np.dot(coef[1:],stat[0])
    return coef
