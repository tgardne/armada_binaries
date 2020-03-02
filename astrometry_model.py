##########################
## Tyler Gardner
##
## model for fitting astrometric orbits
## for least squares and mcmc
#########################

import numpy as np
from PyAstronomy import pyasl
import random
ks=pyasl.MarkleyKESolver()

########################
#astrometry model for fitting x,y,t data with error ellipses
def astrometry_model(params, data_x, data_y, t, error_maj, error_min, error_pa):
   
    #orbital parameters:
    try:
        w = params[0]
        bigw = params[1]
        inc = params[2]
        e= params[3]
        a= params[4]
        P = params[5]
        T= params[6]
        #pscale = params[7]
    except:
        w = params['w']
        bigw = params['bigw']
        inc = params['inc']
        e= params['e']
        a= params['a']
        P = params['P']
        T= params['T']
        #pscale=params['pscale']

    #Thiele-Innes elements:
    A=a*(np.cos(bigw)*np.cos(w)-np.sin(bigw)*np.cos(inc)*np.sin(w))
    B=a*(np.sin(bigw)*np.cos(w)+np.cos(bigw)*np.cos(inc)*np.sin(w))
    F=a*(-np.cos(bigw)*np.sin(w)-np.sin(bigw)*np.cos(inc)*np.cos(w))
    G=a*(-np.sin(bigw)*np.sin(w)+np.cos(bigw)*np.cos(inc)*np.cos(w))
    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t:
        m_anom=2*np.pi/P*(i-T)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        e_anom=ks.getE(j,e)
        E.append(e_anom)
    E=np.asarray(E)
    #Find sep&pa for model, compare to data:
    X=[]
    Y=[]
    for k in E:
        Xk=np.cos(k)-e
        Yk=np.sqrt(1-e**2)*np.sin(k)
        X.append(Xk)
        Y.append(Yk)
    X=np.asarray(X)
    Y=np.asarray(Y)
    
    model_y=A*X+F*Y
    model_x=B*X+G*Y
    
    #idx = np.where(t<57997)
    #model_y[idx]/=pscale
    #model_x[idx]/=pscale
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    resids=np.concatenate([resid_major,resid_minor])
    
    #resids[idx]*=10
    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model(params, data_x, data_y, t, error_maj, error_min, error_pa):
   
   #orbital parameters:
    w = params['w']
    bigw = params['bigw']
    inc = params['inc']
    e = params['e']
    a = params['a']
    P = params['P']
    T = params['T']
    
    w2 = params['w2']
    bigw2 = params['bigw2']
    inc2 = params['inc2']
    e2 = params['e2']
    a2 = params['a2']
    P2 = params['P2']
    T2 = params['T2']
    #mratio = params['mratio']
    #pscale = params['pscale']

    ###############################################################
    #Thiele-Innes elements for secondary
    A = a*(np.cos(bigw)*np.cos(w)-np.sin(bigw)*np.cos(inc)*np.sin(w))
    B = a*(np.sin(bigw)*np.cos(w)+np.cos(bigw)*np.cos(inc)*np.sin(w))
    F = a*(-np.cos(bigw)*np.sin(w)-np.sin(bigw)*np.cos(inc)*np.cos(w))
    G = a*(-np.sin(bigw)*np.sin(w)+np.cos(bigw)*np.cos(inc)*np.cos(w))
    #Calculate the mean anamoly for each t in dataset:
    M = []
    for i in t:
        m_anom=2*np.pi/P*(i-T)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        e_anom=ks.getE(j,e)
        E.append(e_anom)
    E=np.asarray(E)
    #Find sep&pa for model, compare to data:
    X=[]
    Y=[]
    for k in E:
        Xk=np.cos(k)-e
        Yk=np.sqrt(1-e**2)*np.sin(k)
        X.append(Xk)
        Y.append(Yk)
    X=np.asarray(X)
    Y=np.asarray(Y)
    ###############################################################
    
    #Thiele-Innes elements for tertiary
    A2 = a2*(np.cos(bigw2)*np.cos(w2)-np.sin(bigw2)*np.cos(inc2)*np.sin(w2))
    B2 = a2*(np.sin(bigw2)*np.cos(w2)+np.cos(bigw2)*np.cos(inc2)*np.sin(w2))
    F2 = a2*(-np.cos(bigw2)*np.sin(w2)-np.sin(bigw2)*np.cos(inc2)*np.cos(w2))
    G2 = a2*(-np.sin(bigw2)*np.sin(w2)+np.cos(bigw2)*np.cos(inc2)*np.cos(w2))
    #Calculate the mean anamoly for each t in dataset:
    M = []
    for i in t:
        m_anom=2*np.pi/P2*(i-T2)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        e_anom=ks.getE(j,e2)
        E.append(e_anom)
    E=np.asarray(E)
    #Find sep&pa for model, compare to data:
    X2=[]
    Y2=[]
    for k in E:
        Xk=np.cos(k)-e2
        Yk=np.sqrt(1-e2**2)*np.sin(k)
        X2.append(Xk)
        Y2.append(Yk)
    X2=np.asarray(X2)
    Y2=np.asarray(Y2)
    ###############################################################
    
    model_y = A*X+F*Y + A2*X2+F2*Y2
    model_x = B*X+G*Y + B2*X2+G2*Y2
    
    #idx = np.where(t<57997)
    #idx = np.where((t<58380) & (t>57997))
    #model_y[idx]/=pscale
    #model_x[idx]/=pscale
    #model_y[idx2]/=pscale2
    #model_x[idx2]/=pscale2
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    resids=np.concatenate([resid_major,resid_minor])
    return (resids)

def lnlike(params,x,y,t,emaj,emin,epa):
    '''
    The log-likelihood function. Observational model assume independent Gaussian error bars.
    '''
    model = astrometry_model(params,x,y,t,emaj,emin,epa)
    lnlike=-0.5*np.sum(model**2)
    return(lnlike)

def lnprior(params):
    '''
    The log-prior function.
    '''
    pars = [params[x:x+7] for x in range(0, len(params), 7)]
    nPars = len(pars)
    thetaT = np.transpose(pars)
    w,bigw,inc,e,a,P,T=thetaT[0],thetaT[1],thetaT[2],thetaT[3],thetaT[4],thetaT[5],thetaT[6]
            
    for i in range(nPars):
        if 0 < P[i] and 0 < T[i] and 0. <= e[i] < 1. \
        and 0. <= w[i] <= 2*np.pi and 0. <= bigw[i] <= 2*np.pi and 0. <= inc[i] <= 2*np.pi \
        and 0 < a[i]:
                lnp = 0
        else:
                lnp = -np.inf
    return lnp

def lnpost(params,x,y,t,emaj,emin,epa):
    '''
    The log-posterior function. Sum of log-likelihood and log-prior.
    '''
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params,x,y,t,emaj,emin,epa)

def create_init(theta):
    pars = [theta[x:x+7] for x in range(0, len(theta), 7)]
    init = []
    for item in pars:
        init.append(item[0] + random.uniform(-0.1,0.1))
        init.append(item[1] + random.uniform(-0.1,0.1))
        init.append(item[2] + random.uniform(-0.1,0.1))
        init.append(item[3] + random.uniform(-0.1,0.1))
        init.append(item[4] + random.uniform(-1,1))
        init.append(item[5] + random.uniform(-365,365))
        init.append(item[6] + random.uniform(-365,365))
        return init