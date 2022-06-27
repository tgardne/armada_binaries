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
class circular_kepler:
    def getE(self,M,e):
        return M

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
        mirc_scale = params[7]
    except:
        w = params['w']
        bigw = params['bigw']
        inc = params['inc']
        e= params['e']
        a= params['a']
        P = params['P']
        T= params['T']
        mirc_scale= params['mirc_scale']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    pos = ke.xyzPos(t)
    model_x = pos[::,1]
    model_y = pos[::,0]

    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale
    
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
    mirc_scale = params['mirc_scale']
    #mratio = params['mratio']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale
    
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

#astrometry model for fitting x,y,t data with error ellipses
def quad_model(params, data_x, data_y, t, error_maj, error_min, error_pa):
   
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

    w3 = params['w3']
    bigw3 = params['bigw3']
    inc3 = params['inc3']
    e3 = params['e3']
    a3 = params['a3']
    P3 = params['P3']
    T3 = params['T3']
    mirc_scale = params['mirc_scale']
    #mratio = params['mratio']

    ## other method:
    #print('starting kepler solver')
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    ke3 = pyasl.KeplerEllipse(a3,P3,e=e3,Omega=bigw3,i=inc3,w=w3,tau=T3)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    pos3 = ke3.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1] + pos3[::,1]
    model_y = pos[::,0] + pos2[::,0] + pos3[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale
    
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

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_circular(params, data_x, data_y, t, error_maj, error_min, error_pa):
   
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
    mirc_scale = params['mirc_scale']
    #mratio = params['mratio']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2,ks=circular_kepler)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale
    
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

#astrometry model for fitting x,y,t data with error ellipses
def quad_model_circular(params, data_x, data_y, t, error_maj, error_min, error_pa):
   
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

    w3 = params['w3']
    bigw3 = params['bigw3']
    inc3 = params['inc3']
    e3 = params['e3']
    a3 = params['a3']
    P3 = params['P3']
    T3 = params['T3']
    mirc_scale = params['mirc_scale']
    #mratio = params['mratio']

    ## other method:
    #print('starting kepler solver')
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2,ks=circular_kepler)
    ke3 = pyasl.KeplerEllipse(a3,P3,e=e3,Omega=bigw3,i=inc3,w=w3,tau=T3,ks=circular_kepler)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    pos3 = ke3.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1] + pos3[::,1]
    model_y = pos[::,0] + pos2[::,0] + pos3[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale
    
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

########################
#astrometry model for fitting x,y,t data with error ellipses
def astrometry_model_vlti(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x_vlti, data_y_vlti, t_vlti, error_maj_vlti, error_min_vlti, error_pa_vlti):
   
    #orbital parameters:
    try:
        w = params[0]
        bigw = params[1]
        inc = params[2]
        e= params[3]
        a= params[4]
        P = params[5]
        T= params[6]
        mirc_scale = params[7]
    except:
        w = params['w']
        bigw = params['bigw']
        inc = params['inc']
        e= params['e']
        a= params['a']
        P = params['P']
        T= params['T']
        mirc_scale= params['mirc_scale']

    ## other method:
    t_all = np.concatenate([t,t_vlti])
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    pos = ke.xyzPos(t_all)
    model_x = pos[::,1]
    model_y = pos[::,0]

    model_y[len(t):]/=mirc_scale
    model_x[len(t):]/=mirc_scale
    
    major_vector_x=np.sin(np.concatenate([error_pa,error_pa_vlti]))
    major_vector_y=np.cos(np.concatenate([error_pa,error_pa_vlti]))
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=np.concatenate([data_x,data_x_vlti])-model_x
    resid_y=np.concatenate([data_y,data_y_vlti])-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/np.concatenate([error_maj,error_maj_vlti])
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/np.concatenate([error_min,error_min_vlti])
    resids=np.concatenate([resid_major,resid_minor])
    
    #resids[idx]*=10
    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_vlti(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x_vlti, data_y_vlti, t_vlti, error_maj_vlti, error_min_vlti, error_pa_vlti):
   
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
    mirc_scale = params['mirc_scale']
    #mratio = params['mratio']

    ## other method:
    t_all = np.concatenate([t,t_vlti])
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t_all)
    pos2 = ke2.xyzPos(t_all)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    model_y[len(t):]/=mirc_scale
    model_x[len(t):]/=mirc_scale
    
    major_vector_x=np.sin(np.concatenate([error_pa,error_pa_vlti]))
    major_vector_y=np.cos(np.concatenate([error_pa,error_pa_vlti]))
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=np.concatenate([data_x,data_x_vlti])-model_x
    resid_y=np.concatenate([data_y,data_y_vlti])-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/np.concatenate([error_maj,error_maj_vlti])
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/np.concatenate([error_min,error_min_vlti])
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
        and 0. <= w[i] <= 360 and 0. <= bigw[i] <= 360 and 0. <= inc[i] <= 180 \
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