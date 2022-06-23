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

def rv_model(params, data_rv, t, error_rv):
   
    #orbital parameters:
    try:
        w = params[0]*np.pi/180
        e= params[1]
        P = params[2]
        T= params[3]
        K = params[4]
        gamma = params[5]
    except:
        w = params['w']*np.pi/180
        e= params['e']
        P = params['P']
        T= params['T']
        K= params['K']
        gamma= params['gamma']

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t:
        m_anom=2*np.pi/P*(i-T)
        M.append(m_anom)
    M=np.asarray(M)

    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e)
        E.append(e_anom)
    E=np.asarray(E)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    model = K*(np.cos(w+v)+e*np.cos(w))+gamma
    return (data_rv-model)/error_rv

def rv_triple_model(params, data_rv, t, error_rv):
   
    #orbital parameters:
    try:
        w = params[0]*np.pi/180
        e= params[1]
        P = params[2]
        T= params[3]
        K = params[4]
        gamma = params[5]

        w2 = params[6]*np.pi/180
        e2 = params[7]
        P2 = params[8]
        T2 = params[9]
        K2 = params[10]
    except:
        w = params['w']*np.pi/180
        e= params['e']
        P = params['P']
        T= params['T']
        K= params['K']
        gamma= params['gamma']

        w2 = params['w2']*np.pi/180
        e2 = params['e2']
        P2 = params['P2']
        T2 = params['T2']
        K2 = params['K2']

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t:
        m_anom=2*np.pi/P*(i-T)
        M.append(m_anom)
    M=np.asarray(M)

    M2=[]
    for i in t:
        m_anom=2*np.pi/P2*(i-T2)
        M2.append(m_anom)
    M2=np.asarray(M2)

    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e)
        E.append(e_anom)
    E=np.asarray(E)

    E2=[]
    for j in M2:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E2.append(e_anom)
    E2=np.asarray(E2)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    v2 = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E2/2))
    model = K*(np.cos(w+v)+e*np.cos(w)) + K2*(np.cos(w2+v2)+e2*np.cos(w2)) + gamma
    return (data_rv-model)/error_rv

########################
#astrometry model for fitting x,y,t data with error ellipses
def binary_model_combined(params, data_x, data_y, t, error_maj, error_min, error_pa, data_rv,t_rv,error_rv):
   
   #orbital parameters:
    w = params['w']
    bigw = params['bigw']
    inc = params['inc']
    e = params['e']
    a = params['a']
    P = params['P']
    T = params['T']
    mirc_scale = params['mirc_scale']

    K = params['K']
    gamma = params['gamma']
    #mratio = params['mratio']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w-180,tau=T)
    pos = ke.xyzPos(t)
    model_x = pos[::,1]
    model_y = pos[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t_rv:
        m_anom=2*np.pi/P*(i-T)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e)
        E.append(e_anom)
    E=np.asarray(E)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    w_rv = (w)*np.pi/180
    model = K*(np.cos(w_rv+v)+e*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    resids_rv = (data_rv-model)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resids_rv])

    return (resids)

def rv_model_circular(params, data_rv, t, error_rv):
   
    #orbital parameters:
    try:
        w = params[0]*np.pi/180
        e= params[1]
        P = params[2]
        T= params[3]
        K = params[4]
        gamma = params[5]
    except:
        w = params['w']*np.pi/180
        e= params['e']
        P = params['P']
        T= params['T']
        K= params['K']
        gamma= params['gamma']

    #Calculate the mean anamoly for each t in dataset:
    E=[]
    for i in t:
        m_anom=2*np.pi/P*(i-T)
        E.append(m_anom)
    E=np.asarray(E)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    model = K*(np.cos(w+v)+e*np.cos(w))+gamma
    return (data_rv-model)/error_rv

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

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_combined(params, data_x, data_y, t, error_maj, error_min, error_pa, data_rv,t_rv,error_rv):
   
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

    K = params['K']
    gamma = params['gamma']
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

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E.append(e_anom)
    E=np.asarray(E)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E/2))
    w_rv = (w2+180)*np.pi/180
    model = K*(np.cos(w_rv+v)+e2*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    resids_rv = (data_rv-model)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resids_rv])

    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_combined2(params, data_x, data_y, t, error_maj, error_min, error_pa, data_rv,t_rv,error_rv):
   
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

    K = params['K']
    gamma = params['gamma']

    K_outer = params['K_outer']
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

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    M_outer=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        m_anom2=2*np.pi/P*(i-T)
        M.append(m_anom)
        M_outer.append(m_anom2)
    M=np.asarray(M)
    M_outer=np.asarray(M_outer)

    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E.append(e_anom)
    E=np.asarray(E)

    E_outer=[]
    for j in M_outer:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e)
        E_outer.append(e_anom)
    E_outer=np.asarray(E_outer)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E/2))
    v_outer = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E_outer/2))
    w_rv = (w2+180)*np.pi/180
    w_rv_outer = (w+180)*np.pi/180
    model = -K*(np.cos(w_rv+v)+e2*np.cos(w_rv))+gamma - K_outer*(np.cos(w_rv_outer+v_outer)+e*np.cos(w_rv_outer))
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    resids_rv = (data_rv-model)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resids_rv])

    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_combined3(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x2, data_y2, t2, error_maj2, error_min2, error_pa2, data_rv_a,t_rv_a,error_rv_a,data_rv_ab,t_rv_ab,error_rv_ab,data_rv_b,t_rv_b,error_rv_b):
   
   #orbital parameters:
    try:
        w = params[0]
        bigw = params[1]
        inc = params[2]
        e = params[3]
        a = params[4]
        P = params[5]
        T = params[6]
    
        w2 = params[7]
        bigw2 = params[8]
        inc2 = params[9]
        e2 = params[10]
        a_inner = params[11]
        a2 = params[12]
        P2 = params[13]
        T2 = params[14]
        mirc_scale = params[15]

        K_aa = params[16]
        K_ab = params[17]
        K_a = params[18]
        K_b = params[19]
        gamma = params[20]
        
    except:
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
        a_inner = params['a_inner']
        a2 = params['a2']
        P2 = params['P2']
        T2 = params['T2']
        mirc_scale = params['mirc_scale']

        K_aa = params['K_aa']
        K_ab = params['K_ab']
        K_a = params['K_a']
        K_b = params['K_b']
        gamma = params['gamma']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    ## inner orbit
    ke = pyasl.KeplerEllipse(a_inner,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t2)
    model_x_inner = pos[::,1]
    model_y_inner = pos[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale

    #Calculate the mean anamoly for each t in dataset:
    M_a=[]
    M_outer_a=[]
    for i in t_rv_a:
        m_anom=2*np.pi/P2*(i-T2)
        m_anom2=2*np.pi/P*(i-T)
        M_a.append(m_anom)
        M_outer_a.append(m_anom2)
    M_a=np.asarray(M_a)
    M_outer_a=np.asarray(M_outer_a)
    
    M_ab=[]
    M_outer_ab=[]
    for i in t_rv_ab:
        m_anom=2*np.pi/P2*(i-T2)
        m_anom2=2*np.pi/P*(i-T)
        M_ab.append(m_anom)
        M_outer_ab.append(m_anom2)
    M_ab=np.asarray(M_ab)
    M_outer_ab=np.asarray(M_outer_ab)
    
    M_b=[]
    for i in t_rv_b:
        m_anom=2*np.pi/P*(i-T)
        M_b.append(m_anom)
    M_b=np.asarray(M_b)

    #eccentric anamoly calculated for each t (using kepler function):
    E_a=[]
    for j in M_a:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E_a.append(e_anom)
    E_a=np.asarray(E_a)

    E_outer_a=[]
    for j in M_outer_a:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e)
        E_outer_a.append(e_anom)
    E_outer_a=np.asarray(E_outer_a)
    
    E_ab=[]
    for j in M_ab:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E_ab.append(e_anom)
    E_ab=np.asarray(E_ab)

    E_outer_ab=[]
    for j in M_outer_ab:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e)
        E_outer_ab.append(e_anom)
    E_outer_ab=np.asarray(E_outer_ab)
    
    E_b=[]
    for j in M_b:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e)
        E_b.append(e_anom)
    E_b=np.asarray(E_b)

    #Aa component
    v_a = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E_a/2))
    v_outer_a = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E_outer_a/2))
    w_rv = (w2+180)*np.pi/180
    w_rv_outer = (w+180)*np.pi/180
    model_aa = -K_aa*(np.cos(w_rv+v_a)+e2*np.cos(w_rv))+gamma - K_a*(np.cos(w_rv_outer+v_outer_a)+e*np.cos(w_rv_outer))
    
    #Ab component
    v_ab = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E_ab/2))
    v_outer_ab = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E_outer_ab/2))
    w_rv = (w2+180)*np.pi/180
    w_rv_outer = (w+180)*np.pi/180
    model_ab = K_ab*(np.cos(w_rv+v_ab)+e2*np.cos(w_rv))+gamma - K_a*(np.cos(w_rv_outer+v_outer_ab)+e*np.cos(w_rv_outer))
    
    #B component
    v_b = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E_b/2))
    w_rv = (w+180)*np.pi/180
    model_b = K_b*(np.cos(w_rv+v_b)+e*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    major_vector_x_inner=np.sin(error_pa2)
    major_vector_y_inner=np.cos(error_pa2)
    minor_vector_x_inner=-major_vector_y_inner
    minor_vector_y_inner=major_vector_x_inner
    resid_x_inner=data_x2-model_x_inner
    resid_y_inner=data_y2-model_y_inner
    resid_major_inner=(resid_x_inner*major_vector_x_inner+resid_y_inner*major_vector_y_inner)/error_maj2
    resid_minor_inner=(resid_x_inner*minor_vector_x_inner+resid_y_inner*minor_vector_y_inner)/error_min2
    
    resids_rv_aa = (data_rv_a-model_aa)/error_rv_a
    resids_rv_ab = (data_rv_ab-model_ab)/error_rv_ab
    resids_rv_b = (data_rv_b-model_b)/error_rv_b
    resids=np.concatenate([resid_major,resid_minor,resid_major_inner,resid_minor_inner,resids_rv_aa,resids_rv_ab,resids_rv_b])

    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_full_combined(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x2, data_y2, t2, error_maj2, error_min2, error_pa2, data_rv,t_rv,error_rv):
   
   #orbital parameters:
    try:
        w = params[0]
        bigw = params[1]
        inc = params[2]
        e = params[3]
        a = params[4]
        P = params[5]
        T = params[6]
    
        w2 = params[7]
        bigw2 = params[8]
        inc2 = params[9]
        e2 = params[10]
        a_inner = params[11]
        a2 = params[12]
        P2 = params[13]
        T2 = params[14]
        mirc_scale = params[15]

        K = params[16]
        gamma = params[17]
        
    except:
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
        a_inner = params['a_inner']
        a2 = params['a2']
        P2 = params['P2']
        T2 = params['T2']
        mirc_scale = params['mirc_scale']

        K = params['K']
        gamma = params['gamma']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    ## inner orbit
    ke = pyasl.KeplerEllipse(a_inner,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t2)
    model_x_inner = pos[::,1]
    model_y_inner = pos[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale

    #Calculate the mean anamoly for each t in dataset:
    M_a=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        M_a.append(m_anom)
    M_a=np.asarray(M_a)

    #eccentric anamoly calculated for each t (using kepler function):
    E_a=[]
    for j in M_a:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E_a.append(e_anom)
    E_a=np.asarray(E_a)

    #Aa component
    v_a = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E_a/2))
    w_rv = (w2+180)*np.pi/180
    model_aa = K*(np.cos(w_rv+v_a)+e2*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    major_vector_x_inner=np.sin(error_pa2)
    major_vector_y_inner=np.cos(error_pa2)
    minor_vector_x_inner=-major_vector_y_inner
    minor_vector_y_inner=major_vector_x_inner
    resid_x_inner=data_x2-model_x_inner
    resid_y_inner=data_y2-model_y_inner
    resid_major_inner=(resid_x_inner*major_vector_x_inner+resid_y_inner*major_vector_y_inner)/error_maj2
    resid_minor_inner=(resid_x_inner*minor_vector_x_inner+resid_y_inner*minor_vector_y_inner)/error_min2
    
    resids_rv_aa = (data_rv-model_aa)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resid_major_inner,resid_minor_inner,resids_rv_aa])

    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_full_combined_hd1976(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x2, data_y2, t2, error_maj2, error_min2, error_pa2, data_rv,t_rv,error_rv,data_rv2,t_rv2,error_rv2,data_rv3,t_rv3,error_rv3,data_rv4,t_rv4,error_rv4):
   
   #orbital parameters:
    try:
        w = params[0]
        bigw = params[1]
        inc = params[2]
        e = params[3]
        a = params[4]
        P = params[5]
        T = params[6]
    
        w2 = params[7]
        bigw2 = params[8]
        inc2 = params[9]
        e2 = params[10]
        a_inner = params[11]
        a2 = params[12]
        P2 = params[13]
        T2 = params[14]
        mirc_scale = params[15]

        K = params[16]
        gamma = params[17]
        gamma2 = params[17]
        gamma3 = params[17]
        gamma4 = params[17]
        
    except:
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
        a_inner = params['a_inner']
        a2 = params['a2']
        P2 = params['P2']
        T2 = params['T2']
        mirc_scale = params['mirc_scale']

        K = params['K']
        gamma = params['gamma']
        gamma2 = params['gamma2']
        gamma3 = params['gamma3']
        gamma4 = params['gamma4']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    ## inner orbit
    ke = pyasl.KeplerEllipse(a_inner,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t2)
    model_x_inner = pos[::,1]
    model_y_inner = pos[::,0]
    
    #idx = np.where((t<58362) & (t>57997))
    idx = np.where(t<58757)
    model_y[idx]/=mirc_scale
    model_x[idx]/=mirc_scale

    #Calculate the mean anamoly for each t in dataset:
    M_a=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        M_a.append(m_anom)
    M_a=np.asarray(M_a)

    #eccentric anamoly calculated for each t (using kepler function):
    E_a=[]
    for j in M_a:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E_a.append(e_anom)
    E_a=np.asarray(E_a)
    
    #Calculate the mean anamoly for each t in dataset:
    M_a2=[]
    for i in t_rv2:
        m_anom=2*np.pi/P2*(i-T2)
        M_a2.append(m_anom)
    M_a2=np.asarray(M_a2)

    #eccentric anamoly calculated for each t (using kepler function):
    E_a2=[]
    for j in M_a2:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E_a2.append(e_anom)
    E_a2=np.asarray(E_a2)
    
    #Calculate the mean anamoly for each t in dataset:
    M_a3=[]
    for i in t_rv3:
        m_anom=2*np.pi/P2*(i-T2)
        M_a3.append(m_anom)
    M_a3=np.asarray(M_a3)

    #eccentric anamoly calculated for each t (using kepler function):
    E_a3=[]
    for j in M_a3:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E_a3.append(e_anom)
    E_a3=np.asarray(E_a3)
    
    #Calculate the mean anamoly for each t in dataset:
    M_a4=[]
    for i in t_rv4:
        m_anom=2*np.pi/P2*(i-T2)
        M_a4.append(m_anom)
    M_a4=np.asarray(M_a4)

    #eccentric anamoly calculated for each t (using kepler function):
    E_a4=[]
    for j in M_a4:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E_a4.append(e_anom)
    E_a4=np.asarray(E_a4)

    #Aa component
    v_a = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E_a/2))
    w_rv = (w2+180)*np.pi/180
    model_aa = K*(np.cos(w_rv+v_a)+e2*np.cos(w_rv))+gamma
    
    v_a2 = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E_a2/2))
    model_aa2 = K*(np.cos(w_rv+v_a2)+e2*np.cos(w_rv))+gamma2
    
    v_a3 = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E_a3/2))
    model_aa3 = K*(np.cos(w_rv+v_a3)+e2*np.cos(w_rv))+gamma3
    
    v_a4 = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E_a4/2))
    model_aa4 = K*(np.cos(w_rv+v_a4)+e2*np.cos(w_rv))+gamma4
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    major_vector_x_inner=np.sin(error_pa2)
    major_vector_y_inner=np.cos(error_pa2)
    minor_vector_x_inner=-major_vector_y_inner
    minor_vector_y_inner=major_vector_x_inner
    resid_x_inner=data_x2-model_x_inner
    resid_y_inner=data_y2-model_y_inner
    resid_major_inner=(resid_x_inner*major_vector_x_inner+resid_y_inner*major_vector_y_inner)/error_maj2
    resid_minor_inner=(resid_x_inner*minor_vector_x_inner+resid_y_inner*minor_vector_y_inner)/error_min2
    
    resids_rv_aa = (data_rv-model_aa)/error_rv
    resids_rv_aa2 = (data_rv2-model_aa2)/error_rv2
    resids_rv_aa3 = (data_rv3-model_aa3)/error_rv3
    resids_rv_aa4 = (data_rv4-model_aa4)/error_rv4
    resids=np.concatenate([resid_major,resid_minor,resid_major_inner,resid_minor_inner,resids_rv_aa,resids_rv_aa2,resids_rv_aa3,resids_rv_aa4])

    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_full(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x2, data_y2, t2, error_maj2, error_min2, error_pa2):
   
   #orbital parameters:
    try:
        w = params[0]
        bigw = params[1]
        inc = params[2]
        e = params[3]
        a = params[4]
        P = params[5]
        T = params[6]
    
        w2 = params[7]
        bigw2 = params[8]
        inc2 = params[9]
        e2 = params[10]
        a_inner = params[11]
        a2 = params[12]
        P2 = params[13]
        T2 = params[14]
        mirc_scale = params[15]
        
    except:
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
        a_inner = params['a_inner']
        a2 = params['a2']
        P2 = params['P2']
        T2 = params['T2']
        mirc_scale = params['mirc_scale']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    ## inner orbit
    ke = pyasl.KeplerEllipse(a_inner,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t2)
    model_x_inner = pos[::,1]
    model_y_inner = pos[::,0]
    
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
    
    major_vector_x_inner=np.sin(error_pa2)
    major_vector_y_inner=np.cos(error_pa2)
    minor_vector_x_inner=-major_vector_y_inner
    minor_vector_y_inner=major_vector_x_inner
    resid_x_inner=data_x2-model_x_inner
    resid_y_inner=data_y2-model_y_inner
    resid_major_inner=(resid_x_inner*major_vector_x_inner+resid_y_inner*major_vector_y_inner)/error_maj2
    resid_minor_inner=(resid_x_inner*minor_vector_x_inner+resid_y_inner*minor_vector_y_inner)/error_min2
    
    resids=np.concatenate([resid_major,resid_minor,resid_major_inner,resid_minor_inner])

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

#astrometry model for fitting x,y,t data with error ellipses
def quad_model_combined(params, data_x, data_y, t, error_maj, error_min, error_pa, data_rv,t_rv,error_rv):
   
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
    
    K = params['K']
    gamma = params['gamma']
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
    
    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E.append(e_anom)
    E=np.asarray(E)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E/2))
    w_rv = (w2+180)*np.pi/180
    model = K*(np.cos(w_rv+v)+e2*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    resids_rv = (data_rv-model)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resids_rv])
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

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_combined_circular(params, data_x, data_y, t, error_maj, error_min, error_pa, data_rv,t_rv,error_rv):
   
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

    K = params['K']
    gamma = params['gamma']
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

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=M ## circular

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E/2))
    w_rv = (w2+180)*np.pi/180
    model = K*(np.cos(w_rv+v)+e2*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(error_pa)
    major_vector_y=np.cos(error_pa)
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=data_x-model_x
    resid_y=data_y-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/error_maj
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/error_min
    
    resids_rv = (data_rv-model)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resids_rv])

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

    model_y[:len(t)]*=mirc_scale
    model_x[:len(t)]*=mirc_scale
    
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
    
    model_y[:len(t)]*=mirc_scale
    model_x[:len(t)]*=mirc_scale
    
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

#astrometry model for fitting x,y,t data with error ellipses
def triple_model_circular_vlti(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x_vlti, data_y_vlti, t_vlti, error_maj_vlti, error_min_vlti, error_pa_vlti):
   
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
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2,ks=circular_kepler)
    pos = ke.xyzPos(t_all)
    pos2 = ke2.xyzPos(t_all)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]

    model_y[:len(t)]*=mirc_scale
    model_x[:len(t)]*=mirc_scale
    
    #idx = np.where((t<58362) & (t>57997))
    #idx = np.where(t<58757)
    #model_y[idx]/=mirc_scale
    #model_x[idx]/=mirc_scale
    
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

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_vlti_combined(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x_vlti, data_y_vlti, t_vlti, error_maj_vlti, error_min_vlti, error_pa_vlti,data_rv,t_rv,error_rv):
   
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

    K = params['K']
    gamma = params['gamma']
    #mratio = params['mratio']

    ## other method:
    t_all = np.concatenate([t,t_vlti])
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t_all)
    pos2 = ke2.xyzPos(t_all)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    model_y[:len(t)]*=mirc_scale
    model_x[:len(t)]*=mirc_scale

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E.append(e_anom)
    E=np.asarray(E)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E/2))
    w_rv = (w2+180)*np.pi/180
    model = K*(np.cos(w_rv+v)+e2*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(np.concatenate([error_pa,error_pa_vlti]))
    major_vector_y=np.cos(np.concatenate([error_pa,error_pa_vlti]))
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=np.concatenate([data_x,data_x_vlti])-model_x
    resid_y=np.concatenate([data_y,data_y_vlti])-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/np.concatenate([error_maj,error_maj_vlti])
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/np.concatenate([error_min,error_min_vlti])
    
    resids_rv = (data_rv-model)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resids_rv])

    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_full_vlti(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x_vlti, data_y_vlti, t_vlti, error_maj_vlti, error_min_vlti, error_pa_vlti,data_x_inner, data_y_inner, t_inner, error_maj_inner, error_min_inner, error_pa_inner, data_x_inner_vlti, data_y_inner_vlti, t_inner_vlti, error_maj_inner_vlti, error_min_inner_vlti, error_pa_inner_vlti):
   
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
    
    a_inner = params['a_inner']

    ## other method:
    t_all = np.concatenate([t,t_vlti])
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t_all)
    pos2 = ke2.xyzPos(t_all)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    model_y[:len(t)]*=mirc_scale
    model_x[:len(t)]*=mirc_scale
    
    ## other method:
    t_all = np.concatenate([t_inner,t_inner_vlti])
    ke = pyasl.KeplerEllipse(a_inner,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t_all)
    model_x_inner = pos[::,1]
    model_y_inner = pos[::,0]
    
    model_y_inner[:len(t_inner)]*=mirc_scale
    model_x_inner[:len(t_inner)]*=mirc_scale
    
    major_vector_x=np.sin(np.concatenate([error_pa,error_pa_vlti]))
    major_vector_y=np.cos(np.concatenate([error_pa,error_pa_vlti]))
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=np.concatenate([data_x,data_x_vlti])-model_x
    resid_y=np.concatenate([data_y,data_y_vlti])-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/np.concatenate([error_maj,error_maj_vlti])
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/np.concatenate([error_min,error_min_vlti])
    
    major_vector_x_inner=np.sin(np.concatenate([error_pa_inner,error_pa_inner_vlti]))
    major_vector_y_inner=np.cos(np.concatenate([error_pa_inner,error_pa_inner_vlti]))
    minor_vector_x_inner=-major_vector_y_inner
    minor_vector_y_inner=major_vector_x_inner
    resid_x_inner=np.concatenate([data_x_inner,data_x_inner_vlti])-model_x_inner
    resid_y_inner=np.concatenate([data_y_inner,data_y_inner_vlti])-model_y_inner
    resid_major_inner=(resid_x_inner*major_vector_x_inner+resid_y_inner*major_vector_y_inner)/np.concatenate([error_maj_inner,error_maj_inner_vlti])
    resid_minor_inner=(resid_x_inner*minor_vector_x_inner+resid_y_inner*minor_vector_y_inner)/np.concatenate([error_min_inner,error_min_inner_vlti])
    
    resids=np.concatenate([resid_major,resid_minor,resid_major_inner,resid_minor_inner])

    return (resids)

########################
#astrometry model for fitting x,y,t data with error ellipses
def triple_model_full_vlti_combined(params, data_x, data_y, t, error_maj, error_min, error_pa, data_x_vlti, data_y_vlti, t_vlti, error_maj_vlti, error_min_vlti, error_pa_vlti,data_x_inner, data_y_inner, t_inner, error_maj_inner, error_min_inner, error_pa_inner, data_x_inner_vlti, data_y_inner_vlti, t_inner_vlti, error_maj_inner_vlti, error_min_inner_vlti, error_pa_inner_vlti,data_rv,t_rv,error_rv):
   
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
    
    a_inner = params['a_inner']

    K = params['K']
    gamma = params['gamma']
    #mratio = params['mratio']

    ## other method:
    t_all = np.concatenate([t,t_vlti])
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t_all)
    pos2 = ke2.xyzPos(t_all)
    model_x = pos[::,1] + pos2[::,1]
    model_y = pos[::,0] + pos2[::,0]
    
    model_y[:len(t)]*=mirc_scale
    model_x[:len(t)]*=mirc_scale
    
    ## other method:
    t_all = np.concatenate([t_inner,t_inner_vlti])
    ke = pyasl.KeplerEllipse(a_inner,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t_all)
    model_x_inner = pos[::,1]
    model_y_inner = pos[::,0]
    
    model_y_inner[:len(t_inner)]*=mirc_scale
    model_x_inner[:len(t_inner)]*=mirc_scale

    #Calculate the mean anamoly for each t in dataset:
    M=[]
    for i in t_rv:
        m_anom=2*np.pi/P2*(i-T2)
        M.append(m_anom)
    M=np.asarray(M)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        #e_anom=keplerE(j,e)
        e_anom=ks.getE(j,e2)
        E.append(e_anom)
    E=np.asarray(E)

    #Find velocity for model, compare to v1
    v = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E/2))
    w_rv = (w2+180)*np.pi/180
    model = K*(np.cos(w_rv+v)+e2*np.cos(w_rv))+gamma
    
    major_vector_x=np.sin(np.concatenate([error_pa,error_pa_vlti]))
    major_vector_y=np.cos(np.concatenate([error_pa,error_pa_vlti]))
    minor_vector_x=-major_vector_y
    minor_vector_y=major_vector_x
    resid_x=np.concatenate([data_x,data_x_vlti])-model_x
    resid_y=np.concatenate([data_y,data_y_vlti])-model_y
    resid_major=(resid_x*major_vector_x+resid_y*major_vector_y)/np.concatenate([error_maj,error_maj_vlti])
    resid_minor=(resid_x*minor_vector_x+resid_y*minor_vector_y)/np.concatenate([error_min,error_min_vlti])
    
    major_vector_x_inner=np.sin(np.concatenate([error_pa_inner,error_pa_inner_vlti]))
    major_vector_y_inner=np.cos(np.concatenate([error_pa_inner,error_pa_inner_vlti]))
    minor_vector_x_inner=-major_vector_y_inner
    minor_vector_y_inner=major_vector_x_inner
    resid_x_inner=np.concatenate([data_x_inner,data_x_inner_vlti])-model_x_inner
    resid_y_inner=np.concatenate([data_y_inner,data_y_inner_vlti])-model_y_inner
    resid_major_inner=(resid_x_inner*major_vector_x_inner+resid_y_inner*major_vector_y_inner)/np.concatenate([error_maj_inner,error_maj_inner_vlti])
    resid_minor_inner=(resid_x_inner*minor_vector_x_inner+resid_y_inner*minor_vector_y_inner)/np.concatenate([error_min_inner,error_min_inner_vlti])
    
    resids_rv = (data_rv-model)/error_rv
    resids=np.concatenate([resid_major,resid_minor,resids_rv,resid_major_inner,resid_minor_inner])

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