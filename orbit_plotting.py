######################
## functions for plotting orbits
######################

import numpy as np
from PyAstronomy import pyasl
ks=pyasl.MarkleyKESolver()

def orbit_model(a,e,inc,w,bigw,P,T,t):
    #plot results
    A=a*(np.cos(bigw)*np.cos(w)-np.sin(bigw)*np.cos(inc)*np.sin(w))
    B=a*(np.sin(bigw)*np.cos(w)+np.cos(bigw)*np.cos(inc)*np.sin(w))
    F=a*(-np.cos(bigw)*np.sin(w)-np.sin(bigw)*np.cos(inc)*np.cos(w))
    G=a*(-np.sin(bigw)*np.sin(w)+np.cos(bigw)*np.cos(inc)*np.cos(w))

    #Calculate the mean anamoly for each t in model:
    t0=np.linspace(t[0],t[0]+P,1000)
    M=[]
    for i in t0:
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
    dec=A*X+F*Y
    ra=B*X+G*Y

    # GEt model value for each point:
    Mpoints=[]
    for i in t:
        m_anom=2*np.pi/P*(i-T)
        Mpoints.append(m_anom)
    Mpoints=np.asarray(Mpoints)
    #eccentric anamoly calculated for each t (using kepler function):
    Epoints=[]
    for j in Mpoints:
        e_anom=ks.getE(j,e)
        Epoints.append(e_anom)
    Epoints=np.asarray(Epoints)
    #Find sep&pa for model, compare to data:
    Xpoints=[]
    Ypoints=[]
    for k in Epoints:
        Xk=np.cos(k)-e
        Yk=np.sqrt(1-e**2)*np.sin(k)
        Xpoints.append(Xk)
        Ypoints.append(Yk)
    Xpoints=np.asarray(Xpoints)
    Ypoints=np.asarray(Ypoints)
    decpoints=A*Xpoints+F*Ypoints
    rapoints=B*Xpoints+G*Ypoints
    
    return(ra,dec,rapoints,decpoints)