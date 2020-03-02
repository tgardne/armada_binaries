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

def triple_orbit_model(a,e,inc,w,bigw,P,T,a2,e2,inc2,w2,bigw2,P2,T2,t):
    #plot results
    A=a*(np.cos(bigw)*np.cos(w)-np.sin(bigw)*np.cos(inc)*np.sin(w))
    B=a*(np.sin(bigw)*np.cos(w)+np.cos(bigw)*np.cos(inc)*np.sin(w))
    F=a*(-np.cos(bigw)*np.sin(w)-np.sin(bigw)*np.cos(inc)*np.cos(w))
    G=a*(-np.sin(bigw)*np.sin(w)+np.cos(bigw)*np.cos(inc)*np.cos(w))
    
    A2=a2*(np.cos(bigw2)*np.cos(w2)-np.sin(bigw2)*np.cos(inc2)*np.sin(w2))
    B2=a2*(np.sin(bigw2)*np.cos(w2)+np.cos(bigw2)*np.cos(inc2)*np.sin(w2))
    F2=a2*(-np.cos(bigw2)*np.sin(w2)-np.sin(bigw2)*np.cos(inc2)*np.cos(w2))
    G2=a2*(-np.sin(bigw2)*np.sin(w2)+np.cos(bigw2)*np.cos(inc2)*np.cos(w2))

    #Calculate the mean anamoly for each t in model:
    t0=np.linspace(t[0],t[0]+P,1000)
    M=[]
    for i in t0:
        m_anom=2*np.pi/P*(i-T)
        M.append(m_anom)
    M=np.asarray(M)
    M2=[]
    for i in t0:
        m_anom=2*np.pi/P2*(i-T2)
        M2.append(m_anom)
    M2=np.asarray(M2)
    #eccentric anamoly calculated for each t (using kepler function):
    E=[]
    for j in M:
        e_anom=ks.getE(j,e)
        E.append(e_anom)
    E=np.asarray(E)
    E2=[]
    for j in M2:
        e_anom=ks.getE(j,e2)
        E2.append(e_anom)
    E2=np.asarray(E2)
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
    X2=[]
    Y2=[]
    for k in E2:
        Xk=np.cos(k)-e2
        Yk=np.sqrt(1-e2**2)*np.sin(k)
        X2.append(Xk)
        Y2.append(Yk)
    X2=np.asarray(X2)
    Y2=np.asarray(Y2)
    dec=A*X+F*Y + A2*X2+F2*Y2
    ra=B*X+G*Y + B2*X2+G2*Y2

    # GEt model value for each point:
    Mpoints=[]
    for i in t:
        m_anom=2*np.pi/P*(i-T)
        Mpoints.append(m_anom)
    Mpoints=np.asarray(Mpoints)
    Mpoints2=[]
    for i in t:
        m_anom=2*np.pi/P2*(i-T2)
        Mpoints2.append(m_anom)
    Mpoints2=np.asarray(Mpoints2)
    #eccentric anamoly calculated for each t (using kepler function):
    Epoints=[]
    for j in Mpoints:
        e_anom=ks.getE(j,e)
        Epoints.append(e_anom)
    Epoints=np.asarray(Epoints)
    Epoints2=[]
    for j in Mpoints2:
        e_anom=ks.getE(j,e2)
        Epoints2.append(e_anom)
    Epoints2=np.asarray(Epoints2)
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
    Xpoints2=[]
    Ypoints2=[]
    for k in Epoints2:
        Xk=np.cos(k)-e2
        Yk=np.sqrt(1-e2**2)*np.sin(k)
        Xpoints2.append(Xk)
        Ypoints2.append(Yk)
    Xpoints2=np.asarray(Xpoints2)
    Ypoints2=np.asarray(Ypoints2)
    
    decpoints=A*Xpoints+F*Ypoints + A2*Xpoints2+F2*Ypoints2
    rapoints=B*Xpoints+G*Ypoints + B2*Xpoints2+G2*Ypoints2
    
    return(ra,dec,rapoints,decpoints)