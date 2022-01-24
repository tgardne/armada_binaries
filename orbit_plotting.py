######################
## functions for plotting orbits
######################

from re import L
import numpy as np
from PyAstronomy import pyasl
ks=pyasl.MarkleyKESolver()

def orbit_model(a,e,inc,w,bigw,P,T,tepoch,tmodel='None'):
    if tmodel=='None':
        tmodel=np.linspace(tepoch[0],tepoch[0]+P,1000)
    #Calculate the mean anamoly for each t in model:
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    pos = ke.xyzPos(tmodel)
    ra = pos[::,1]
    dec = pos[::,0]

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    pos = ke.xyzPos(tepoch)
    rapoints = pos[::,1]
    decpoints = pos[::,0]
    
    return(ra,dec,rapoints,decpoints)

def triple_orbit_model(a,e,inc,w,bigw,P,T,a2,e2,inc2,w2,bigw2,P2,T2,tepoch,tmodel='None'):
    if tmodel=='None':
        tmodel=np.linspace(tepoch[0],tepoch[0]+P,1000)
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(tmodel)
    pos2 = ke2.xyzPos(tmodel)
    ra = pos[::,1] + pos2[::,1]
    dec = pos[::,0] + pos2[::,0]

    # GEt model value for each point:
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(tepoch)
    pos2 = ke2.xyzPos(tepoch)
    rapoints = pos[::,1] + pos2[::,1]
    decpoints = pos[::,0] + pos2[::,0]
    
    return(ra,dec,rapoints,decpoints)

def quad_orbit_model(a,e,inc,w,bigw,P,T,a2,e2,inc2,w2,bigw2,P2,T2,a3,e3,inc3,w3,bigw3,P3,T3,t):

    t0=np.linspace(t[0],t[0]+P,1000)
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    ke3 = pyasl.KeplerEllipse(a3,P3,e=e3,Omega=bigw3,i=inc3,w=w3,tau=T3)
    pos = ke.xyzPos(t0)
    pos2 = ke2.xyzPos(t0)
    pos3 = ke3.xyzPos(t0)
    ra = pos[::,1] + pos2[::,1] + pos3[::,1]
    dec = pos[::,0] + pos2[::,0] + pos3[::,0]

    # GEt model value for each point:
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    ke3 = pyasl.KeplerEllipse(a3,P3,e=e3,Omega=bigw3,i=inc3,w=w3,tau=T3)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    pos3 = ke3.xyzPos(t)
    rapoints = pos[::,1] + pos2[::,1] + pos3[::,1]
    decpoints = pos[::,0] + pos2[::,0] + pos3[::,0]
    
    return(ra,dec,rapoints,decpoints)