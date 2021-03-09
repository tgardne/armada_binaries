######################
## functions for plotting orbits
######################

import numpy as np
from PyAstronomy import pyasl
ks=pyasl.MarkleyKESolver()

def orbit_model(a,e,inc,w,bigw,P,T,t):

    #Calculate the mean anamoly for each t in model:
    t0=np.linspace(t[0],t[0]+P,10000)
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    pos = ke.xyzPos(t0)
    ra = pos[::,1]
    dec = pos[::,0]

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    pos = ke.xyzPos(t)
    rapoints = pos[::,1]
    decpoints = pos[::,0]
    
    return(ra,dec,rapoints,decpoints)

def triple_orbit_model(a,e,inc,w,bigw,P,T,a2,e2,inc2,w2,bigw2,P2,T2,t):

    t0=np.linspace(t[0],t[0]+P,10000)
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t0)
    pos2 = ke2.xyzPos(t0)
    ra = pos[::,1] + pos2[::,1]
    dec = pos[::,0] + pos2[::,0]

    # GEt model value for each point:
    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    ke2 = pyasl.KeplerEllipse(a2,P2,e=e2,Omega=bigw2,i=inc2,w=w2,tau=T2)
    pos = ke.xyzPos(t)
    pos2 = ke2.xyzPos(t)
    rapoints = pos[::,1] + pos2[::,1]
    decpoints = pos[::,0] + pos2[::,0]
    
    return(ra,dec,rapoints,decpoints)