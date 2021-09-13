######################################################################
## Tyler Gardner
##
## Pipeline to fit binary orbits
## and search for additional companions
##
## For binary orbits from MIRCX/GRAVITY
##
######################################################################

from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
import matplotlib.cm as cm
from read_data import read_data,read_wds,read_orb6
from astrometry_model import astrometry_model,triple_model,triple_model_circular,lnlike,lnprior,lnpost,create_init
from orbit_plotting import orbit_model,triple_orbit_model
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import random
import emcee
import corner
from PyAstronomy import pyasl

def cart2pol(x,y):
    x=-x
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x) * 180 / np.pi
    if theta>0 and theta<90:
        theta_new = theta+270
    if theta>90 and theta<360:
        theta_new = theta-90
    if theta<0:
        theta_new = 270+theta
    if np.isnan(theta):
        theta_new=theta
    return(r,theta_new)

def astrometry_model_mcmc(params, data_x, data_y, t, error_maj, error_min, error_pa):
   
    #orbital parameters:
    try:
        w = params[0]
        bigw = params[1]
        inc = params[2]
        e= params[3]
        a= params[4]
        P = params[5]
        T= params[6]
    except:
        w = params['w']
        bigw = params['bigw']
        inc = params['inc']
        e= params['e']
        a= params['a']
        P = params['P']
        T= params['T']

    ## other method:
    ke = pyasl.KeplerEllipse(a,P,e=e,Omega=bigw,i=inc,w=w,tau=T)
    pos = ke.xyzPos(t)
    model_x = pos[::,1]
    model_y = pos[::,0]
    
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

def lnlike(params,x,y,t,emaj,emin,epa):
    '''
    The log-likelihood function. Observational model assume independent Gaussian error bars.
    '''
    model = astrometry_model_mcmc(params,x,y,t,emaj,emin,epa)
    lnlike=-0.5*np.nansum(model**2)
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
        init.append(item[0] + random.uniform(-1,1))
        init.append(item[1] + random.uniform(-1,1))
        init.append(item[2] + random.uniform(-1,1))
        init.append(item[3] + random.uniform(-0.1,0.1))
        init.append(item[4] + random.uniform(-1,1))
        init.append(item[5] + random.uniform(-0.1,0.1))
        init.append(item[6] + random.uniform(-0.1,0.1))
        return init

###########################################
## SETUP PATHS
###########################################

if os.getcwd()[7:14] == 'tgardne':
    ## setup paths for user
    path = '/Users/tgardne/ARMADA_orbits'
    path_etalon = '/Users/tgardne/etalon_epochs/etalon_fits/etalon_factors_fit.txt'
    path_wds = '/Users/tgardne/wds_targets'
    path_orb6 = '/Users/tgardne/catalogs/orb6orbits.sql.txt'
    
elif os.getcwd()[7:19] == 'adam.scovera':
    ## Adam's path
    path = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/ARMADA_orbits'
    path_etalon = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/etalon_factors_fit.txt'
    path_wds = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/wds_targets'
    path_orb6 = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/orb6orbits.sql.txt'

###########################################
## Specify Target
###########################################
target_hd = input('Target HD #: ')
date = input('Date for savefile: ')
#target = input('Target HIP #: ')
#target_wds = input('Target WDS #: ')

emethod = input('bootstrap errors? (y/n) ')
#emethod = 'n'
mirc_scale = input('Scale OLD MIRCX data? (y/n) ')
#mirc_scale = 'n'

query = Simbad.query_objectids('HD %s'%target_hd)
for item in query:
    if 'HIP' in item[0]:
        target = item[0].split()[1]
        print('HIP %s'%target)
    if 'WDS' in item[0]:
        target_wds = item[0][5:15]
        print('WDS %s'%target_wds)

try:
    print(target_wds)
except:
    print('No WDS number queried')
    target_wds = input('Enter WDS: ')


###########################################
## Read in ARMADA data
###########################################
if emethod == 'y':
    print('reading bootstrap errors')
    file=open('%s/HD_%s_bootstrap.txt'%(path,target_hd))
else:
    print('reading chi2 errors')
    file=open('%s/HD_%s_chi2err.txt'%(path,target_hd))
weight=1

t,p,theta,error_maj,error_min,error_pa,error_deg = read_data(file,weight)
file.close()

### correct PAs based on precession (only for WDS):
coord = SkyCoord.from_name("HD %s"%target_hd,parse=True)
ra = coord.ra.value*np.pi/180
dec = coord.dec.value*np.pi/180
#theta -= (0.00557*np.sin(ra)/np.cos(dec)*((t-51544.5)/365.25))/180*np.pi

## Include only NEW data after upgrade:
#idx = np.where(t>58362)
#t = t[idx]
#p = p[idx]
#theta = theta[idx]
#error_maj = error_maj[idx]
#error_min = error_min[idx]
#error_pa = error_pa[idx]
#error_deg = error_deg[idx]


###########################################
## Apply etalon correction
###########################################
file=open(path_etalon)
mjd_etalon=[]
f_etalon=[]
for line in file.readlines():
    if line.startswith('#'):
        continue
    mjd_etalon.append(float(line.split()[0]))
    f_etalon.append(float(line.split()[1]))
file.close()
mjd_etalon=np.array(mjd_etalon)
f_etalon=np.array(f_etalon)

etalon_factor=[]
for i in t:
    idx = np.where(abs(i-mjd_etalon)==min(abs(i-mjd_etalon)))
    if min(abs(i-mjd_etalon))>0.5:
        print('Closest factor for %s is %s days away'%(i,min(abs(i-mjd_etalon))))
    f = f_etalon[idx][0]
    etalon_factor.append(f)
etalon_factor=np.array(etalon_factor)

print('   date      etalon factor')
for i,j in zip(t,etalon_factor):
    print(i,j)

## apply etalon correction
etalon = input('Apply etalon correction? (y/n) ')

## FIXME: make it easier to choose vlti data
#vlti = input('Add indices for vlti (y/n)? ')
vlti = 'n'
if vlti=='y':
    vlti_idx = input('enter indices (e.g. 1 2 3): ').split(' ')
    vlti_idx = np.array([int(i) for i in vlti_idx])
else:
    vlti_idx = np.array([])

if etalon=='y':
    print('Applying etalon correction')
    if len(vlti_idx)>0:
        print('Setting VLTI correction factors to 1.0')
        etalon_factor[vlti_idx] = 1.0
    p = p/etalon_factor
else:
    print('No etalon correction applied')
xpos=p*np.sin(theta)
ypos=p*np.cos(theta)


###########################################
## Read in WDS data - and plot to check
###########################################
input_wds = input('Include WDS? (y/n): ')
if input_wds == 'y':
    try:
        file=open(os.path.expanduser("%s/wds%s.txt"%(path_wds,target_wds)))
        weight = 10
        dtype = input('dtype for wds (e.g. S, leave blank for ALL data): ')

        t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds = read_wds(file,weight,dtype)
        print('Number of WDS data points = %s'%len(p_wds))

        ## correct WDS for PA
        theta_wds -= (0.00557*np.sin(ra)/np.cos(dec)*((t_wds-51544.5)/365.25))/180*np.pi

        xpos_wds=p_wds*np.sin(theta_wds)
        ypos_wds=p_wds*np.cos(theta_wds)

        plt.plot(xpos_wds,ypos_wds,'o',label='WDS')
        plt.plot(xpos_wds[0],ypos_wds[0],'*')
        try:
            idx = np.argmin(t)
            plt.plot(xpos[idx],ypos[idx],'*')
            plt.plot(xpos,ypos,'+',label='ARMADA')
        except:
            pass
        plt.plot(0,0,'*')
        plt.gca().invert_xaxis()
        plt.title('All Data')
        plt.xlabel('dra (mas)')
        plt.ylabel('ddec (mas)')
        plt.legend()
        plt.show()

        flip = input('Flip WDS data? (y/n): ')
        if flip=='y':
            xpos_wds=-p_wds*np.sin(theta_wds)
            ypos_wds=-p_wds*np.cos(theta_wds)
            plt.plot(xpos_wds,ypos_wds,'o',label='WDS')
            plt.plot(xpos_wds[0],ypos_wds[0],'*')
            try:
                plt.plot(xpos[idx],ypos[idx],'*')
                plt.plot(xpos,ypos,'+',label='ARMADA')
            except:
                pass
            plt.plot(0,0,'*')
            plt.gca().invert_xaxis()
            plt.title('All Data')
            plt.xlabel('dra (mas)')
            plt.ylabel('ddec (mas)')
            plt.legend()
            plt.show()

            better = input('Flip data back to original? (y/n): ')
            if better=='y':
                xpos_wds=p_wds*np.sin(theta_wds)
                ypos_wds=p_wds*np.cos(theta_wds)
    except:
        t_wds = np.array([np.nan])
        p_wds = np.array([np.nan])
        theta_wds = np.array([np.nan])
        error_maj_wds = np.array([np.nan])
        error_min_wds = np.array([np.nan])
        error_pa_wds = np.array([np.nan])
        error_deg_wds = np.array([np.nan])
        xpos_wds=p_wds*np.sin(theta_wds)
        ypos_wds=p_wds*np.cos(theta_wds)
        print('NO WDS NUMBER')
else:
    t_wds = np.array([np.nan])
    p_wds = np.array([np.nan])
    theta_wds = np.array([np.nan])
    error_maj_wds = np.array([np.nan])
    error_min_wds = np.array([np.nan])
    error_pa_wds = np.array([np.nan])
    error_deg_wds = np.array([np.nan])
    xpos_wds=p_wds*np.sin(theta_wds)
    ypos_wds=p_wds*np.cos(theta_wds)
    print('NO WDS DATA')

###########################################
## Get an estimate of the orbital parameters
###########################################
try:
    a,P,e,inc,omega,bigomega,T = read_orb6(target,path_orb6)
except:
    print('No elements found in ORB6. Will need to enter your own.')

self_params = input('Input own params? (y/n)')
if self_params=='y':
    a = float(input('a (mas): '))
    P = float(input('P (year): '))*365.25
    e = float(input('ecc : '))
    inc = float(input('inc (deg): '))
    omega = float(input('omega (deg): '))
    bigomega = float(input('bigomega (deg): '))
    T = float(input('T (mjd): '))

###########################################
## Combined WDS+ARMADA for fitting
###########################################
xpos_all = np.concatenate([xpos,xpos_wds])
ypos_all = np.concatenate([ypos,ypos_wds])
t_all = np.concatenate([t,t_wds])
error_maj_all = np.concatenate([error_maj,error_maj_wds])
error_min_all = np.concatenate([error_min,error_min_wds])
error_pa_all = np.concatenate([error_pa,error_pa_wds])
error_deg_all = np.concatenate([error_deg,error_deg_wds])

##########################################
## Function for fitting/plotting data
#########################################
def ls_fit(params,xp,yp,tp,emaj,emin,epa):
    #do fit, minimizer uses LM for least square fitting of model to data
    minner = Minimizer(astrometry_model, params, fcn_args=(xp,yp,tp,
                                                       emaj,emin,epa),
                            nan_policy='omit')
    result = minner.minimize()
    # write error report
    print(report_fit(result))

    ## plot fit
    a_start = result.params['a']
    P_start = result.params['P']
    e_start = result.params['e']
    inc_start = result.params['inc']
    w_start = result.params['w']
    bigw_start = result.params['bigw']
    T_start = result.params['T']

    ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                            w_start,bigw_start,P_start,
                                            T_start,t_all)
    fig,ax=plt.subplots()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('HD%s Outer Orbit'%target_hd)
    plt.legend()
    plt.show()

    return result

###########################################
## Do a least-squares fit
###########################################
guess_params = input('Randomize outer elements? (y/n)')
if guess_params=='y':

    astart = float(input('semi start (mas): '))
    aend = float(input('semi end (mas): '))
    Pstart = float(input('P start (year): '))*365
    Pend = float(input('P end (year): '))*365

    w_results = []
    bigw_results = []
    inc_results = []
    e_results = []
    a_results = []
    P_results = []
    T_results = []
    chi2_results = []

    for s in tqdm(np.arange(100)):
        x1 = random.uniform(0,360)
        x2 = random.uniform(0,360)
        x3 = random.uniform(0,180)
        x4 = random.uniform(0,0.9)
        x5 = random.uniform(astart,aend)
        x6 = random.uniform(Pstart,Pend)
        x7 = random.uniform(30000,80000)

        params = Parameters()
        params.add('w',   value= x1, min=0, max=360)
        params.add('bigw', value= x2, min=0, max=360)
        params.add('inc', value= x3, min=0, max=180)
        params.add('e', value= x4, min=0, max=0.99)
        params.add('a', value= x5, min=0)
        params.add('P', value= x6, min=0)
        params.add('T', value= x7, min=0)
        if mirc_scale == 'y':
            params.add('mirc_scale', value= 1.0,vary=False)
        else:
            params.add('mirc_scale', value= 1.0, vary=False)

        #do fit, minimizer uses LM for least square fitting of model to data
        minner = Minimizer(astrometry_model, params, fcn_args=(xpos_all,ypos_all,t_all,
                                error_maj_all,error_min_all,error_pa_all),
                                nan_policy='omit')
        result = minner.minimize()

        w_results.append(result.params['w'].value)
        bigw_results.append(result.params['bigw'].value)
        inc_results.append(result.params['inc'].value)
        e_results.append(result.params['e'].value)
        a_results.append(result.params['a'].value)
        P_results.append(result.params['P'].value)
        T_results.append(result.params['T'].value)
        chi2_results.append(result.redchi)

    w_results = np.array(w_results)
    bigw_results = np.array(bigw_results)
    inc_results = np.array(inc_results)
    e_results = np.array(e_results)
    a_results = np.array(a_results)
    P_results = np.array(P_results)
    T_results = np.array(T_results)
    chi2_results = np.array(chi2_results)

    idx = np.argmin(chi2_results)
    omega = w_results[idx]
    bigomega = bigw_results[idx]
    inc = inc_results[idx]
    e = e_results[idx]
    a = a_results[idx]
    P = P_results[idx]
    T = T_results[idx]

    print('P, a, e, inc, w, bigw, T: ')
    print(P/365, a, e, inc, omega, bigomega, T)

params = Parameters()
params.add('w',   value= omega, min=0, max=360)
params.add('bigw', value= bigomega, min=0, max=360)
params.add('inc', value= inc, min=0, max=180)
params.add('e', value= e, min=0, max=0.99)
params.add('a', value= a, min=0)
params.add('P', value= P, min=0)
params.add('T', value= T, min=0)
if mirc_scale == 'y':
    params.add('mirc_scale', value= 1.0,vary=False)
else:
    params.add('mirc_scale', value= 1.0, vary=False)

result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all)

#############################################
## Filter through bad WDS points
#############################################
def on_click_remove(event):
    bad_x = event.xdata
    bad_y = event.ydata
    diff = np.sqrt((xpos_all-bad_x)**2+(ypos_all-bad_y)**2)
    idx = np.nanargmin(diff)
    xpos_all[idx] = np.nan
    ypos_all[idx] = np.nan

    ax.cla()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to REMOVE points')
    plt.legend()
    plt.draw()

def on_click_flip(event):
    bad_x = event.xdata
    bad_y = event.ydata
    diff = np.sqrt((xpos_all-bad_x)**2+(ypos_all-bad_y)**2)
    idx = np.nanargmin(diff)
    xpos_all[idx] = -xpos_all[idx]
    ypos_all[idx] = -ypos_all[idx]

    ax.cla()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to FLIP points')
    plt.legend()
    plt.draw()

filter_wds = input('Remove/flip any WDS data? (y/n)')
while filter_wds == 'y':
    a_start = result.params['a']
    P_start = result.params['P']
    e_start = result.params['e']
    inc_start = result.params['inc']
    w_start = result.params['w']
    bigw_start = result.params['bigw']
    T_start = result.params['T']
    ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                        w_start,bigw_start,P_start,
                                        T_start,t_all)
    fig,ax=plt.subplots()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to FLIP points')
    plt.legend()
    cid = fig.canvas.mpl_connect('button_press_event', on_click_flip)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close()

    fig,ax=plt.subplots()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to REMOVE points')
    plt.legend()
    cid = fig.canvas.mpl_connect('button_press_event', on_click_remove)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close()


    params = Parameters()
    params.add('w',   value= omega, min=0, max=360)
    params.add('bigw', value= bigomega, min=0, max=360)
    params.add('inc', value= inc, min=0, max=180)
    params.add('e', value= e, min=0, max=0.99)
    params.add('a', value= a, min=0)
    params.add('P', value= P, min=0)
    params.add('T', value= T, min=0)
    if mirc_scale == 'y':
        params.add('mirc_scale', value= 1.0, vary=False)
    else:
        params.add('mirc_scale', value= 1.0, vary=False)

    result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all)
    filter_wds = input('Remove more data? (y/n)')

##########################################
## Save Plots
##########################################
resids_armada = astrometry_model(result.params,xpos,ypos,t,error_maj,
                            error_min,error_pa)
resids_wds = astrometry_model(result.params,xpos_all[len(xpos):],ypos_all[len(xpos):],t_all[len(xpos):],
                            error_maj_all[len(xpos):],error_min_all[len(xpos):],error_pa_all[len(xpos):])
ndata_armada = 2*sum(~np.isnan(xpos))
ndata_wds = 2*sum(~np.isnan(xpos_all[len(xpos):]))
chi2_armada = np.nansum(resids_armada**2)/(ndata_armada-len(result.params))
chi2_wds = np.nansum(resids_wds**2)/(ndata_wds-len(result.params))
print('-'*10)
print('chi2 armada = %s'%chi2_armada)
print('chi2 WDS = %s'%chi2_wds)
print('-'*10)

rescale = input('Rescale errors based off chi2? (y/n): ')
while rescale=='y':

    ## we don't want to raise armada errors
    if chi2_armada>1:
        chi2_armada=1.0

    error_maj*=np.sqrt(chi2_armada)
    error_min*=np.sqrt(chi2_armada)

    error_maj_all[:len(xpos)]*=np.sqrt(chi2_armada)
    error_maj_all[len(xpos):]*=np.sqrt(chi2_wds)
    error_min_all[:len(xpos)]*=np.sqrt(chi2_armada)
    error_min_all[len(xpos):]*=np.sqrt(chi2_wds)

    ###########################################
    ## Do a least-squares fit
    ###########################################
    params = Parameters()
    params.add('w',   value= omega, min=0, max=360)
    params.add('bigw', value= bigomega, min=0, max=360)
    params.add('inc', value= inc, min=0, max=180)
    params.add('e', value= e, min=0, max=0.99)
    params.add('a', value= a, min=0)
    params.add('P', value= P, min=0)
    params.add('T', value= T, min=0)
    if mirc_scale == 'y':
        params.add('mirc_scale', value= 1.0,vary=False)
    else:
        params.add('mirc_scale', value= 1.0, vary=False)

    result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all)

    resids_armada = astrometry_model(result.params,xpos,ypos,t,error_maj,
                                error_min,error_pa)
    resids_wds = astrometry_model(result.params,xpos_all[len(xpos):],ypos_all[len(xpos):],t_all[len(xpos):],
                                error_maj_all[len(xpos):],error_min_all[len(xpos):],error_pa_all[len(xpos):])
    ndata_armada = 2*sum(~np.isnan(xpos))
    ndata_wds = 2*sum(~np.isnan(xpos_all[len(xpos):]))
    chi2_armada = np.nansum(resids_armada**2)/(ndata_armada-len(result.params))
    chi2_wds = np.nansum(resids_wds**2)/(ndata_wds-len(result.params))
    print('-'*10)
    print('chi2 armada = %s'%chi2_armada)
    print('chi2 WDS = %s'%chi2_wds)
    print('-'*10)
    rescale = input('Rescale errors based off chi2? (y/n): ')

if emethod == 'y':
    directory='%s/HD%s_bootstrap_mcmc/'%(path,target_hd)
else:
    directory='%s/HD%s_chi2err_mcmc/'%(path,target_hd)
if not os.path.exists(directory):
    os.makedirs(directory)

### plot fit
#if len(vlti_idx)>0:
#    xpos[vlti_idx]*=result.params['pscale']
#    ypos[vlti_idx]*=result.params['pscale']
#    xpos_all[vlti_idx]*=result.params['pscale']
#    ypos_all[vlti_idx]*=result.params['pscale']

scale=1
if chi2_armada<1.0:
    scale=1/np.sqrt(chi2_armada)
a_start = result.params['a']
P_start = result.params['P']
e_start = result.params['e']
inc_start = result.params['inc']
w_start = result.params['w']
bigw_start = result.params['bigw']
T_start = result.params['T']
chi2_best_binary = result.chisqr
ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                        w_start,bigw_start,P_start,
                                        T_start,t_all)
fig,ax=plt.subplots()
ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
ax.plot(xpos,ypos,'o', label='ARMADA')
ax.plot(0,0,'*')
ax.plot(ra, dec, '--',color='g')
#plot lines from data to best fit orbit
i=0
while i<len(decpoints):
    x=[xpos_all[i],rapoints[i]]
    y=[ypos_all[i],decpoints[i]]
    ax.plot(x,y,color="black")
    i+=1
ax.set_xlabel('milli-arcsec')
ax.set_ylabel('milli-arcsec')
ax.invert_xaxis()
ax.axis('equal')
ax.set_title('HD%s Outer Orbit'%target_hd)
plt.legend()
plt.savefig('%s/HD%s_%s_outer_leastsquares.pdf'%(directory,target_hd,date))
plt.close()

## plot resids for ARMADA
fig,ax=plt.subplots()
xresid = xpos - rapoints[:len(xpos)]
yresid = ypos - decpoints[:len(ypos)]

#need to measure error ellipse angle east of north
for ras, decs, w, h, angle, d in zip(xresid,yresid,error_maj/scale,error_min/scale,error_deg,t):
    ellipse = Ellipse(xy=(ras, decs), width=2*w, height=2*h, 
                      angle=90-angle, facecolor='none', edgecolor='black')
    ax.annotate(d,xy=(ras,decs))
    ax.add_patch(ellipse)

ax.plot(xresid, yresid, 'o')
ax.plot(0,0,'*')
ax.set_xlabel('milli-arcsec')
ax.set_ylabel('milli-arcsec')
ax.invert_xaxis()
ax.axis('equal')
ax.set_title('HD%s Resids'%target_hd)
plt.savefig('%s/HD%s_%s_resid_leastsquares.pdf'%(directory,target_hd,date))
plt.close()

## residuals
resids = np.sqrt(xresid**2 + yresid**2)
resids_median = np.around(np.median(resids)*1000,2)
print('-'*10)
print('Median residual = %s micro-as'%resids_median)
print('-'*10)

## Run mcmc
initial_params = [w_start.value,bigw_start.value,inc_start.value,
                e_start.value,a_start.value,P_start.value,T_start.value]
nDim = len(initial_params)
nWalkers = 2*nDim
init = [create_init(initial_params) for i in range(nWalkers)]
sampler = emcee.EnsembleSampler(nWalkers, nDim, lnpost,args=[xpos_all,ypos_all,t_all,error_maj_all,
                                                            error_min_all,error_pa_all])
sampler.run_mcmc(init, 10000, progress=True)

fig, axes = plt.subplots(7, figsize=(10, 14), sharex=True)
samples = sampler.get_chain()
labels=["$\omega$ (deg)","$\Omega$ (deg)","i (deg)","e","a (mas)","P (d)","T (MJD)"]
for i in range(nDim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");
plt.savefig('%s/HD%s_chains.pdf'%(directory,target_hd))
plt.close()

try:
    tau = sampler.get_autocorr_time()
    print('tau = ')
    print(tau)
    print(10*'-')

    burn = int(3*np.mean(tau))
    print('burn = ')
    print(burn)
    print(10*'-')

    thin = int(np.mean(tau)/2)
    print('thin = ')
    print(thin)
    print(10*'-')
except:
    print('did not run long enough!!!')
    burn = 100
    thin = 10

flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
print(flat_samples.shape)
w_mcmc,bigw_mcmc,inc_mcmc,e_mcmc,a_mcmc,P_mcmc,T_mcmc = [np.std(flat_samples[:,0]),np.std(flat_samples[:,1]),
                                                        np.std(flat_samples[:,2]),np.std(flat_samples[:,3]),
                                                        np.std(flat_samples[:,4]),np.std(flat_samples[:,5]),
                                                        np.std(flat_samples[:,6])]


fig = corner.corner(flat_samples, labels=labels,truths=initial_params,label_kwargs={"fontsize":15})
plt.savefig('%s/HD%s_corner.pdf'%(directory,target_hd))
plt.close()

## Save txt file with best orbit
f = open("%s/%s_%s_orbit_mcmc.txt"%(directory,target_hd,date),"w+")
f.write("# P(d) a(mas) e i(deg) w(deg) W(deg) T(mjd) mean_resid(mu-as)\r\n")
f.write("# Perr(d) aerr(mas) eerr ierr(deg) werr(deg) Werr(deg) Terr(mjd)\r\n")
f.write("%s %s %s %s %s %s %s %s\r\n"%(P_start.value,a_start.value,e_start.value,
                                   inc_start.value,w_start.value,
                                   bigw_start.value,T_start.value,
                                  resids_median))
try:
    f.write("%s %s %s %s %s %s %s"%(P_mcmc,a_mcmc,e_mcmc,
                                       inc_mcmc,w_mcmc,
                                       bigw_mcmc,T_mcmc))
except:
    f.write("Errors not estimated")
f.close()

## Save txt file for paper
f = open("%s/%s_%s_orbit_paper.txt"%(directory,target_hd,date),"w+")
f.write("# P(d) T(mjd) e w(deg) W(deg) i(deg) a(mas) med_resid(mu-as)\r\n")
f.write("$%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s$\r\n"%(P_start.value,
                                P_mcmc,T_start.value,T_mcmc,e_start.value,e_mcmc,w_start.value,
                                w_mcmc,bigw_start.value,bigw_mcmc,inc_start.value,inc_mcmc,
                                a_start.value,a_mcmc,resids_median))
f.close()

## Save txt file with wds orbit
p_wds_new=[]
theta_wds_new = []
for i,j in zip(xpos_all[len(xpos):],ypos_all[len(xpos):]):
    pnew,tnew = cart2pol(i,j)
    p_wds_new.append(pnew)
    theta_wds_new.append(tnew)
p_wds_new = np.array(p_wds_new)
theta_wds_new = np.array(theta_wds_new)
f = open("%s/HD_%s_wds.txt"%(directory,target_hd),"w+")
f.write("# date mjd sep pa err_maj err_min err_pa\r\n")
for i,j,k,l,m,n in zip(t_wds,p_wds_new,theta_wds_new,error_maj_all[len(xpos):],error_min_all[len(xpos):],error_deg_all[len(xpos):]):
    f.write("-- %s %s %s %s %s %s\r\n"%(i,j,k,l,m,n))
f.write('#')
f.close()