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
from astrometry_model import astrometry_model,astrometry_model_vlti,triple_model,triple_model_vlti,triple_model_circular,triple_model_circular_vlti,lnlike,lnprior,lnpost,create_init
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

def synthetic_data(kepler_params,obstime,err_maj,err_min,err_pa):
    maj_random = np.random.normal(0,err_maj)
    min_random = np.random.normal(0,err_min)
    x_random = (min_random * np.cos(err_pa)) - (maj_random * np.sin(err_pa))
    y_random = (min_random * np.sin(err_pa)) + (maj_random * np.cos(err_pa))

    semi = kepler_params[0]
    period = kepler_params[1]
    ecc = kepler_params[2]
    bigomega = kepler_params[3]
    inc = kepler_params[4]
    omega = kepler_params[5]
    t0 = kepler_params[6]

    ke = pyasl.KeplerEllipse(semi,period,e=ecc,Omega=bigomega,i=inc,w=omega,tau=t0)
    pos = ke.xyzPos(obstime)
    x_synthetic = pos[::,1] + x_random
    y_synthetic = pos[::,0] + y_random

    return x_synthetic,y_synthetic

###########################################
## SETUP PATHS
###########################################

if os.getcwd()[7:14] == 'tgardne':
    ## setup paths for user
    path = '/Users/tgardner/ARMADA_orbits'
    path_etalon = '/Users/tgardner/etalon_epochs/etalon_fits/etalon_factors_fit.txt'
    path_wds = '/Users/tgardner/wds_targets'
    path_orb6 = '/Users/tgardner/catalogs/orb6orbits.sql.txt'
    
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

#emethod = input('bootstrap errors? (y/n) ')
emethod = 'n'
#mirc_scale = input('Scale OLD MIRCX data? (y/n) ')
mirc_scale = 'n'

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

own_files = input('Input your own orbit files? (y,[n]): ')
if own_files=='y':
    file = input('Path to armada orbit file: ')
    file = open(file)
else:
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
if own_files=='y':
    print('WARNING: Do not apply etalon correction if your own file has it already!')
etalon = input('Apply etalon correction? (y/n) ')

## FIXME: make it easier to choose vlti data
vlti = input('Add indices for vlti (y/n)? ')
#vlti = 'n'
if vlti=='y':
    vlti_idx = input('enter indices (e.g. 1 2 3): ').split(' ')
    vlti_idx = np.array([int(i) for i in vlti_idx])
else:
    vlti_idx = []

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

vlti_mask = np.ones(len(t),dtype=bool)
vlti_mask[vlti_idx] = False

###########################################
## Read in WDS data - and plot to check
###########################################
input_wds = input('Include WDS? (y/n): ')
if input_wds == 'y':
    try:
        if own_files=='y':
            file = input('Path to WDS orbit file: ')
            file = open(file)
            t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds = read_data(file,1)
            print('Number of WDS data points = %s'%len(p_wds))
        else:
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
        if len(vlti_idx)>0:
            plt.plot(xpos[vlti_mask],ypos[vlti_mask],'+',label='ARMADA-CHARA')
            plt.plot(xpos[vlti_idx],ypos[vlti_idx],'+',label='ARMADA-VLTI')
        else:
            plt.plot(xpos,ypos,'+',label='ARMADA')
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
                if len(vlti_idx)>0:
                    plt.plot(xpos[vlti_mask],ypos[vlti_mask],'+',label='ARMADA-CHARA')
                    plt.plot(xpos[vlti_idx],ypos[vlti_idx],'+',label='ARMADA-VLTI')
                else:
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

vlti_mask_all = np.ones(len(t_all),dtype=bool)
vlti_mask_all[vlti_idx] = False

##########################################
## Function for fitting/plotting data
#########################################
def ls_fit(params,xp,yp,tp,emaj,emin,epa):
    #do fit, minimizer uses LM for least square fitting of model to data
    if len(vlti_idx)>0:
        minner = Minimizer(astrometry_model_vlti, params, fcn_args=(xp[vlti_mask_all],yp[vlti_mask_all],tp[vlti_mask_all],
                                                        emaj[vlti_mask_all],emin[vlti_mask_all],epa[vlti_mask_all],
                                                        xp[vlti_idx],yp[vlti_idx],tp[vlti_idx],
                                                        emaj[vlti_idx],emin[vlti_idx],epa[vlti_idx]),
                                nan_policy='omit')
        result = minner.minimize()
    else:
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
    mirc_scale_start = result.params['mirc_scale']

    ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                            w_start,bigw_start,P_start,
                                            T_start,t_all)
    fig,ax=plt.subplots()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx],ypos[vlti_idx],'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask]/mirc_scale_start,ypos[vlti_mask]/mirc_scale_start,'o', label='ARMADA-CHARA')
    else:
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
    mirc_scale_results = []
    chi2_results = []

    for s in tqdm(np.arange(100)):
        x1 = random.uniform(0,360)
        x2 = random.uniform(0,360)
        x3 = random.uniform(0,180)
        x4 = random.uniform(0,0.9)
        x5 = random.uniform(astart,aend)
        x6 = random.uniform(Pstart,Pend)
        x7 = random.uniform(58000,60000)
        #x7 = random.uniform(58500,59000)

        params = Parameters()
        params.add('w',   value= x1)#, min=0, max=360)
        params.add('bigw', value= x2)#, min=-0, max=360)
        params.add('inc', value= x3, min=0, max=180)
        params.add('e', value= x4, min=0, max=0.99)
        params.add('a', value= x5, min=0)
        params.add('P', value= x6, min=0)
        params.add('T', value= x7, min=0)
        if len(vlti_idx)>0:
            params.add('mirc_scale', value=1)
        else:
            params.add('mirc_scale',value=1,vary=False)

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
        mirc_scale_results.append(result.params['mirc_scale'].value)
        chi2_results.append(result.redchi)

    w_results = np.array(w_results)
    bigw_results = np.array(bigw_results)
    inc_results = np.array(inc_results)
    e_results = np.array(e_results)
    a_results = np.array(a_results)
    P_results = np.array(P_results)
    T_results = np.array(T_results)
    mirc_scale_results = np.array(mirc_scale_results)
    chi2_results = np.array(chi2_results)

    idx = np.argmin(chi2_results)
    omega = w_results[idx]
    bigomega = bigw_results[idx]
    inc = inc_results[idx]
    e = e_results[idx]
    a = a_results[idx]
    P = P_results[idx]
    T = T_results[idx]
    mirc_scale = mirc_scale_results[idx]

    print('P, a, e, inc, w, bigw, T, mirc_scale: ')
    print(P/365, a, e, inc, omega, bigomega, T, mirc_scale)

params = Parameters()
params.add('w',   value= omega)#, min=0, max=360)
params.add('bigw', value= bigomega)#, min=0, max=360)
params.add('inc', value= inc, min=0, max=180)
params.add('e', value= e, min=0, max=0.99)
params.add('a', value= a, min=0)
params.add('P', value= P, min=0)
params.add('T', value= T, min=0)
if len(vlti_idx)>0:
    params.add('mirc_scale', value=1)
else:
    params.add('mirc_scale',value=1,vary=False)

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
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx],ypos[vlti_idx],'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask]/mirc_scale_start,ypos[vlti_mask]/mirc_scale_start,'o', label='ARMADA-CHARA')
    else:
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
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx],ypos[vlti_idx],'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask]/mirc_scale_start,ypos[vlti_mask]/mirc_scale_start,'o', label='ARMADA-CHARA')
    else:
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
    mirc_scale_start = result.params['mirc_scale']
    ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                        w_start,bigw_start,P_start,
                                        T_start,t_all)
    fig,ax=plt.subplots()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx],ypos[vlti_idx],'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask]/mirc_scale_start,ypos[vlti_mask]/mirc_scale_start,'o', label='ARMADA-CHARA')
    else:
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
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx],ypos[vlti_idx],'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask]/mirc_scale_start,ypos[vlti_mask]/mirc_scale_start,'o', label='ARMADA-CHARA')
    else:
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
    params.add('w',   value= omega)#, min=0, max=360)
    params.add('bigw', value= bigomega)#, min=0, max=360)
    params.add('inc', value= inc, min=0, max=180)
    params.add('e', value= e, min=0, max=0.99)
    params.add('a', value= a, min=0)
    params.add('P', value= P, min=0)
    params.add('T', value= T, min=0)
    if len(vlti_idx)>0:
        params.add('mirc_scale', value=1)
    else:
        params.add('mirc_scale',value=1,vary=False)

    result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all)
    filter_wds = input('Remove more data? (y/n)')

##########################################
## Save Plots
##########################################
if len(vlti_idx)>0:
    resids_armada = astrometry_model_vlti(result.params,xpos[vlti_mask],ypos[vlti_mask],t[vlti_mask],error_maj[vlti_mask],
                                error_min[vlti_mask],error_pa[vlti_mask],
                                xpos[vlti_idx],ypos[vlti_idx],t[vlti_idx],
                                error_maj[vlti_idx],error_min[vlti_idx],error_pa[vlti_idx])
else:
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

#rescale = input('Rescale errors based off chi2? (y/n): ')
rescale = 'n'
while rescale=='y':

    ## we don't want to raise armada errors
    if chi2_armada<0: # or chi2_armada>1:
        chi2_armada=1.0
    if chi2_wds<0:
        chi2_wds=1.0

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
    params.add('w',   value= omega)#, min=0, max=360)
    params.add('bigw', value= bigomega)#, min=0, max=360)
    params.add('inc', value= inc, min=0, max=180)
    params.add('e', value= e, min=0, max=0.99)
    params.add('a', value= a, min=0)
    params.add('P', value= P, min=0)
    params.add('T', value= T, min=0)
    if len(vlti_idx)>0:
        params.add('mirc_scale', value=1)
    else:
        params.add('mirc_scale',value=1,vary=False)

    result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all)

    if len(vlti_idx)>0:
        resids_armada = astrometry_model_vlti(result.params,xpos[vlti_mask],ypos[vlti_mask],t[vlti_mask],error_maj[vlti_mask],
                                    error_min[vlti_mask],error_pa[vlti_mask],
                                    xpos[vlti_idx],ypos[vlti_idx],t[vlti_idx],
                                    error_maj[vlti_idx],error_min[vlti_idx],error_pa[vlti_idx])
    else:
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

directory='%s/synthetic_data/HD%s/'%(path,target_hd)
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
mirc_scale_start = result.params['mirc_scale']
chi2_best_binary = result.chisqr
chi2_best_binary_armada = chi2_armada*(ndata_armada-len(result.params))
ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                        w_start,bigw_start,P_start,
                                        T_start,t_all)
fig,ax=plt.subplots()
ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
if len(vlti_idx)>0:
    ax.plot(xpos[vlti_idx],ypos[vlti_idx],'o', label='ARMADA-VLTI')
    ax.plot(xpos[vlti_mask]/mirc_scale_start,ypos[vlti_mask]/mirc_scale_start,'o', label='ARMADA-CHARA')
else:
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

## calculate resids for ARMADA
if len(vlti_idx)>0:
    xresid_vlti = xpos[vlti_idx] - rapoints[:len(xpos)][vlti_idx]
    yresid_vlti = ypos[vlti_idx] - decpoints[:len(ypos)][vlti_idx]
    xresid_chara = xpos[vlti_mask]/mirc_scale_start - rapoints[:len(xpos)][vlti_mask]
    yresid_chara = ypos[vlti_mask]/mirc_scale_start - decpoints[:len(ypos)][vlti_mask]
    xresid = np.concatenate([xresid_chara,xresid_vlti])
    yresid = np.concatenate([yresid_chara,yresid_vlti])
else:
    xresid = xpos - rapoints[:len(xpos)]
    yresid = ypos - decpoints[:len(ypos)]

## residuals
resids = np.sqrt(xresid**2 + yresid**2)
resids_median = np.around(np.median(resids)*1000,2)
print('-'*10)
print('Median residual = %s micro-as'%resids_median)
print('-'*10)

params_inner=[]
params_outer=[]
chi2 = []
chi2_armada = []
zval = []
zval_armada = []

## New test -- try period spacing from PHASES III paper
specify_period = input('Specify period search (y/[n])?')
if specify_period=='y':
    ps = float(input('period search start (days): '))
    pe = float(input('period search end (days): '))
    steps = int(input('steps: '))
    P2 = np.logspace(np.log10(ps),np.log10(pe),steps)
    #P2 = np.linspace(ps,pe,steps)
else:
    time_span = max(t) - min(t)
    print('Time span of data = %s days'%time_span)
    f = 3
    min_per = float(input('minimum period to search (days) = '))
    #min_per = 2
    max_k = int(2*f*time_span / min_per)
    k_range = np.arange(max_k)[:-1] + 1
    P2 = 2*f*time_span / k_range
    #P2 = np.logspace(np.log10(0.5),np.log10(5),5000)
    print('Min/Max period (days) = %s / %s ; %s steps'%(min(P2),max(P2),len(k_range)))
    #P2 = np.linspace(20,30,1000)

nsims = 10
kepler_params = [a_start,P_start,e_start,bigw_start,inc_start,w_start,T_start]
xspread = np.nanmean(np.abs(xresid))
yspread = np.nanmean(np.abs(yresid))
for sim in tqdm(np.arange(nsims)):

    #######################################################
    ## Create a synthetic dataset to run a triple search on
    #######################################################

    ## could do this based on error ellipses, but I will try based on residuals first
    ## since we seem to be dominated by residuals not always captured by error ellipse
    xpos_synthetic,ypos_synthetic = synthetic_data(kepler_params,t,np.full(len(xpos),xspread),np.full(len(xpos),yspread),np.pi/180*90)

    ###########################################
    ## Combined WDS+ARMADA for synthetic data
    ###########################################
    xpos_all_new = np.concatenate([xpos_synthetic,xpos_wds])
    ypos_all_new = np.concatenate([ypos_synthetic,ypos_wds])
    xpos_new = xpos_synthetic
    ypos_new = ypos_synthetic

    ##########################################
    ## First compute binary chi2
    ##########################################
    if len(vlti_idx)>0:
        minner = Minimizer(astrometry_model_vlti, params, fcn_args=(xpos_all_new[vlti_mask_all],ypos_all_new[vlti_mask_all],t_all[vlti_mask_all],
                                                        error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],error_pa_all[vlti_mask_all],
                                                        xpos_all_new[vlti_idx],ypos_all_new[vlti_idx],t_all[vlti_idx],
                                                        error_maj_all[vlti_idx],error_min_all[vlti_idx],error_pa_all[vlti_idx]),
                                nan_policy='omit')
        result = minner.minimize()
    else:
        minner = Minimizer(astrometry_model, params, fcn_args=(xpos_all_new,ypos_all_new,t_all,
                                                        error_maj_all,error_min_all,error_pa_all),
                                nan_policy='omit')
        result = minner.minimize()

    if len(vlti_idx)>0:
        resids_armada = astrometry_model_vlti(result.params,xpos_new[vlti_mask],ypos_new[vlti_mask],t[vlti_mask],error_maj[vlti_mask],
                                    error_min[vlti_mask],error_pa[vlti_mask],
                                    xpos_new[vlti_idx],ypos_new[vlti_idx],t[vlti_idx],
                                    error_maj[vlti_idx],error_min[vlti_idx],error_pa[vlti_idx])
    else:
        resids_armada = astrometry_model(result.params,xpos_new,ypos_new,t,error_maj,
                                    error_min,error_pa)
        
    chi2_best_binary = result.chisqr
    chi2_best_binary_armada = np.nansum(resids_armada**2)

    a2 = resids_median/1000
    if np.isnan(a2):
        a2=1
    #T2 = 55075

    params_inner_sim=[]
    params_outer_sim=[]
    chi2_sim = []
    chi2_armada_sim = []

    params = Parameters()
    params.add('w',   value= w_start)#, min=0, max=360)
    params.add('bigw', value= bigw_start)#, min=0, max=360)
    params.add('inc', value= inc_start, min=0, max=180)
    params.add('e', value= e_start, min=0, max=0.99)
    params.add('a', value= a_start, min=0)
    params.add('P', value= P_start, min=0)
    params.add('T', value= T_start, min=0)
    params.add('w2',   value= 0, vary=False)
    params.add('bigw2', value= 100)#, min=0, max=360)
    params.add('inc2', value= 100, min=0, max=180)
    params.add('e2', value= 0, vary=False)
    params.add('a2', value= a2, min=0)
    params.add('P2', value= 1, vary=False)
    params.add('T2', value= 1, min=0)
    if len(vlti_idx)>0:
        params.add('mirc_scale', value=mirc_scale_start)
    else:
        params.add('mirc_scale',value=1,vary=False)

    ## randomize orbital elements
    niter = 20
    bigw2 = np.random.uniform(0,360,niter)
    inc2 = np.random.uniform(0,180,niter)
    T2 = np.random.uniform(np.nanmean(t)-max(P2),np.nanmean(t)+max(P2),niter)

    #w1 = np.random.uniform(0,360,niter)
    #bigw1 = np.random.uniform(0,360,niter)
    #inc1 = np.random.uniform(0,180,niter)

    iter_num = 0
    for period in P2:

        params_inner_n=[]
        params_outer_n=[]
        chi2_n = []
        chi2_n_armada = []
        #chi2_noise_n = []

        for i in np.arange(niter):

            params['bigw2'].value = bigw2[i]
            params['inc2'].value = inc2[i]
            params['P2'].value = period
            params['T2'].value = T2[i]

            #params['w'].value = w1[i]
            #params['bigw'].value = bigw1[i]
            #params['inc'].value = inc1[i]

            #do fit, minimizer uses LM for least square fitting of model to data
            if len(vlti_idx)>0:
                minner = Minimizer(triple_model_circular_vlti, params, fcn_args=(xpos_all_new[vlti_mask_all],ypos_all_new[vlti_mask_all],t_all[vlti_mask_all],
                                                                    error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],
                                                                    error_pa_all[vlti_mask_all],
                                                                    xpos_all_new[vlti_idx],ypos_all_new[vlti_idx],t_all[vlti_idx],
                                                                    error_maj_all[vlti_idx],error_min_all[vlti_idx],
                                                                    error_pa_all[vlti_idx]),
                                    nan_policy='omit')
            else:
                minner = Minimizer(triple_model_circular, params, fcn_args=(xpos_all_new,ypos_all_new,t_all,
                                                                    error_maj_all,error_min_all,
                                                                    error_pa_all),
                                    nan_policy='omit')

            result = minner.leastsq(xtol=1e-5,ftol=1e-5)
            #result = minner.minimize()

            params_inner_n.append([period,result.params['a2'],result.params['e2'],result.params['w2']
                                ,result.params['bigw2'],result.params['inc2'],result.params['T2'],result.params['mirc_scale']])
            params_outer_n.append([result.params['P'],result.params['a'],result.params['e'],result.params['w']
                                ,result.params['bigw'],result.params['inc'],result.params['T']])
            #chi2_n.append(result.redchi)
            chi2_n.append(result.chisqr)

            if len(vlti_idx)>0:
                resids_armada = triple_model_circular_vlti(result.params,xpos_new[vlti_mask],ypos_new[vlti_mask],t[vlti_mask],
                                                                    error_maj[vlti_mask],error_min[vlti_mask],
                                                                    error_pa[vlti_mask],xpos_new[vlti_idx],ypos_new[vlti_idx],
                                                                    t[vlti_idx],error_maj[vlti_idx],error_min[vlti_idx],
                                                                    error_pa[vlti_idx])
            else:
                resids_armada = triple_model_circular(result.params,xpos_new,ypos_new,t,error_maj,
                                error_min,error_pa)

            chi2_armada_only = np.nansum(resids_armada**2)
            chi2_n_armada.append(chi2_armada_only)

        params_inner_n=np.array(params_inner_n)
        params_outer_n=np.array(params_outer_n)
        chi2_n = np.array(chi2_n)
        chi2_n_armada = np.array(chi2_n_armada)
        #chi2_noise_n = np.array(chi2_noise_n)

        idx = np.nanargmin(chi2_n)
        #idx_n = np.argmin(chi2_noise_n)
        chi2_sim.append(chi2_n[idx])
        chi2_armada_sim.append(chi2_n_armada[idx])
        #chi2_noise.append(chi2_noise_n[idx_n])
        params_inner_sim.append(params_inner_n[idx])
        params_outer_sim.append(params_outer_n[idx])


    params_inner.append(np.array(params_inner_sim))
    params_outer.append(np.array(params_outer_sim))
    chi2.append(np.array(chi2_sim))
    chi2_armada.append(np.array(chi2_armada_sim))

    zval.append(((4*sum(~np.isnan(xpos_all_new))-11)/(11-7))*(chi2_best_binary-chi2_sim)/min(chi2_sim))
    zval_armada.append(((4*sum(~np.isnan(xpos_new))-11)/(11-7))*(chi2_best_binary_armada-chi2_armada_sim)/min(chi2_armada_sim))


params_inner = np.array(params_inner)
params_outer = np.array(params_outer)
chi2 = np.array(chi2)
chi2_armada = np.array(chi2_armada)
zval = np.array(zval)
zval_armada = np.array(zval_armada)

#idx = np.argmin(chi2)
#chi2_best = np.nanmin(chi2)
#idx_armada = np.argmin(chi2_armada)
#chi2_best_armada = np.nanmin(chi2_armada)
#
#period_best = params_inner[:,0][idx]
#period_best_armada = params_inner[:,0][idx_armada]

## save parameter arrays
np.save('%s/HD%s_%s_params_inner.npy'%(directory,target_hd,date),params_inner)
np.save('%s/HD%s_%s_params_outer.npy'%(directory,target_hd,date),params_outer)
np.save('%s/HD%s_%s_chi2.npy'%(directory,target_hd,date),chi2)
np.save('%s/HD%s_%s_chi2_armada.npy'%(directory,target_hd,date),chi2_armada)
np.save('%s/HD%s_%s_zval.npy'%(directory,target_hd,date),zval)
np.save('%s/HD%s_%s_zval_armada.npy'%(directory,target_hd,date),zval_armada)

##plt.plot(params_inner[:,0],1/chi2_noise,'.--')
for x,y, in zip(params_inner[:,:,0],1/chi2):
    plt.plot(x,y,'o--')
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('1/chi2')
plt.savefig('%s/HD%s_%s_chi2_period.pdf'%(directory,target_hd,date))
plt.close()
#
##plt.plot(params_inner[:,0],1/chi2_noise,'.--')
for x,y in zip(params_inner[:,:,0],1/chi2_armada):
    plt.plot(x,y,'o--')
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('1/chi2')
plt.savefig('%s/HD%s_%s_chi2_period_armada.pdf'%(directory,target_hd,date))
plt.close()
#
for x,y in zip(params_inner[:,:,0],zval):
    plt.plot(x,y,'o-')
zval_max = np.max(zval)
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('z(P)')
plt.title('zval max = %s'%np.around(zval_max,2))
plt.savefig('%s/HD%s_%s_zval_period.pdf'%(directory,target_hd,date))
plt.close()
#
for x,y, in zip(params_inner[:,:,0],zval_armada):
    plt.plot(x,y,'o-')
zval_max = np.max(zval_armada)
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('z(P)')
plt.title('zval max = %s'%np.around(zval_max,2))
plt.savefig('%s/HD%s_%s_zval_period_armada.pdf'%(directory,target_hd,date))
plt.close()