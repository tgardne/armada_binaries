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

def add_planet(period,wobble,bigomega,inc,t0,epochs):
    omega = 0
    ecc = 0
    bigomega = bigomega
    inc = inc
    
    ## other method:
    ke = pyasl.KeplerEllipse(wobble,period,e=ecc,Omega=bigomega,i=inc,w=omega,tau=t0)
    pos = ke.xyzPos(epochs)
    xx = pos[::,1]
    yy = pos[::,0]

    return(xx,yy)

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
distance = 100
mass_star = 2
#distance = float(input('Distance (pc): '))
#mass_star = float(input('Mass Star (Msun): '))
#target = input('Target HIP #: ')
#target_wds = input('Target WDS #: ')

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

rescale = input('Rescale errors based off chi2? (y/n): ')
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

if emethod == 'y':
    directory='%s/HD%s_bootstrap/'%(path,target_hd)
else:
    directory='%s/HD%s_chi2err/'%(path,target_hd)
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
chi2_best_binary = chi2_armada*(ndata_armada-len(result.params))
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

## plot resids for ARMADA
fig,ax=plt.subplots()
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

#need to measure error ellipse angle east of north
for ras, decs, w, h, angle, d in zip(xresid,yresid,error_maj/scale,error_min/scale,error_deg,t):
    ellipse = Ellipse(xy=(ras, decs), width=2*w, height=2*h, 
                      angle=90-angle, facecolor='none', edgecolor='black')
    ax.annotate(d,xy=(ras,decs))
    ax.add_patch(ellipse)

if len(vlti_idx)>0:
    ax.plot(xresid[vlti_idx],yresid[vlti_idx],'o', label='ARMADA-VLTI')
    ax.plot(xresid[vlti_mask],yresid[vlti_mask],'o', label='ARMADA-CHARA')
else:
    ax.plot(xresid,yresid,'o', label='ARMADA')
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

## plot residuals vs time
plt.errorbar(t,resids,yerr=error_maj/scale,fmt='o')
plt.xlabel('Time (MJD)')
plt.ylabel('residual (mas)')
plt.title('Residuals vs Time')
plt.savefig('%s/HD%s_%s_resid_time.pdf'%(directory,target_hd,date))
plt.close()

## Option to run MCMC routine for errors
mcmc_option = input("Run MCMC on outer orbit? (y / [n]): ")
if mcmc_option == 'y':
    ## Run mcmc
    emcee_params = result.params.copy()
    nwalkers = 2*len(emcee_params)
    steps = 50000
    burn = 10000
    thin = 100

    if len(vlti_idx)>0:
        minner = Minimizer(astrometry_model_vlti, params, fcn_args=(xpos_all[vlti_mask_all],ypos_all[vlti_mask_all],t_all[vlti_mask_all],
                                                        error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],error_pa_all[vlti_mask_all],
                                                        xpos_all[vlti_idx],ypos_all[vlti_idx],t_all[vlti_idx],
                                                        error_maj_all[vlti_idx],error_min_all[vlti_idx],error_pa_all[vlti_idx]),
                                nan_policy='omit')
        result = minner.minimize(method='emcee',steps=steps,burn=burn,thin=thin,nwalkers=nwalkers)
    else:
        minner = Minimizer(astrometry_model, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                        error_maj_all,error_min_all,error_pa_all),
                                nan_policy='omit')
        result = minner.minimize(method='emcee',steps=steps,burn=burn,thin=thin,nwalkers=nwalkers)
    print(report_fit(result))
    chains = result.flatchain
    ## save chains
    print(chains.shape)
    np.save("%s/HD%s_%s_chains.npy"%(directory,target_hd,date),chains)
    ## load chains
    chains = np.load("%s/HD%s_%s_chains.npy"%(directory,target_hd,date))

    try:
        emcee_plot = corner.corner(chains,labels=result.var_names)
                            #truths=list(result.params.valuesdict().values()))
        plt.savefig('%s/HD%s_%s_corner.pdf'%(directory,target_hd,date))
    except:
        print(result.var_names)
        emcee_plot = corner.corner(chains)
        plt.savefig('%s/HD%s_%s_corner.pdf'%(directory,target_hd,date))
    plt.close()

    w_chain = chains[:,0]
    bigw_chain = chains[:,1]
    inc_chain = chains[:,2]
    e_chain = chains[:,3]
    a_chain = chains[:,4]
    P_chain = chains[:,5]
    T_chain = chains[:,6]
    
    w_mcmc = np.std(chains[:,0])
    bigw_mcmc = np.std(chains[:,1])
    inc_mcmc = np.std(chains[:,2])
    e_mcmc = np.std(chains[:,3])
    a_mcmc = np.std(chains[:,4])
    P_mcmc = np.std(chains[:,5])
    T_mcmc = np.std(chains[:,6])

    ## select random orbits from chains
    idx = np.random.randint(0,len(chains),size=100)
    fig,ax=plt.subplots()

    for orbit in idx:
        tmod = np.linspace(min(t_all),min(t_all)+2*P_start.value,1000)
        ra,dec,rapoints,decpoints = orbit_model(a_chain[orbit],e_chain[orbit],inc_chain[orbit],
                                                    w_chain[orbit],bigw_chain[orbit],P_chain[orbit],
                                                    T_chain[orbit],t_all,tmod)
        ax.plot(ra, dec, '--',color='lightgrey')

    tmod = np.linspace(min(t_all),min(t_all)+P_start.value,1000)
    ra,dec,rapoints,decpoints = orbit_model(a_start.value,e_start.value,inc_start.value,
                                            w_start.value,bigw_start.value,P_start.value,
                                            T_start.value,t_all,tmod)

    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS',color='grey')
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx],ypos[vlti_idx],'*', label='ARMADA-VLTI',color='red')
        ax.plot(xpos[vlti_mask],ypos[vlti_mask],'+', label='ARMADA-CHARA',color='blue')
    else:
        ax.plot(xpos,ypos,'+', label='ARMADA',color='red')
    ax.plot(0,0,'*',color='g')
    ax.plot(ra, dec, '--',color='g')

    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1

    ax.set_xlabel('dRA (mas)')
    ax.set_ylabel('dDEC (mas)')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('HD%s Outer Orbit'%target_hd)
    ax.legend()
    plt.savefig('%s/HD%s_%s_outer_mcmc.pdf'%(directory,target_hd,date))
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

## Save txt file with best orbit
f = open("%s/%s_%s_orbit_ls.txt"%(directory,target_hd,date),"w+")
f.write("# P(d) a(mas) e i(deg) w(deg) W(deg) T(mjd) scale mean_resid(mu-as)\r\n")
f.write("# Perr(d) aerr(mas) eerr ierr(deg) werr(deg) Werr(deg) Terr(mjd) scale_err \r\n")
f.write("%s %s %s %s %s %s %s %s %s\r\n"%(P_start.value,a_start.value,e_start.value,
                                   inc_start.value,w_start.value,
                                   bigw_start.value,T_start.value,mirc_scale_start.value,
                                  resids_median))
try:
    f.write("%s %s %s %s %s %s %s %s"%(P_start.stderr,a_start.stderr,e_start.stderr,
                                       inc_start.stderr,w_start.stderr,
                                       bigw_start.stderr,T_start.stderr,mirc_scale_start.stderr))
except:
    f.write("Errors not estimated")
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

## Save txt file with armada orbit
p_armada_new=[]
theta_armada_new = []
for i,j in zip(xpos_all[:len(xpos)],ypos_all[:len(xpos)]):
    pnew,tnew = cart2pol(i,j)
    p_armada_new.append(pnew)
    theta_armada_new.append(tnew)
p_armada_new = np.array(p_armada_new)
theta_armada_new = np.array(theta_armada_new)
f = open("%s/HD_%s_armada.txt"%(directory,target_hd),"w+")
f.write("# date mjd sep pa err_maj err_min err_pa\r\n")
for i,j,k,l,m,n in zip(t,p_armada_new,theta_armada_new,error_maj_all[:len(xpos)],error_min_all[:len(xpos)],error_deg_all[:len(xpos)]):
    f.write("-- %s %s %s %s %s %s\r\n"%(i,j,k,l,m,n))
f.write('#')
f.close()

search_triple = input('Search for triple? (y/n): ')
if search_triple=='n':
    exit()

##########################################
## Grid Search for Additional Companions
##########################################

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

ss = float(input('semi search start (mas): '))
se = float(input('semi search end (mas): '))

a2 = resids_median/1000
if np.isnan(a2):
    a2=1
#T2 = 55075

print('Grid Searching over period')
params_inner=[]
params_outer=[]
chi2 = []
chi2_noise = []

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
for period in tqdm(P2):

    params_inner_n=[]
    params_outer_n=[]
    chi2_n = []
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
            minner = Minimizer(triple_model_circular_vlti, params, fcn_args=(xpos_all[vlti_mask_all],ypos_all[vlti_mask_all],t_all[vlti_mask_all],
                                                                error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],
                                                                error_pa_all[vlti_mask_all],
                                                                xpos_all[vlti_idx],ypos_all[vlti_idx],t_all[vlti_idx],
                                                                error_maj_all[vlti_idx],error_min_all[vlti_idx],
                                                                error_pa_all[vlti_idx]),
                                nan_policy='omit')
        else:
            minner = Minimizer(triple_model_circular, params, fcn_args=(xpos_all,ypos_all,t_all,
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
        #chi2_n.append(result.chisqr)
        if len(vlti_idx)>0:
            resids_armada = triple_model_circular_vlti(result.params,xpos[vlti_mask],ypos[vlti_mask],t[vlti_mask],
                                                                error_maj[vlti_mask],error_min[vlti_mask],
                                                                error_pa[vlti_mask],xpos[vlti_idx],ypos[vlti_idx],
                                                                t[vlti_idx],error_maj[vlti_idx],error_min[vlti_idx],
                                                                error_pa[vlti_idx])
        else:
            resids_armada = triple_model_circular(result.params,xpos,ypos,t,error_maj,
                            error_min,error_pa)
            
        #ndata_armada = 2*sum(~np.isnan(xpos))
        chi2_armada = np.nansum(resids_armada**2)#/(ndata_armada-12)
        chi2_n.append(chi2_armada)

        ### noise for period search
        #ra,dec,rapoints,decpoints = orbit_model(result.params['a'],result.params['e'],
        #                                result.params['inc'],result.params['w'],
        #                                result.params['bigw'],result.params['P'],
        #                                result.params['T'],t)
        #xresid = xpos - rapoints
        #yresid = ypos - decpoints

        #params = Parameters()
        #params.add('w',   value= 0, vary=False)
        #params.add('bigw', value= bigw2, min=0, max=2*np.pi)
        #params.add('inc', value= inc2, min=0, max=np.pi)
        #params.add('e', value= 0, vary=False)
        #params.add('a', value= a2, min=0)
        #params.add('P', value= period, vary=False)
        #params.add('T', value= T2, min=0)
        #if mirc_scale == 'y':
        #    params.add('mirc_scale', value= 1.0)
        #else:
        #    params.add('mirc_scale', value= 1.0, vary=False)
        ##params.add('pscale', value=1)

        ##do fit, minimizer uses LM for least square fitting of model to data
        ##print(np.array(random.sample(list(t), len(t))))
        #tshuffle = np.array(random.sample(list(t), len(t)))
        #minner = Minimizer(astrometry_model, params, fcn_args=(xresid,yresid,tshuffle,
        #                                                   error_maj,error_min,
        #                                                   error_pa),nan_policy='omit')
        #result = minner.leastsq(xtol=1e-5,ftol=1e-5)
        #chi2_noise_n.append(result.redchi)
        ##resids_armada = astrometry_model(result.params,xresid,yresid,tshuffle,error_maj,
        ##                    error_min,error_pa)
        ##ndata_armada = 2*sum(~np.isnan(xresid))
        ##chi2_armada = np.nansum(resids_armada**2)/(ndata_armada-12)
        ##chi2_noise_n.append(chi2_armada)

    params_inner_n=np.array(params_inner_n)
    params_outer_n=np.array(params_outer_n)
    chi2_n = np.array(chi2_n)
    #chi2_noise_n = np.array(chi2_noise_n)

    idx = np.nanargmin(chi2_n)
    #idx_n = np.argmin(chi2_noise_n)
    chi2.append(chi2_n[idx])
    #chi2_noise.append(chi2_noise_n[idx_n])
    params_inner.append(params_inner_n[idx])
    params_outer.append(params_outer_n[idx])
    

params_inner=np.array(params_inner)
params_outer=np.array(params_outer)
chi2 = np.array(chi2)
#chi2_noise = np.array(chi2_noise)
zval = ((4*sum(~np.isnan(xpos))-11)/(11-7))*(chi2_best_binary-chi2)/min(chi2)
zval_max = np.nanmax(zval)

idx = np.argmin(chi2)
chi2_best = np.nanmin(chi2)
chi2_median = np.nanmedian(chi2)
n_over_median = chi2_median / chi2_best
#idx = np.argmax(zval)
period_best = params_inner[:,0][idx]

## save parameter arrays
np.save('%s/HD%s_%s_params_inner.npy'%(directory,target_hd,date),params_inner)
np.save('%s/HD%s_%s_params_outer.npy'%(directory,target_hd,date),params_outer)
np.save('%s/HD%s_%s_chi2.npy'%(directory,target_hd,date),chi2)
np.save('%s/HD%s_%s_zval.npy'%(directory,target_hd,date),zval)

#plt.plot(params_inner[:,0],1/chi2_noise,'.--')
plt.plot(params_inner[:,0],1/chi2,'o-')
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('1/chi2')
plt.title('Best Period = %s'%period_best)
plt.savefig('%s/HD%s_%s_chi2_period.pdf'%(directory,target_hd,date))
plt.close()

plt.plot(params_inner[:,0],zval,'o-')
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('z(P)')
plt.title('Best Period = %s'%period_best)
plt.savefig('%s/HD%s_%s_zval_period.pdf'%(directory,target_hd,date))
plt.close()

abin = ((params_inner[:,0]/365.25)**2*mass_star)**(1/3)
mass_planet = mass_star/(abin-params_inner[:,1]*distance/1000)*(params_inner[:,1]*distance/1000)/0.0009546
idx2 = np.where(params_inner[:,0]>20)
plt.plot(params_inner[:,0][idx2],mass_planet[idx2],'o-')
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('Mass (MJ)')
plt.title('Best Period = %s'%period_best)
plt.savefig('%s/HD%s_%s_mass_period.pdf'%(directory,target_hd,date))
plt.close()

print('Best inner period = %s'%period_best)

## Do a fit at best period
params = Parameters()
params.add('w',   value= params_outer[:,3][idx])#, min=0, max=360)
params.add('bigw', value= params_outer[:,4][idx])#, min=0, max=360)
params.add('inc', value= params_outer[:,5][idx], min=0, max=180)
params.add('e', value= params_outer[:,2][idx], min=0, max=0.99)
params.add('a', value=params_outer[:,1][idx], min=0)
params.add('P', value= params_outer[:,0][idx], min=0)
params.add('T', value= params_outer[:,6][idx], min=0)
params.add('w2',   value= 0, vary=False)#w2, min=0, max=360)
params.add('bigw2', value= params_inner[:,4][idx])#, min=0, max=360)
params.add('inc2', value= params_inner[:,5][idx], min=0, max=180)
params.add('e2', value= 0, vary=False)#0.1, min=0,max=0.99)
params.add('a2', value= params_inner[:,1][idx], min=0)
params.add('P2', value= period_best, min=0)
params.add('T2', value= params_inner[:,6][idx], min=0)
if len(vlti_idx)>0:
    params.add('mirc_scale', value=params_inner[:,7][idx])
else:
    params.add('mirc_scale', value= 1.0, vary=False)

#params.add('pscale', value=1)

#do fit, minimizer uses LM for least square fitting of model to data
if len(vlti_idx)>0:
    minner = Minimizer(triple_model_circular_vlti, params, fcn_args=(xpos_all[vlti_mask_all],ypos_all[vlti_mask_all],t_all[vlti_mask_all],
                                                        error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],
                                                        error_pa_all[vlti_mask_all],
                                                        xpos_all[vlti_idx],ypos_all[vlti_idx],t_all[vlti_idx],
                                                        error_maj_all[vlti_idx],error_min_all[vlti_idx],
                                                        error_pa_all[vlti_idx]),
                        nan_policy='omit')
else:
    minner = Minimizer(triple_model_circular, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                        error_maj_all,error_min_all,
                                                        error_pa_all),
                        nan_policy='omit')
result = minner.minimize()

best_inner = [result.params['P2'],result.params['a2'],result.params['e2'],result.params['w2']
                    ,result.params['bigw2'],result.params['inc2'],result.params['T2'],result.params['mirc_scale']]
best_outer = [result.params['P'],result.params['a'],result.params['e'],result.params['w']
                    ,result.params['bigw'],result.params['inc'],result.params['T']]
try:
    report_fit(result)
except:
    print('-'*10)
    print('Triple fit FAILED!!!!')
    print('-'*10)

############################
## Grid Search on a/i 
############################
a_grid = np.linspace(ss,se,50)
i_grid = np.linspace(0,180,50)

params_inner=[]
params_outer=[]
chi2 = []

params = Parameters()
params.add('w',   value= best_outer[3], vary=False)#min=0, max=260)
params.add('bigw', value= best_outer[4], vary=False)#min=0, max=260)
params.add('inc', value= best_outer[5], vary=False)#min=0, max=180)
params.add('e', value= best_outer[2], vary=False)#min=0, max=0.99)
params.add('a', value= best_outer[1], vary=False)#min=0)
params.add('P', value= best_outer[0], vary=False)#min=0)
params.add('T', value= best_outer[6], vary=False)#min=0)
params.add('w2',   value= 0, vary=False)
params.add('bigw2', value= best_inner[4], vary=False)#min=0, max=260)
params.add('inc2', value= 1, vary=False)#min=0, max=180)
params.add('e2', value= 0, vary=False)
params.add('a2', value= 1, vary=False)#a2, min=0)
params.add('P2', value= best_inner[0], vary=False)#min=0)
params.add('T2', value= best_inner[6], vary=False)#min=0)
if len(vlti_idx)>0:
    params.add('mirc_scale', value= best_inner[7])
else:
    params.add('mirc_scale', value= 1.0, vary=False)

for semi in tqdm(a_grid):
    for angle in i_grid:

        params['inc2'].value = angle
        params['a2'].value = semi

        #do fit, minimizer uses LM for least square fitting of model to data
        if len(vlti_idx)>0:
            minner = Minimizer(triple_model_circular_vlti, params, fcn_args=(xpos_all[vlti_mask_all],ypos_all[vlti_mask_all],t_all[vlti_mask_all],
                                                                error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],
                                                                error_pa_all[vlti_mask_all],
                                                                xpos_all[vlti_idx],ypos_all[vlti_idx],t_all[vlti_idx],
                                                                error_maj_all[vlti_idx],error_min_all[vlti_idx],
                                                                error_pa_all[vlti_idx]),
                                nan_policy='omit')
            result = minner.leastsq(xtol=1e-5,ftol=1e-5)
        else:
            minner = Minimizer(triple_model_circular, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                                error_maj_all,error_min_all,
                                                                error_pa_all),
                                nan_policy='omit')
            result = minner.leastsq(xtol=1e-5,ftol=1e-5)

        params_inner.append([result.params['P2'],semi,result.params['e2'],result.params['w2']
                            ,result.params['bigw2'],angle,result.params['T2'],result.params['mirc_scale']])
        params_outer.append([result.params['P'],result.params['a'],result.params['e'],result.params['w']
                            ,result.params['bigw'],result.params['inc'],result.params['T']])
        
        if len(vlti_idx)>0:
            resids_armada = triple_model_circular_vlti(result.params,xpos[vlti_mask],ypos[vlti_mask],t[vlti_mask],
                                                                error_maj[vlti_mask],error_min[vlti_mask],
                                                                error_pa[vlti_mask],xpos[vlti_idx],ypos[vlti_idx],
                                                                t[vlti_idx],error_maj[vlti_idx],error_min[vlti_idx],
                                                                error_pa[vlti_idx])
        else:
            #chi2.append(result.redchi)
            resids_armada = triple_model_circular(result.params,xpos,ypos,t,error_maj,
                            error_min,error_pa)
        
        chi2_armada = np.nansum(resids_armada**2)#/(ndata_armada-12)
        chi2.append(chi2_armada)

params_inner = np.array(params_inner)
params_outer = np.array(params_outer)
chi2 = np.array(chi2)

a_inner = params_inner[:,1]
i_inner = params_inner[:,5]
plt.scatter(a_inner,i_inner,c=1/chi2,cmap=cm.inferno)
#x=np.unique(a_inner)
#y=np.unique(i_inner)
#X,Y=np.meshgrid(x,y)
#Z=chi2.reshape(len(y),len(x))
#plt.pcolormesh(X,Y,1/Z)

plt.colorbar(label='1 / $\chi^2$')
plt.xlabel('semi-major (mas)')
plt.ylabel('inclination (deg)')
plt.savefig('%s/HD%s_%s_semi_inc_grid.pdf'%(directory,target_hd,date))
plt.close()