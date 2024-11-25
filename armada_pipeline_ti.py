######################################################################
## Tyler Gardner
##
## Pipeline to fit binary orbits
## and search for additional companions
##
## For binary orbits from MIRCX/GRAVITY
##
######################################################################

from lmfit import minimize, Minimizer, Parameters, report_fit
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from read_data import read_data,read_wds,read_orb6
from astrometry_model import astrometry_model_ti,triple_model_circular_ti,triple_model_circular_solve_linear
from orbit_plotting import orbit_model,orbit_model_ti
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import random
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

## Convert to Thiele-Innes params
A = a*(np.cos(omega*np.pi/180)*np.cos(bigomega*np.pi/180)-np.sin(omega*np.pi/180)*np.sin(bigomega*np.pi/180)*np.cos(inc*np.pi/180))
B = a*(np.cos(omega*np.pi/180)*np.sin(bigomega*np.pi/180)+np.sin(omega*np.pi/180)*np.cos(bigomega*np.pi/180)*np.cos(inc*np.pi/180))
F = -a*(np.sin(omega*np.pi/180)*np.cos(bigomega*np.pi/180)+np.cos(omega*np.pi/180)*np.sin(bigomega*np.pi/180)*np.cos(inc*np.pi/180))
G = -a*(np.sin(omega*np.pi/180)*np.sin(bigomega*np.pi/180)-np.cos(omega*np.pi/180)*np.cos(bigomega*np.pi/180)*np.cos(inc*np.pi/180))

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
        minner = Minimizer(astrometry_model_ti, params, fcn_args=(xp[vlti_mask_all],yp[vlti_mask_all],tp[vlti_mask_all],
                                                        emaj[vlti_mask_all],emin[vlti_mask_all],epa[vlti_mask_all],
                                                        xp[vlti_idx],yp[vlti_idx],tp[vlti_idx],
                                                        emaj[vlti_idx],emin[vlti_idx],epa[vlti_idx]),
                                nan_policy='omit')
        result = minner.minimize()
    else:
        minner = Minimizer(astrometry_model_ti, params, fcn_args=(xp,yp,tp,
                                                        emaj,emin,epa),
                                nan_policy='omit')
        result = minner.minimize()
    # write error report
    print(report_fit(result))

    ## plot fit
    A_start = result.params['A']
    B_start = result.params['B']
    F_start = result.params['F']
    G_start = result.params['G']
    e_start = result.params['e']
    P_start = result.params['P']
    T_start = result.params['T']
    mirc_scale_start = result.params['mirc_scale']

    ra,dec,rapoints,decpoints = orbit_model_ti(A_start,B_start,F_start,
                                            G_start,e_start,P_start,
                                            T_start,t_all)
    fig,ax=plt.subplots()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx]*mirc_scale_start,ypos[vlti_idx]*mirc_scale_start,'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask],ypos[vlti_mask],'o', label='ARMADA-CHARA')
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

    A_results = []
    B_results = []
    F_results = []
    G_results = []
    e_results = []
    P_results = []
    T_results = []
    mirc_scale_results = []
    chi2_results = []

    for s in tqdm(np.arange(100)):
        x1 = random.uniform(-aend,aend)
        x2 = random.uniform(-aend,aend)
        x3 = random.uniform(-aend,aend)
        x4 = random.uniform(-aend,aend)
        x5 = random.uniform(0,0.99)
        x6 = random.uniform(Pstart,Pend)
        x7 = random.uniform(np.nanmean(t_all)-Pend/2,np.nanmean(t_all)+Pend/2)
        #x7 = random.uniform(58500,59000)

        params = Parameters()
        params.add('A',   value= x1)#, min=0, max=360)
        params.add('B', value= x2)#, min=-0, max=360)
        params.add('F', value= x3)#, min=0, max=180)
        params.add('G', value= x4)#, min=0, max=0.99)
        params.add('e', value= x5, min=0,max=0.99)
        params.add('P', value= x6, min=0)
        params.add('T', value= x7, min=0)
        if len(vlti_idx)>0:
            params.add('mirc_scale', value=1)
        else:
            params.add('mirc_scale',value=1,vary=False)

        #do fit, minimizer uses LM for least square fitting of model to data
        minner = Minimizer(astrometry_model_ti, params, fcn_args=(xpos_all,ypos_all,t_all,
                                error_maj_all,error_min_all,error_pa_all),
                                nan_policy='omit')
        result = minner.minimize()

        A_results.append(result.params['A'].value)
        B_results.append(result.params['B'].value)
        F_results.append(result.params['F'].value)
        G_results.append(result.params['G'].value)
        e_results.append(result.params['e'].value)
        P_results.append(result.params['P'].value)
        T_results.append(result.params['T'].value)
        mirc_scale_results.append(result.params['mirc_scale'].value)
        chi2_results.append(result.redchi)

    A_results = np.array(A_results)
    B_results = np.array(B_results)
    F_results = np.array(F_results)
    G_results = np.array(G_results)
    e_results = np.array(e_results)
    P_results = np.array(P_results)
    T_results = np.array(T_results)
    mirc_scale_results = np.array(mirc_scale_results)
    chi2_results = np.array(chi2_results)

    idx = np.argmin(chi2_results)
    A = A_results[idx]
    B = B_results[idx]
    F = F_results[idx]
    G = G_results[idx]
    e = e_results[idx]
    P = P_results[idx]
    T = T_results[idx]
    mirc_scale = mirc_scale_results[idx]

params = Parameters()
params.add('A',   value= A)#, min=0, max=360)
params.add('B', value= B)#, min=0, max=360)
params.add('F', value= F)#, min=0, max=180)
params.add('G', value= G)#, min=0, max=0.99)
params.add('e', value= e, min=0, max=0.99)
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
        ax.plot(xpos[vlti_idx]*mirc_scale_start,ypos[vlti_idx]*mirc_scale_start,'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask],ypos[vlti_mask],'o', label='ARMADA-CHARA')
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
        ax.plot(xpos[vlti_idx]*mirc_scale_start,ypos[vlti_idx]*mirc_scale_start,'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask],ypos[vlti_mask],'o', label='ARMADA-CHARA')
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
    A_start = result.params['A']
    B_start = result.params['B']
    F_start = result.params['F']
    G_start = result.params['G']
    e_start = result.params['e']
    P_start = result.params['P']
    T_start = result.params['T']
    mirc_scale_start = result.params['mirc_scale']
    ra,dec,rapoints,decpoints = orbit_model_ti(A_start,B_start,F_start,
                                        G_start,e_start,P_start,
                                        T_start,t_all)
    fig,ax=plt.subplots()
    ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    if len(vlti_idx)>0:
        ax.plot(xpos[vlti_idx]*mirc_scale_start,ypos[vlti_idx]*mirc_scale_start,'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask],ypos[vlti_mask],'o', label='ARMADA-CHARA')
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
        ax.plot(xpos[vlti_idx]*mirc_scale_start,ypos[vlti_idx]*mirc_scale_start,'o', label='ARMADA-VLTI')
        ax.plot(xpos[vlti_mask],ypos[vlti_mask],'o', label='ARMADA-CHARA')
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
    params.add('A',   value= A)#, min=0, max=360)
    params.add('B', value= B)#, min=0, max=360)
    params.add('F', value= F)#, min=0, max=180)
    params.add('G', value= G)#, min=0, max=0.99)
    params.add('e', value= e, min=0,max=0.99)
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
    resids_armada = astrometry_model_ti(result.params,xpos[vlti_mask],ypos[vlti_mask],t[vlti_mask],error_maj[vlti_mask],
                                error_min[vlti_mask],error_pa[vlti_mask],
                                xpos[vlti_idx],ypos[vlti_idx],t[vlti_idx],
                                error_maj[vlti_idx],error_min[vlti_idx],error_pa[vlti_idx])
else:
    resids_armada = astrometry_model_ti(result.params,xpos,ypos,t,error_maj,
                                error_min,error_pa)
resids_wds = astrometry_model_ti(result.params,xpos_all[len(xpos):],ypos_all[len(xpos):],t_all[len(xpos):],
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
    params.add('A',   value= A)#, min=0, max=360)
    params.add('B', value= B)#, min=0, max=360)
    params.add('F', value= F)#, min=0, max=180)
    params.add('G', value= G)#, min=0, max=0.99)
    params.add('e', value= e, min=0, max=0.99)
    params.add('P', value= P, min=0)
    params.add('T', value= T, min=0)
    if len(vlti_idx)>0:
        params.add('mirc_scale', value=1)
    else:
        params.add('mirc_scale',value=1,vary=False)

    result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all)

    if len(vlti_idx)>0:
        resids_armada = astrometry_model_ti(result.params,xpos[vlti_mask],ypos[vlti_mask],t[vlti_mask],error_maj[vlti_mask],
                                    error_min[vlti_mask],error_pa[vlti_mask],
                                    xpos[vlti_idx],ypos[vlti_idx],t[vlti_idx],
                                    error_maj[vlti_idx],error_min[vlti_idx],error_pa[vlti_idx])
    else:
        resids_armada = astrometry_model_ti(result.params,xpos,ypos,t,error_maj,
                                    error_min,error_pa)

    resids_wds = astrometry_model_ti(result.params,xpos_all[len(xpos):],ypos_all[len(xpos):],t_all[len(xpos):],
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
A_start = result.params['A']
B_start = result.params['B']
F_start = result.params['F']
G_start = result.params['G']
e_start = result.params['e']
P_start = result.params['P']
T_start = result.params['T']
mirc_scale_start = result.params['mirc_scale']
chi2_best_binary = chi2_armada*(ndata_armada-len(result.params))
ra,dec,rapoints,decpoints = orbit_model_ti(A_start,B_start,F_start,
                                        G_start,e_start,P_start,
                                        T_start,t_all)
fig,ax=plt.subplots()
ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
if len(vlti_idx)>0:
    ax.plot(xpos[vlti_idx]*mirc_scale_start,ypos[vlti_idx]*mirc_scale_start,'o', label='ARMADA-VLTI')
    ax.plot(xpos[vlti_mask],ypos[vlti_mask],'o', label='ARMADA-CHARA')
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
    xresid_vlti = xpos[vlti_idx]*mirc_scale_start - rapoints[:len(xpos)][vlti_idx]
    yresid_vlti = ypos[vlti_idx]*mirc_scale_start - decpoints[:len(ypos)][vlti_idx]
    xresid_chara = xpos[vlti_mask] - rapoints[:len(xpos)][vlti_mask]
    yresid_chara = ypos[vlti_mask] - decpoints[:len(ypos)][vlti_mask]
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
        minner = Minimizer(astrometry_model_ti, params, fcn_args=(xpos_all[vlti_mask_all],ypos_all[vlti_mask_all],t_all[vlti_mask_all],
                                                        error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],error_pa_all[vlti_mask_all],
                                                        xpos_all[vlti_idx],ypos_all[vlti_idx],t_all[vlti_idx],
                                                        error_maj_all[vlti_idx],error_min_all[vlti_idx],error_pa_all[vlti_idx]),
                                nan_policy='omit')
        result = minner.minimize(method='emcee',steps=steps,burn=burn,thin=thin,nwalkers=nwalkers)
    else:
        minner = Minimizer(astrometry_model_ti, params, fcn_args=(xpos_all,ypos_all,t_all,
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

    A_chain = chains[:,0]
    B_chain = chains[:,1]
    F_chain = chains[:,2]
    G_chain = chains[:,3]
    e_chain = chains[:,4]
    P_chain = chains[:,5]
    T_chain = chains[:,6]
    
    A_mcmc = np.std(chains[:,0])
    B_mcmc = np.std(chains[:,1])
    F_mcmc = np.std(chains[:,2])
    G_mcmc = np.std(chains[:,3])
    e_mcmc = np.std(chains[:,4])
    P_mcmc = np.std(chains[:,5])
    T_mcmc = np.std(chains[:,6])

    ## select random orbits from chains
    idx = np.random.randint(0,len(chains),size=100)
    fig,ax=plt.subplots()

    for orbit in idx:
        tmod = np.linspace(min(t_all),min(t_all)+2*P_start.value,1000)
        ra,dec,rapoints,decpoints = orbit_model_ti(A_chain[orbit],B_chain[orbit],F_chain[orbit],
                                                    G_chain[orbit],e_chain[orbit],P_chain[orbit],
                                                    T_chain[orbit],t_all,tmod)
        ax.plot(ra, dec, '--',color='lightgrey')

    tmod = np.linspace(min(t_all),min(t_all)+P_start.value,1000)
    ra,dec,rapoints,decpoints = orbit_model(A_start.value,B_start.value,F_start.value,
                                            G_start.value,e_start.value,P_start.value,
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
    f.write("# A B F G e P(d) T(mjd) mean_resid(mu-as)\r\n")
    f.write("# Aerr Berr Ferr Gerr eerr Perr(d) Terr(mjd)\r\n")
    f.write("%s %s %s %s %s %s %s %s\r\n"%(A_start.value,B_start.value,F_start.value,
                                       G_start.value,e_start.value,
                                       P_start.value,T_start.value,
                                      resids_median))
    try:
        f.write("%s %s %s %s %s %s %s"%(A_mcmc,B_mcmc,F_mcmc,
                                           G_mcmc,e_mcmc,
                                           P_mcmc,T_mcmc))
    except:
        f.write("Errors not estimated")
    f.close()

    ### Save txt file for paper
    #f = open("%s/%s_%s_orbit_paper.txt"%(directory,target_hd,date),"w+")
    #f.write("# P(d) T(mjd) e w(deg) W(deg) i(deg) a(mas) med_resid(mu-as)\r\n")
    #f.write("$%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s\pm%s$ & $%s$\r\n"%(P_start.value,
    #                                w_mcmc,bigw_start.value,bigw_mcmc,inc_start.value,inc_mcmc,
    #                                P_mcmc,T_start.value,T_mcmc,e_start.value,e_mcmc,w_start.value,
    #                                a_start.value,a_mcmc,resids_median))
    #f.close()

## Save txt file with best orbit
f = open("%s/%s_%s_orbit_ls.txt"%(directory,target_hd,date),"w+")
f.write("# A B F G e P(d) T(mjd) mean_resid(mu-as)\r\n")
f.write("# Aerr Berr Ferr Gerr eerr Perr(d) Terr(mjd)\r\n")
f.write("%s %s %s %s %s %s %s %s %s\r\n"%(A_start.value,B_start.value,F_start.value,
                                   G_start.value,e_start.value,
                                   P_start.value,T_start.value,mirc_scale_start.value,
                                  resids_median))
try:
    f.write("%s %s %s %s %s %s %s %s"%(A_start.stderr,B_start.stderr,F_start.stderr,
                                       G_start.stderr,e_start.stderr,
                                       P_start.stderr,T_start.stderr,mirc_scale_start.stderr))
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
params.add('e', value= e_start.value, min=0)
params.add('P', value= P_start.value, min=0)
params.add('T', value= T_start.value, min=0)
params.add('e2', value= 0, vary=False)
params.add('P2', value= 1, vary=False)
params.add('T2', value= 1, min=0)
if len(vlti_idx)>0:
    params.add('mirc_scale', value=mirc_scale_start)
else:
    params.add('mirc_scale',value=1,vary=False)

## randomize orbital elements
niter = 20
T2 = np.random.uniform(np.nanmean(t)-max(P2)/2,np.nanmean(t)+max(P2)/2,niter)

iter_num = 0
for period in tqdm(P2):

    params['P2'].value = period

    params_inner_n=[]
    params_outer_n=[]
    chi2_n = []
    #chi2_noise_n = []

    for i in np.arange(niter):

        params['T2'].value = T2[i]

        #do fit, minimizer uses LM for least square fitting of model to data
        if len(vlti_idx)>0:
            minner = Minimizer(triple_model_circular_solve_linear, params, fcn_args=(xpos_all[vlti_mask_all],ypos_all[vlti_mask_all],t_all[vlti_mask_all],
                                                                error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],
                                                                error_pa_all[vlti_mask_all],
                                                                xpos_all[vlti_idx],ypos_all[vlti_idx],t_all[vlti_idx],
                                                                error_maj_all[vlti_idx],error_min_all[vlti_idx],
                                                                error_pa_all[vlti_idx]),
                                nan_policy='omit')
        else:
            minner = Minimizer(triple_model_circular_solve_linear, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                                error_maj_all,error_min_all,
                                                                error_pa_all),
                                nan_policy='omit')
        
        try:
            result = minner.leastsq(xtol=1e-5,ftol=1e-5)
        except:
            continue
        #result = minner.minimize()

        params_inner_n.append([result.params['P2'],result.params['T2'],result.params['mirc_scale']])
        params_outer_n.append([result.params['e'],result.params['P'],result.params['T']])
        #chi2_n.append(result.redchi)
        #chi2_n.append(result.chisqr)
        if len(vlti_idx)>0:
            resids_armada = triple_model_circular_solve_linear(result.params,xpos[vlti_mask],ypos[vlti_mask],t[vlti_mask],
                                                                error_maj[vlti_mask],error_min[vlti_mask],
                                                                error_pa[vlti_mask],xpos[vlti_idx],ypos[vlti_idx],
                                                                t[vlti_idx],error_maj[vlti_idx],error_min[vlti_idx],
                                                                error_pa[vlti_idx])
        else:
            resids_armada = triple_model_circular_solve_linear(result.params,xpos,ypos,t,error_maj,
                            error_min,error_pa)
            
        #ndata_armada = 2*sum(~np.isnan(xpos))
        chi2_armada = np.nansum(resids_armada**2)#/(ndata_armada-12)
        chi2_n.append(chi2_armada)

    params_inner_n=np.array(params_inner_n)
    params_outer_n=np.array(params_outer_n)
    chi2_n = np.array(chi2_n)

    idx = np.nanargmin(chi2_n)
    chi2.append(chi2_n[idx])
    params_inner.append(params_inner_n[idx])
    params_outer.append(params_outer_n[idx])
    

params_inner=np.array(params_inner)
params_outer=np.array(params_outer)
chi2 = np.array(chi2)
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

print('Best inner period = %s'%period_best)

## Do a fit at best period
params = Parameters()
params.add('e', value=params_outer[:,0][idx], min=0)
params.add('P', value= params_outer[:,1][idx], min=0)
params.add('T', value= params_outer[:,2][idx], min=0)
params.add('e2', value= 0, vary=False)
params.add('P2', value= period_best, min=0)
params.add('T2', value= params_inner[:,1][idx], min=0)
if len(vlti_idx)>0:
    params.add('mirc_scale', value=params_inner[:,2][idx])
else:
    params.add('mirc_scale', value= 1.0, vary=False)

#params.add('pscale', value=1)

#do fit, minimizer uses LM for least square fitting of model to data
if len(vlti_idx)>0:
    minner = Minimizer(triple_model_circular_solve_linear, params, fcn_args=(xpos_all[vlti_mask_all],ypos_all[vlti_mask_all],t_all[vlti_mask_all],
                                                        error_maj_all[vlti_mask_all],error_min_all[vlti_mask_all],
                                                        error_pa_all[vlti_mask_all],
                                                        xpos_all[vlti_idx],ypos_all[vlti_idx],t_all[vlti_idx],
                                                        error_maj_all[vlti_idx],error_min_all[vlti_idx],
                                                        error_pa_all[vlti_idx]),
                        nan_policy='omit')
else:
    minner = Minimizer(triple_model_circular_solve_linear, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                        error_maj_all,error_min_all,
                                                        error_pa_all),
                        nan_policy='omit')
result = minner.minimize()

try:
    report_fit(result)
except:
    print('-'*10)
    print('Triple fit FAILED!!!!')
    print('-'*10)