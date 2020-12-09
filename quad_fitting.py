######################################################################
## Tyler Gardner
##
## Do a fit for a quadruple
######################################################################

from chara_uvcalc import uv_calc
from binary_disks_vector import binary_disks_vector
from read_oifits import read_chara,read_vlti
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
eachindex = lambda lst: range(len(lst))
from tqdm import tqdm
import os
import matplotlib.cm as cm
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from matplotlib.patches import Ellipse
from ellipse_fitting import ellipse_hull_fit

######################################################################
## DEFINE FITTING FUNCTIONS
######################################################################

## function which converts cartesian to polar coords
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
    return(r,theta_new)

## function which returns complex vis given sep, pa, flux ratio, HA, dec, UD1, UD2, wavelength
def cvis_model(params, u, v, wl):
    try:   
        ra = params['ra']
        dec = params['dec']
        ratio = params['ratio']
        ud1 = params['ud1']
        ud2 = params['ud2']
        bw = params['bw']  
    except: 
        ra = params[0]
        dec = params[1]
        ratio = params[2]
        ud1 = params[3]
        ud2 = params[4]
        bw = params[5]
    
    ul=np.array([u/i for i in wl])
    vl=np.array([v/i for i in wl])
    
    vis=binary_disks_vector().binary2(ul,vl,ra,dec,ratio,ud1,ud2,bw)
    return vis

## same thing, but for quadruple
def quad_model(params, u, v, wl):  

    ra12 = params['ra12']
    dec12 = params['dec12']
    ra13 = params['ra13']
    dec13 = params['dec13']
    ra14 = params['ra14']
    dec14 = params['dec14']
    ratio12 = params['ratio12']
    ratio13 = params['ratio13']
    ratio14 = params['ratio14']
    ud1 = params['ud1']
    ud2 = params['ud2']
    ud3 = params['ud3']
    ud4 = params['ud4']  
    bw = params['bw']
    
    ul=np.array([u/i for i in wl])
    vl=np.array([v/i for i in wl])
    
    vis=binary_disks_vector().quad2(ul,vl,ra12,dec12,ra13,dec13,ra14,dec14,ratio12,ratio13,ratio14,ud1,ud2,ud3,ud4,bw)
    return vis

## function which returns residual of model and data to be minimized
def combined_minimizer(params,cp,cp_err,vphi,vphierr,v2,v2err,u_coord,v_coord,ucoord,vcoord,wl):

    diff=np.empty((0,len(wl)))

    if 'cphase' in flag:
        cp_model = []
        for item1,item2 in zip(u_coord,v_coord):
            complex_vis = cvis_model(params, item1, item2, wl)
            phase = np.angle(complex_vis[:,0])+np.angle(complex_vis[:,1])+np.angle(complex_vis[:,2])
            cp_model.append(phase)
        cp_model=np.array(cp_model)
        cp = cp*np.pi/180
        cp_err = cp_err*np.pi/180
        if absolute=='y':
            cp_diff = np.arctan2(np.sin(abs(cp)-abs(cp_model)),np.cos(abs(cp)-abs(cp_model)))/cp_err
        else:
            cp_diff = np.arctan2(np.sin(cp-cp_model),np.cos(cp-cp_model))/cp_err
        diff = np.append(diff,cp_diff,axis=0)

    if 'vis2' in flag:
        vis2_model = []
        for item1,item2 in zip(ucoord,vcoord):
            complex_vis2 = cvis_model(params,item1,item2,wl)
            visibility2 = complex_vis2*np.conj(complex_vis2)
            vis2_model.append(visibility2.real)
        vis2_model=np.array(vis2_model)
        vis2_diff = (v2 - vis2_model) / v2err
        diff = np.append(diff,vis2_diff,axis=0)

    if 'dphase' in flag:
        vis_model = []
        for item1,item2 in zip(ucoord,vcoord):
            complex_vis = cvis_model(params,item1,item2,wl)
            visibility = np.angle(complex_vis)
            if method=='dphase':
                dphase = visibility[1:]-visibility[:-1]
                dphase = np.insert(dphase,len(dphase),np.nan)
            else:
                dphase = visibility
            vis_model.append(dphase)
        vis_model=np.array(vis_model)
        vphi_data = vphi*np.pi/180
        vphi_err = vphierr*np.pi/180
        vphi_err[vphi_err==0]=100
        if absolute=='y':
            vphi_diff = np.arctan2(np.sin(abs(vphi_data)-abs(vis_model)),np.cos(abs(vphi_data)-abs(vis_model))) / vphi_err
        else:
            vphi_diff = np.arctan2(np.sin(vphi_data-vis_model),np.cos(vphi_data-vis_model)) / vphi_err
        vphi_diff = np.arctan2(np.sin(vphi_data-vis_model),np.cos(vphi_data-vis_model)) / vphi_err
        diff = np.append(diff,vphi_diff,axis=0)
    
    diff = np.array(diff)
    return diff

## for quadruple
## function which returns residual of model and data to be minimized
def quad_minimizer(params,cp,cp_err,vphi,vphierr,v2,v2err,u_coord,v_coord,ucoord,vcoord,wl):

    diff=np.empty((0,len(wl)))

    if 'cphase' in flag:
        cp_model = []
        for item1,item2 in zip(u_coord,v_coord):
            complex_vis = quad_model(params, item1, item2, wl)
            phase = np.angle(complex_vis[:,0])+np.angle(complex_vis[:,1])+np.angle(complex_vis[:,2])
            cp_model.append(phase)
        cp_model=np.array(cp_model)
        cp = cp*np.pi/180
        cp_err = cp_err*np.pi/180
        cp_diff = np.arctan2(np.sin(cp-cp_model),np.cos(cp-cp_model))/cp_err
        diff = np.append(diff,cp_diff,axis=0)

    if 'vis2' in flag:
        vis2_model = []
        for item1,item2 in zip(ucoord,vcoord):
            complex_vis2 = quad_model(params,item1,item2,wl)
            visibility2 = complex_vis2*np.conj(complex_vis2)
            vis2_model.append(visibility2.real)
        vis2_model=np.array(vis2_model)
        vis2_diff = (v2 - vis2_model) / v2err
        diff = np.append(diff,vis2_diff,axis=0)

    if 'dphase' in flag:
        vis_model = []
        for item1,item2 in zip(ucoord,vcoord):
            complex_vis = quad_model(params,item1,item2,wl)
            visibility = np.angle(complex_vis)
            if method=='dphase':
                dphase = visibility[1:]-visibility[:-1]
                dphase = np.insert(dphase,len(dphase),np.nan)
            else:
                dphase = visibility
            vis_model.append(dphase)
        vis_model=np.array(vis_model)
        vphi_data = vphi*np.pi/180
        vphi_err = vphierr*np.pi/180
        vphi_err[vphi_err==0]=100
        vphi_diff = np.arctan2(np.sin(vphi_data-vis_model),np.cos(vphi_data-vis_model)) / vphi_err
        diff = np.append(diff,vphi_diff,axis=0)
    
    diff = np.array(diff)
    return diff

######################################################################
## LOAD DATA
######################################################################

## Ask the user which file contains the closure phases
dtype = input('chara/vlti? ')
date=input('Date for saved files (e.g. 2018Jul19):')
dir=input('Path to oifits directory:')
target_id=input('Target ID (e.g. HD_206901): ')
interact = input('interactive session with data? (y/n): ')
exclude = input('exclude a telescope (e.g. E1): ')
absolute = input('use absolute phase value (y/n)?')

## get information from fits file
if dtype=='chara':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara(dir,target_id,interact,exclude)
if dtype=='vlti':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_vlti(dir,interact)
########################################################

################################################
## Dispersion fit for dphase
################################################

dispersion=[]
for vis in visphi:
    if np.count_nonzero(~np.isnan(vis))>0:
        y=vis
        x=eff_wave[0]
        idx = np.isfinite(x) & np.isfinite(y)
        z=np.polyfit(x[idx],y[idx],2)
        p = np.poly1d(z)
        dispersion.append(p(x))
    else:
        dispersion.append(vis)
dispersion=np.array(dispersion)

## subtract dispersion
visphi_new = visphi-dispersion

#method=input('VISPHI METHOD (dphase or visphi): ')
if dtype=='chara':
    method='dphase'
else:
    method='visphi'

flag = input('vis2,cphase,dphase (separate by spaces):')
reduction_params = input('ncoh nbs ncs int reduction_method: ').split(' ')

sep12_guess=float(input('sep12 guess (mas):'))
pa12_guess=float(input('PA12 guess (deg):'))
f12_guess = float(input('flux ratio (f1/f2): '))

sep13_guess=float(input('sep13 guess (mas):'))
pa13_guess=float(input('PA13 guess (deg):'))
f13_guess = float(input('flux ratio (f1/f3): '))

sep14_guess=float(input('sep14 guess (mas):'))
pa14_guess=float(input('PA14 guess (deg):'))
f14_guess = float(input('flux ratio (f1/f4): '))

ud1_guess = 0.5
ud2_guess = 0.5
ud3_guess = 0.5
ud4_guess = 0.5
bw_guess = 0.005

ra12_guess = -sep12_guess*np.cos((90+pa12_guess)*np.pi/180)
dec12_guess = sep12_guess*np.sin((90+pa12_guess)*np.pi/180)
ra13_guess = -sep13_guess*np.cos((90+pa13_guess)*np.pi/180)
dec13_guess = sep13_guess*np.sin((90+pa13_guess)*np.pi/180)
ra14_guess = -sep14_guess*np.cos((90+pa14_guess)*np.pi/180)
dec14_guess = sep14_guess*np.sin((90+pa14_guess)*np.pi/180)

######################################################################
## DO CHI2 FIT
######################################################################

print('Now doing joint fit for all components')

######################################################################
## DO CHI2 FIT
######################################################################

## Do a chi2 fit for phases
params = Parameters()
params.add('ra12',   value= ra12_guess)
params.add('dec12', value= dec12_guess)
params.add('ra13',   value= ra13_guess)
params.add('dec13', value= dec13_guess)
params.add('ra14',   value= ra14_guess)
params.add('dec14', value= dec14_guess)
params.add('ratio12', value= f12_guess, min=1.0)
params.add('ratio13', value= f13_guess, min=1.0)
params.add('ratio14', value= f14_guess, min=1.0)
params.add('ud1',   value= ud1_guess, vary=False)#min=0.0,max=2.0)
params.add('ud2', value= ud2_guess, vary=False)#min=0.0,max=2.0)
params.add('ud3', value= ud3_guess, vary=False)#min=0.0,max=2.0)
params.add('ud4', value= ud4_guess, vary=False)#min=0.0,max=2.0)
params.add('bw', value= bw_guess, min=0,max=1)

minner = Minimizer(quad_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
result = minner.minimize()
report_fit(result)

chi_sq_best = result.redchi
ra12_best = result.params['ra12'].value
dec12_best = result.params['dec12'].value
ra13_best = result.params['ra13'].value
dec13_best = result.params['dec13'].value
ra14_best = result.params['ra14'].value
dec14_best = result.params['dec14'].value
ratio12_best = result.params['ratio12'].value
ratio13_best = result.params['ratio13'].value
ratio14_best = result.params['ratio14'].value
ud1_best = result.params['ud1'].value
ud2_best = result.params['ud2'].value
ud3_best = result.params['ud3'].value
ud4_best = result.params['ud4'].value
bw_best = result.params['bw'].value

### Do a fit again to constrain UDs
#params = Parameters()
#params.add('ra12',   value= ra12_best,vary=False)
#params.add('dec12', value= dec12_best,vary=False)
#params.add('ra13',   value= ra13_best,vary=False)
#params.add('dec13', value= dec13_best,vary=False)
#params.add('ratio12', value= ratio12_best, vary=False)#min=1.0)
#params.add('ratio13', value= ratio13_best, vary=False)#min=1.0)
#params.add('ud1',   value= ud1_best, min=0.0,max=2.0)
#params.add('ud2', value= ud2_best, min=0.0,max=2.0)
#params.add('ud3', value= ud3_best, min=0.0,max=2.0)
#params.add('bw', value= bw_best, vary=False)#min=0,max=1)
#
#minner = Minimizer(triple_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
#result = minner.minimize()
#report_fit(result)
#
#chi_sq_best = result.redchi
#ra12_best = result.params['ra12'].value
#dec12_best = result.params['dec12'].value
#ra13_best = result.params['ra13'].value
#dec13_best = result.params['dec13'].value
#ratio12_best = result.params['ratio12'].value
#ratio13_best = result.params['ratio13'].value
#ud1_best = result.params['ud1'].value
#ud2_best = result.params['ud2'].value
#ud3_best = result.params['ud3'].value
#bw_best = result.params['bw'].value

######################################################################
## FORM MODEL FROM BEST FIT PARAMS TO PLOT AGAINST DATA
######################################################################

best_params = Parameters()
best_params.add('ra12',   value= ra12_best)
best_params.add('dec12', value= dec12_best)
best_params.add('ra13',   value= ra13_best)
best_params.add('dec13', value= dec13_best)
best_params.add('ra14',   value= ra14_best)
best_params.add('dec14', value= dec14_best)
best_params.add('ratio12', value= ratio12_best)
best_params.add('ratio13', value= ratio13_best)
best_params.add('ratio14', value= ratio14_best)
best_params.add('ud1',   value= ud1_best)
best_params.add('ud2', value= ud2_best)
best_params.add('ud3', value= ud3_best)
best_params.add('ud4', value= ud4_best)
best_params.add('bw', value=bw_best)

best_fit = np.around(np.array([ra12_best,dec12_best,ra13_best,dec13_best,ra14_best,dec14_best,ratio12_best,ratio13_best,ratio14_best,ud1_best,ud2_best,ud3_best,ud4_best,bw_best,chi_sq_best]),decimals=4)

cp_model = []
for item1,item2 in zip(u_coords,v_coords):
    complex_vis = quad_model(best_params, item1, item2, eff_wave[0])
    phase = (np.angle(complex_vis[:,0])+np.angle(complex_vis[:,1])+np.angle(complex_vis[:,2]))*180/np.pi
    cp_model.append(phase)
cp_model=np.array(cp_model)

visphi_model = []
for item1,item2 in zip(ucoords,vcoords):     
    complex_vis = quad_model(best_params,item1, item2, eff_wave[0])
    visibility = np.angle(complex_vis)*180/np.pi
    if method=='dphase':
        dphase = visibility[1:]-visibility[:-1]
        dphase = np.insert(dphase,len(dphase),np.nan)
    else:
        dphase = visibility
    visphi_model.append(dphase)
visphi_model=np.array(visphi_model)

vis2_model = []
for item1,item2 in zip(ucoords,vcoords):
    complex_vis2 = quad_model(best_params,item1,item2,eff_wave[0])
    visibility2 = complex_vis2*np.conj(complex_vis2)
    vis2_model.append(visibility2.real)
vis2_model=np.array(vis2_model)

## plot results
with PdfPages("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_QUAD_fit.pdf"%{"1":target_id,"2":date}) as pdf:

    ## regroup data by measurements
    n_cp = len(t3phi)/20
    n_vp = len(visphi)/15
    n_v2 = len(vis2)/15
    t3phi_plot = np.array(np.array_split(t3phi,n_cp))
    t3phierr_plot = np.array(np.array_split(t3phierr,n_cp))
    cp_model = np.array_split(cp_model,n_cp)
    visphi_new_plot = np.array_split(visphi_new,n_vp)
    visphierr_plot = np.array_split(visphierr,n_vp)
    visphi_model = np.array_split(visphi_model,n_vp)
    vis2_plot = np.array(np.array_split(vis2,n_v2))
    vis2err_plot = np.array(np.array_split(vis2err,n_v2))
    vis2_model = np.array(np.array_split(vis2_model,n_v2))
    
    ## next pages - cp model fits
    index = np.linspace(0,19,20)
    for t3data,t3errdata,modeldata in zip(t3phi_plot,t3phierr_plot,cp_model):

        label_size = 4
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        fig,axs = plt.subplots(4,5,figsize=(10,7),facecolor='w',edgecolor='k')
        fig.subplots_adjust(hspace=0.5,wspace=.001)
        axs=axs.ravel()

        for ind,y,yerr,m,tri in zip(index,t3data,t3errdata,modeldata,tels[:20]):
            x=eff_wave[0]
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
            axs[int(ind)].set_title(str(tri))

        fig.suptitle('%s Closure Phase'%target_id)
        fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
        fig.text(0.05, 0.5, 'CP (deg)', va='center', rotation='vertical')
        pdf.savefig()
        plt.close()

    ## next pages - visphi fits
    index = np.linspace(0,14,15)
    for visdata,viserrdata,modeldata in zip(visphi_new_plot,visphierr_plot,visphi_model):

        label_size = 4
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
        fig.subplots_adjust(hspace=0.5,wspace=.001)
        axs=axs.ravel()

        for ind,y,yerr,m,tri in zip(index,visdata,viserrdata,modeldata,vistels[:15]):
            x=eff_wave[0]
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
            axs[int(ind)].set_title(str(tri))

        fig.suptitle('%s VisPhi'%target_id)
        fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
        fig.text(0.05, 0.5, 'visphi (deg)', va='center', rotation='vertical')
        pdf.savefig()
        plt.close()

    ## next pages - vis2 fits
    index = np.linspace(0,14,15)
    for visdata,viserrdata,modeldata in zip(vis2_plot,vis2err_plot,vis2_model):

        label_size = 4
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
        fig.subplots_adjust(hspace=0.5,wspace=.001)
        axs=axs.ravel()

        for ind,y,yerr,m,tri in zip(index,visdata,viserrdata,modeldata,vistels[:15]):
            x=eff_wave[0]
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
            axs[int(ind)].set_title(str(tri))

        fig.suptitle('%s Vis2'%target_id)
        fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
        fig.text(0.05, 0.5, 'vis2', va='center', rotation='vertical')
        pdf.savefig()
        plt.close()

    ## next page - reduction and fit parameters
    textfig = plt.figure(figsize=(11.69,8.27))
    textfig.clf()
    txt_fit_cp = ('Best fit (ra12,dec12,ra13,dec13,ratio12,ratio13,ud1,ud2,ud2,bw,redchi2): %s'%best_fit)
    reduction = ('Reduction params (ncoh,nbs,ncs,int,oivis):%s'%reduction_params)
    textfig.text(0.5,0.75,txt_fit_cp,size=12,ha="center")
    textfig.text(0.5,0.25,reduction,size=12,ha="center")
    pdf.savefig()
    plt.close()