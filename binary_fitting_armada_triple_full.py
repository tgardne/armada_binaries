######################################################################
## Tyler Gardner
## Modified from John Monnier's IDL code
##
## Do a grid search to fit binary model
## Fitting to closure phase only
######################################################################

from chara_uvcalc import uv_calc
from binary_disks_vector import binary_disks_vector
from read_oifits import read_chara,read_vlti,read_chara_old
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

## same thing, but for triple
def triple_model(params, u, v, wl):  
    try:
        ra12 = params['ra12']
        dec12 = params['dec12']
        ra13 = params['ra13']
        dec13 = params['dec13']
        ratio12 = params['ratio12']
        ratio13 = params['ratio13']
        ud1 = params['ud1']
        ud2 = params['ud2']
        ud3 = params['ud3']  
        bw = params['bw']
    except: 
        ra12 = params[0]
        dec12 = params[1]
        ra13 = params[2]
        dec13 = params[3]
        ratio12 = params[4]
        ratio13 = params[5]
        ud1 = params[6]
        ud2 = params[7]
        ud3 = params[8]  
        bw = params[9]
    
    ul=np.array([u/i for i in wl])
    vl=np.array([v/i for i in wl])
    
    vis=binary_disks_vector().triple2(ul,vl,ra12,dec12,ra13,dec13,ratio12,ratio13,ud1,ud2,ud3,bw)
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
        vphi_diff = np.arctan2(np.sin(vphi_data-vis_model),np.cos(vphi_data-vis_model)) / vphi_err
        diff = np.append(diff,vphi_diff,axis=0)
    
    diff = np.array(diff)
    return diff

## for triple
## function which returns residual of model and data to be minimized
def triple_minimizer(params,cp,cp_err,vphi,vphierr,v2,v2err,u_coord,v_coord,ucoord,vcoord,wl):

    diff=np.empty((0,len(wl)))

    if 'cphase' in flag:
        cp_model = []
        for item1,item2 in zip(u_coord,v_coord):
            complex_vis = triple_model(params, item1, item2, wl)
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
            complex_vis2 = triple_model(params,item1,item2,wl)
            visibility2 = complex_vis2*np.conj(complex_vis2)
            vis2_model.append(visibility2.real)
        vis2_model=np.array(vis2_model)
        vis2_diff = (v2 - vis2_model) / v2err
        diff = np.append(diff,vis2_diff,axis=0)

    if 'dphase' in flag:
        vis_model = []
        for item1,item2 in zip(ucoord,vcoord):
            complex_vis = triple_model(params,item1,item2,wl)
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

## get information from fits file
#bl_drop = input('Drop long baselines? (y/n): ')
bl_drop='n'

if dtype=='chara':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,az = read_chara(dir,target_id,interact,exclude)
if dtype=='vlti':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,flux,fluxerr = read_vlti(dir,interact)
if dtype=='chara_old':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara_old(dir,interact,exclude)
########################################################
print("T3phi shape = ", t3phi.shape)
print("Vis2 shape = ", visphi.shape)

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

######################################################################
## Now fit for OUTER binary for best guess at elements
######################################################################

#method=input('VISPHI METHOD (dphase or visphi): ')
if dtype=='chara':
    method='dphase'
else:
    method='visphi'

flag = input('vis2,cphase,dphase (separate by spaces):')
reduction_params = input('ncoh nbs ncs int reduction_method: ').split(' ')

sep_value=float(input('outer sep guess (mas):'))
pa_value=float(input('outer PA guess (deg):'))
a3 = float(input('outer flux ratio (f1/f2): '))
a4 = float(input('UD1 (mas): '))
a5 = float(input('UD2 (mas): '))
a6 = float(input('bw smearing (0.004): '))

plot_grid = input('plot grid (y/n)? ')

dra = -sep_value*np.cos((90+pa_value)*np.pi/180)
ddec = sep_value*np.sin((90+pa_value)*np.pi/180)

## Do a chi2 fit for phases
params = Parameters()
params.add('ra',   value= dra)
params.add('dec', value= ddec)
params.add('ratio', value= a3, min=1.0)
params.add('ud1',   value= a4, vary=False)#min=0, max=2.0)
params.add('ud2', value= a5, vary=False)#min=0, max=2.0)
params.add('bw', value=a6, min=0, max=1)

if dtype=='vlti':
    minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi[:,::50],t3phierr[:,::50],visphi_new[:,::50],visphierr[:,::50],vis2[:,::50],vis2err[:,::50],u_coords,v_coords,ucoords,vcoords,eff_wave[0][::50]),nan_policy='omit')
    result = minner.minimize()
    report_fit(result)
else:
    minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
    result = minner.minimize()
    report_fit(result)

chi_sq_best_inner = result.redchi
ra_best_inner = result.params['ra'].value
dec_best_inner = result.params['dec'].value
ratio_best_inner = result.params['ratio'].value
ud1_best_inner = result.params['ud1'].value
ud2_best_inner = result.params['ud2'].value
bw_best_inner = result.params['bw'].value

######################################################################
## Now run a grid search for inner companion
######################################################################

sep_value=float(input('inner sep start (mas):'))
pa_value=float(input('inner PA start (deg):'))
grid_size = float(input('search grid size (mas): '))
steps = int(input('steps in grid: '))
a3 = float(input('inner flux ratio (f1/f2): '))
a7 = float(input('UD3 (mas): '))

dra = -sep_value*np.cos((90+pa_value)*np.pi/180)
ddec = sep_value*np.sin((90+pa_value)*np.pi/180)

ra_grid = np.linspace(dra-grid_size,dra+grid_size,steps)
dec_grid = np.linspace(ddec-grid_size,ddec+grid_size,steps)

chi_sq = []
ra12_results = []
dec12_results = []
ra13_results = []
dec13_results = []
ratio12_results = []
ratio13_results = []

## draw plot -- for now just show one grid map being plotted for a check (combined phases)
if plot_grid == 'y':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

vary_inner = input('Vary inner ratio? ([y]/n)? ')
## lmfit for varying params (slower)
params = Parameters()
params.add('ra12',   value= ra_best_inner)
params.add('dec12', value= dec_best_inner)
params.add('ra13',   value= 1,vary=False)
params.add('dec13', value= 1,vary=False)
params.add('ratio12', value= ratio_best_inner, vary=False)#min=1.0)
if vary_inner=='n':
    params.add('ratio13', value= a3, vary=False)
else:
    params.add('ratio13', value= a3, min=1.0)
params.add('ud1',   value= ud1_best_inner, vary=False)#min=0.0,max=2.0)
params.add('ud2', value= ud2_best_inner, vary=False)#min=0.0,max=2.0)
params.add('ud3', value= a7, vary=False)#min=0.0,max=2.0)
params.add('bw', value= a6, vary=False)#min=0,max=1)

for ra_try in tqdm(ra_grid):
    for dec_try in dec_grid:

        params['ra13'].value = ra_try
        params['dec13'].value = dec_try

        if dtype=='vlti':
            #print(t3phi.shape,u_coords.shape,ucoords.shape,eff_wave[0].shape)
            #print(t3phi[:,::3].shape)
            minner = Minimizer(triple_minimizer, params, fcn_args=(t3phi[:,::50],t3phierr[:,::50],visphi[:,::50],visphierr[:,::50],vis2[:,::50],vis2err[:,::50],u_coords,v_coords,ucoords,vcoords,eff_wave[0][::50]),nan_policy='omit')
            #result = minner.minimize()
            ## faster:
            result = minner.leastsq(xtol=1e-5,ftol=1e-5)
        else:
            minner = Minimizer(triple_minimizer, params, fcn_args=(t3phi,t3phierr,visphi,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
            #result = minner.minimize()
            ## faster:
            result = minner.leastsq(xtol=1e-5,ftol=1e-5)

        chi_sq.append(result.redchi)
        ra12_results.append(result.params['ra12'].value)
        dec12_results.append(result.params['dec12'].value)
        ra13_results.append(ra_try)
        dec13_results.append(dec_try)
        ratio12_results.append(result.params['ratio12'].value)
        ratio13_results.append(result.params['ratio13'].value)
    
    if plot_grid == 'y':
        ax.cla()
        ax.set_xlim(min(ra_grid),max(ra_grid))
        ax.set_ylim(min(dec_grid),max(dec_grid))
        ax.set_xlabel('d_RA (mas)')
        ax.set_ylabel('d_DE (mas)')
        ax.scatter(ra13_results,dec13_results,c=1/np.array(chi_sq),cmap=cm.inferno)
        ax.invert_xaxis()
        #plt.colorbar()
        plt.draw()
        plt.pause(0.001)

if plot_grid=='y':
    plt.show()
    plt.close()

ra12_results = np.array(ra12_results)
dec12_results = np.array(dec12_results)
ra13_results = np.array(ra13_results)
dec13_results = np.array(dec13_results)
ratio12_results = np.array(ratio12_results)
ratio13_results = np.array(ratio13_results)
chi_sq = np.array(chi_sq)

index = np.argmin(chi_sq)
best_params = [ra12_results[index],dec12_results[index],ra13_results[index],dec13_results[index],ratio12_results[index],ratio13_results[index],ud1_best_inner,ud2_best_inner,a7,a6]

######################################################################
## DO CHI2 FIT AT BEST POINT ON GRID FOR PLOT
######################################################################

print('Now doing joint fit for both components')

ra12_value=best_params[0]
dec12_value=best_params[1]
ra13_value=best_params[2]
dec13_value=best_params[3]
a5 = best_params[4]
a6 = best_params[5]
a7 = best_params[6]
a8 = best_params[7]
a9 = best_params[8]
a10 = best_params[9]

######################################################################
## DO CHI2 FIT
######################################################################

## Do a chi2 fit for phases
params = Parameters()
params.add('ra12',   value= ra12_value)
params.add('dec12', value= dec12_value)
params.add('ra13',   value= ra13_value)
params.add('dec13', value= dec13_value)
params.add('ratio12', value= a5, min=1.0)
params.add('ratio13', value= a6, min=1.0)
params.add('ud1',   value= a7, vary=False)#min=0.0,max=2.0)
params.add('ud2', value= a8, vary=False)#min=0.0,max=2.0)
params.add('ud3', value= a9, vary=False)#min=0.0,max=2.0)
params.add('bw', value= a10, min=0,max=1)

minner = Minimizer(triple_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
result = minner.minimize()
report_fit(result)

chi_sq_best = result.redchi
ra12_best = result.params['ra12'].value
dec12_best = result.params['dec12'].value
ra13_best = result.params['ra13'].value
dec13_best = result.params['dec13'].value
ratio12_best = result.params['ratio12'].value
ratio13_best = result.params['ratio13'].value
ud1_best = result.params['ud1'].value
ud2_best = result.params['ud2'].value
ud3_best = result.params['ud3'].value
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
best_params.add('ratio12', value= ratio12_best)
best_params.add('ratio13', value= ratio13_best)
best_params.add('ud1',   value= ud1_best)
best_params.add('ud2', value= ud2_best)
best_params.add('ud3', value= ud3_best)
best_params.add('bw', value=bw_best)

best_fit = np.around(np.array([ra12_best,dec12_best,ra13_best,dec13_best,ratio12_best,ratio13_best,ud1_best,ud2_best,ud3_best,bw_best,chi_sq_best]),decimals=4)

cp_model = []
for item1,item2 in zip(u_coords,v_coords):
    complex_vis = triple_model(best_params, item1, item2, eff_wave[0])
    phase = (np.angle(complex_vis[:,0])+np.angle(complex_vis[:,1])+np.angle(complex_vis[:,2]))*180/np.pi
    cp_model.append(phase)
cp_model=np.array(cp_model)

visphi_model = []
for item1,item2 in zip(ucoords,vcoords):     
    complex_vis = triple_model(best_params,item1, item2, eff_wave[0])
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
    complex_vis2 = triple_model(best_params,item1,item2,eff_wave[0])
    visibility2 = complex_vis2*np.conj(complex_vis2)
    vis2_model.append(visibility2.real)
vis2_model=np.array(vis2_model)

## plot results
with PdfPages("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_TRIPLE_search.pdf"%{"1":target_id,"2":date}) as pdf:
    
    ## first page - chisq grid
    plt.scatter(ra13_results, dec13_results, c=1/chi_sq, cmap=cm.inferno)
    plt.colorbar()
    plt.xlabel('d_RA (mas)')
    plt.ylabel('d_DE (mas)')
    plt.title('Best Fit - %s'%np.around(np.array([ra13_results[index],dec13_results[index]]),decimals=4))
    plt.gca().invert_xaxis()
    plt.axis('equal')
    pdf.savefig()
    plt.close()

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

##########################################################
## Now do error ellipses
print('Computing error ellipses for component 1')
size = 0.5
steps = 100
ra_grid = np.linspace(ra12_best-size,ra12_best+size,steps)
dec_grid = np.linspace(dec12_best-size,dec12_best+size,steps)

chi_sq = []
ra_results = []
dec_results = []

## draw plot
if plot_grid=='y':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()
for ra_try in tqdm(ra_grid):
    for dec_try in dec_grid:

        #create a set of Parameters
        params = [ra_try,dec_try,ra13_best,dec13_best,ratio12_best,ratio13_best,ud1_best,ud2_best,ud3_best,bw_best]

        #do fit, minimizer uses LM for least square fitting of model to data
        chi = triple_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
        raw_chi2 = np.nansum(chi**2)#/(len(np.ndarray.flatten(t3phi))-len(params))

        chi_sq.append(raw_chi2)
        ra_results.append(ra_try)
        dec_results.append(dec_try)

    if plot_grid=='y':
        ax.cla()
        ax.set_xlim(min(ra_grid),max(ra_grid))
        ax.set_ylim(min(dec_grid),max(dec_grid))
        ax.set_xlabel('d_RA (mas)')
        ax.set_ylabel('d_DE (mas)')
        ax.scatter(ra_results,dec_results,c=chi_sq,cmap=cm.inferno_r)
        ax.invert_xaxis()
        #plt.colorbar()
        plt.draw()
        plt.pause(0.001)
if plot_grid=='y':
    plt.show()
    plt.close()

ra_results = np.array(ra_results)
dec_results = np.array(dec_results)
chi_sq = np.array(chi_sq)

# write results
#report_fit(result)
index = np.argmin(chi_sq)

print('-----RESULTS-------')
print('ra12 = %s'%ra_results[index])
print('dec12 = %s'%dec_results[index])
print('rawchi12 = %s'%chi_sq[index])
print('-------------------')

## plot chisq surface grid
plt.scatter(ra_results, dec_results, c=chi_sq, cmap=cm.inferno_r)
plt.colorbar()
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.axis('equal')
plt.savefig('/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_chi2_comp2.pdf'%{"1":target_id,"2":date})
plt.close()

## isolate region where delta_chisq < 1
index_err = np.where(chi_sq < (chi_sq[index]+1) )
chi_err = chi_sq[index_err]
ra_err = ra_results[index_err]
dec_err = dec_results[index_err]

## fit an ellipse to the data
ra_mean = np.mean(ra_err)
dec_mean = np.mean(dec_err)
a,b,theta = ellipse_hull_fit(ra_err,dec_err,ra_mean,dec_mean)
angle = theta*180/np.pi

## want to measure east of north (different than python)
angle_new = 90-angle
if angle_new<0:
    angle_new=360+angle_new
ellipse_params = np.around(np.array([a,b,angle_new]),decimals=4)

ell = Ellipse(xy=(ra_mean,dec_mean),width=2*a,height=2*b,angle=angle,facecolor='lightgrey')
plt.gca().add_patch(ell)
plt.scatter(ra_err, dec_err, c=chi_err, cmap=cm.inferno_r,zorder=2)
plt.colorbar()
plt.title('a,b,thet=%s'%ellipse_params)
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig('/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_ellipse_comp1.pdf'%{"1":target_id,"2":date})
plt.close()

#############################################
print('Computing error ellipses for component 2')
size = 0.5
steps = 100
ra_grid = np.linspace(ra13_best-size,ra13_best+size,steps)
dec_grid = np.linspace(dec13_best-size,dec13_best+size,steps)

chi_sq = []
ra_results = []
dec_results = []

## draw plot
if plot_grid=='y':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()
for ra_try in tqdm(ra_grid):
    for dec_try in dec_grid:

        #create a set of Parameters
        params = [ra12_best,dec12_best,ra_try,dec_try,ratio12_best,ratio13_best,ud1_best,ud2_best,ud3_best,bw_best]

        #do fit, minimizer uses LM for least square fitting of model to data
        chi = triple_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
        raw_chi2 = np.nansum(chi**2)/(len(np.ndarray.flatten(t3phi))-len(params))

        chi_sq.append(raw_chi2)
        ra_results.append(ra_try)
        dec_results.append(dec_try)
    if plot_grid=='y':
        ax.cla()
        ax.set_xlim(min(ra_grid),max(ra_grid))
        ax.set_ylim(min(dec_grid),max(dec_grid))
        ax.set_xlabel('d_RA (mas)')
        ax.set_ylabel('d_DE (mas)')
        ax.scatter(ra_results,dec_results,c=chi_sq,cmap=cm.inferno_r)
        ax.invert_xaxis()
        #plt.colorbar()
        plt.draw()
        plt.pause(0.001)
if plot_grid=='y':
    plt.show()
    plt.close()

ra_results = np.array(ra_results)
dec_results = np.array(dec_results)
chi_sq = np.array(chi_sq)

# write results
#report_fit(result)
index = np.argmin(chi_sq)

print('-----RESULTS-------')
print('ra13 = %s'%ra_results[index])
print('dec13 = %s'%dec_results[index])
print('rawchi13 = %s'%chi_sq[index])
print('-------------------')

## plot chisq surface grid
plt.scatter(ra_results, dec_results, c=chi_sq, cmap=cm.inferno_r)
plt.colorbar()
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.axis('equal')
plt.savefig('/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_chi2_comp2.pdf'%{"1":target_id,"2":date})
plt.close()

## isolate region where delta_chisq < 1
index_err = np.where(chi_sq < (chi_sq[index]+1) )
chi_err = chi_sq[index_err]
ra_err = ra_results[index_err]
dec_err = dec_results[index_err]

## fit an ellipse to the data
ra_mean = np.mean(ra_err)
dec_mean = np.mean(dec_err)
a2,b2,theta2 = ellipse_hull_fit(ra_err,dec_err,ra_mean,dec_mean)
angle2 = theta2*180/np.pi

## want to measure east of north (different than python)
angle_new2 = 90-angle2
if angle_new2<0:
    angle_new2=360+angle_new2
ellipse_params2 = np.around(np.array([a2,b2,angle_new2]),decimals=4)

ell = Ellipse(xy=(ra_mean,dec_mean),width=2*a2,height=2*b2,angle=angle2,facecolor='lightgrey')
plt.gca().add_patch(ell)
plt.scatter(ra_err, dec_err, c=chi_err, cmap=cm.inferno_r,zorder=2)
plt.colorbar()
plt.title('a,b,thet=%s'%ellipse_params2)
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig('/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_ellipse_comp2.pdf'%{"1":target_id,"2":date})
plt.close()

txt_fit_cp = ('Best fit (ra12,dec12,ra13,dec13,ratio12,ratio13,ud1,ud2,ud2,bw,redchi2): %s'%best_fit)
## write results to a txt file
t = np.around(np.nanmedian(time_obs),4)
sep12,pa12 = np.around(cart2pol(best_fit[0],best_fit[1]),decimals=4)
sep13,pa13 = np.around(cart2pol(best_fit[2],best_fit[3]),decimals=4)
f = open("/Users/tgardne/ARMADA_epochs/%s/%s_%s_triple.txt"%(target_id,target_id,date),"w+")
f.write("# date mjd sep12(mas) pa12(Deg) sep13(mas) pa13(Deg) err_maj12(mas) err_min12(mas) err_pa12(deg) err_maj13(mas) err_min13(mas) err_pa13(deg)\r\n")
f.write("%s %s %s %s %s %s %s %s %s %s %s %s"%(date,t,sep12,pa12,ellipse_params[0],ellipse_params[1],ellipse_params[2],sep13,pa13,ellipse_params2[0],ellipse_params2[1],ellipse_params2[2]))
f.close()