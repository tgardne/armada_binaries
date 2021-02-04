######################################################################
## Tyler Gardner
##
## Do a grid search to fit binary model
## Designed for fitting to MIRCX/GRAVITY data
## Can fit to closure phase, dphase, vis2
## 
## Outputs pdf and txt files with plots, results
##
######################################################################

from chara_uvcalc import uv_calc
from binary_disks_vector import binary_disks_vector
from read_oifits import read_chara,read_vlti,read_chara_old
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
eachindex = lambda lst: range(len(lst))
import os
import matplotlib.cm as cm
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from tqdm import tqdm
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

## function which returns complex vis given params
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

## function which returns residual of model and data to be minimized
def combined_minimizer(params,cp,cp_err,vphi,vphierr,v2,v2err,vamp,vamperr,u_coord,v_coord,ucoord,vcoord,wl):

    diff=np.empty((0,len(wl)))

    if 'cphase' in flag:
        complex_vis = cvis_model(params,u_coord,v_coord,wl)
        cp_model = np.angle(complex_vis[:,:,0])+np.angle(complex_vis[:,:,1])+np.angle(complex_vis[:,:,2])
        cp_model = np.swapaxes(cp_model,0,1)
        cp = cp*np.pi/180
        cp_err = cp_err*np.pi/180
        cp[cp_err==0]=np.nan
        if absolute=='y':
            cp_diff = np.arctan2(np.sin(abs(cp)-abs(cp_model)),np.cos(abs(cp)-abs(cp_model)))/cp_err
        else:
            cp_diff = np.arctan2(np.sin(cp-cp_model),np.cos(cp-cp_model))/cp_err

        diff = np.append(diff,cp_diff,axis=0)

    if 'vis2' in flag:
        complex_vis2 = cvis_model(params,ucoord,vcoord,wl)
        visibility2 = complex_vis2*np.conj(complex_vis2)
        vis2_model = visibility2.real
        vis2_model = np.swapaxes(vis2_model,0,1)
        v2[v2err==0]=np.nan
        vis2_diff = (v2 - vis2_model) / v2err
        diff = np.append(diff,vis2_diff,axis=0)

    if 'dphase' in flag:
        complex_vis = cvis_model(params,ucoord,vcoord,wl)
        visibility = np.angle(complex_vis)
        if method=='dphase':
            dphase = visibility[1:,:]-visibility[:-1,:]
            dphase = np.insert(dphase,dphase.shape[0],np.nan,axis=0)
        else:
            dphase = visibility
        vis_model = dphase
        vis_model = np.swapaxes(vis_model,0,1)
        vis_model=np.array(vis_model)
        vphi_data = vphi*np.pi/180
        vphi_err = vphierr*np.pi/180
        vphi_data[vphi_err==0]=np.nan
        if absolute=='y':
            vphi_diff = np.arctan2(np.sin(abs(vphi_data)-abs(vis_model)),np.cos(abs(vphi_data)-abs(vis_model))) / vphi_err
        else:
            vphi_diff = np.arctan2(np.sin(vphi_data-vis_model),np.cos(vphi_data-vis_model)) / vphi_err
        diff = np.append(diff,vphi_diff,axis=0)

    if 'vamp' in flag:
        complex_vis = cvis_model(params,ucoord,vcoord,wl)
        visibility = np.absolute(complex_vis)
        #if method=='dphase':
        #    damp = visibility[1:,:]-visibility[:-1,:]
        #    damp = np.insert(damp,damp.shape[0],np.nan,axis=0)
        #else:
        damp = visibility
        vis_model = damp
        vis_model = np.swapaxes(vis_model,0,1)
        vis_model=np.array(vis_model)
        vamp[vamperr==0]=np.nan
        vamp_diff = (np.diff(vamp,axis=0) - np.diff(vis_model,axis=0)) / vamperr[:-1,:]
        diff = np.append(diff,vamp_diff,axis=0)
    
    diff = np.array(diff)
    return diff

## function for bootstrap
def bootstrap_data(params,t3,t3err,vp,vperr,v2,v2err,vamp,vamperr,ucc,vcc,uc,vc,wl):
    ra_results=[]
    dec_results=[]

    for i in tqdm(np.arange(1000)):
    
        r = np.random.randint(t3.shape[0],size=len(t3))
        t3phi_boot = t3[r,:]
        t3phierr_boot = t3err[r,:]
        u_coords_boot = ucc[r,:]
        v_coords_boot = vcc[r,:]

        visphi_boot = vp[r,:]
        visphierr_boot = vperr[r,:]
        visamp_boot =vamp[r,:]
        visamperr_boot =vamperr[r,:]
        ucoords_boot = uc[r,:]
        vcoords_boot = vc[r,:]

        vis2_boot = v2[r,:]
        vis2err_boot = v2err[r,:]
    
        t3phi_boot = t3phi_boot.reshape(int(t3phi_boot.shape[0])*int(t3phi_boot.shape[1]),t3phi_boot.shape[2])
        t3phierr_boot = t3phierr_boot.reshape(int(t3phierr_boot.shape[0])*int(t3phierr_boot.shape[1]),t3phierr_boot.shape[2])
        u_coords_boot = u_coords_boot.reshape(int(u_coords_boot.shape[0])*int(u_coords_boot.shape[1]),u_coords_boot.shape[2])
        v_coords_boot = v_coords_boot.reshape(int(v_coords_boot.shape[0])*int(v_coords_boot.shape[1]),v_coords_boot.shape[2])
        visphi_boot = visphi_boot.reshape(int(visphi_boot.shape[0])*int(visphi_boot.shape[1]),visphi_boot.shape[2])
        visphierr_boot = visphierr_boot.reshape(int(visphierr_boot.shape[0])*int(visphierr_boot.shape[1]),visphierr_boot.shape[2])
        visamp_boot = visamp_boot.reshape(int(visamp_boot.shape[0])*int(visamp_boot.shape[1]),visamp_boot.shape[2])
        visamperr_boot = visamperr_boot.reshape(int(visamperr_boot.shape[0])*int(visamperr_boot.shape[1]),visamperr_boot.shape[2])
        ucoords_boot = ucoords_boot.reshape(int(ucoords_boot.shape[0])*int(ucoords_boot.shape[1]))
        vcoords_boot = vcoords_boot.reshape(int(vcoords_boot.shape[0])*int(vcoords_boot.shape[1]))
        vis2_boot = vis2_boot.reshape(int(vis2_boot.shape[0])*int(vis2_boot.shape[1]),vis2_boot.shape[2])
        vis2err_boot = vis2err_boot.reshape(int(vis2err_boot.shape[0])*int(vis2err_boot.shape[1]),vis2err_boot.shape[2])

        #do fit, minimizer uses LM for least square fitting of model to data
        minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi_boot,t3phierr_boot,visphi_boot,visphierr_boot,vis2_boot,vis2err_boot,visamp_boot,visamperr_boot,u_coords_boot,v_coords_boot,ucoords_boot,vcoords_boot,wl),
                           nan_policy='omit')
        result = minner.minimize()

        ra_results.append(result.params['ra'].value)
        dec_results.append(result.params['dec'].value)
    ra_results=np.array(ra_results)
    dec_results=np.array(dec_results)
    return ra_results,dec_results

######################################################################
## LOAD DATA
######################################################################

## Ask the user which file contains the closure phases
dtype = input('chara/vlti/chara_old? ')
date=input('Date for saved files (e.g. 2018Jul19):')
dir=input('Path to oifits directory:')
target_id=input('Target ID (e.g. HD_206901): ')

method=input('VISPHI METHOD (dphase or visphi): ')
#if dtype=='chara' or dtype==:
#    method = 'dphase'
#else:
#    method = 'visphi'

interact = input('interactive session with data? (y/n): ')
exclude = input('exclude a telescope (e.g. E1): ')

reduction_params = input('ncoh int_time (notes): ').split(' ')
flag = input('fit to: vis2,cphase,dphase,vamp (separate with spaces): ').split(' ')
absolute = input('use absolute phase value (y/n)?')
#absolute='n'

## check directory exists for save files
save_dir="/Users/tgardne/ARMADA_epochs/%s/"%target_id
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

## Test dropping long baselines
bl_drop = input('Drop long baselines? (y/n): ')

## get information from fits file
if dtype=='chara':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara(dir,target_id,interact,exclude,bl_drop)
if dtype=='vlti':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_vlti(dir,interact)
if dtype=='chara_old':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara_old(dir,interact,exclude)
########################################################

## Split spectrum in half
#side = input('red or blue? ')
side = ''
if side=='blue':
    idx = int(eff_wave.shape[-1]/2)
    eff_wave = eff_wave[:,:idx]
    t3phi = t3phi[:,:idx]
    t3phierr = t3phierr[:,:idx]
    vis2 = vis2[:,:idx]
    vis2err = vis2err[:,:idx]
    visphi = visphi[:,:idx]
    visphierr = visphierr[:,:idx]
    visamp = visamp[:,:idx]
    visamperr = visamperr[:,:idx]
if side=='red':
    idx = int(eff_wave.shape[-1]/2)
    eff_wave = eff_wave[:,idx:]
    t3phi = t3phi[:,idx:]
    t3phierr = t3phierr[:,idx:]
    vis2 = vis2[:,idx:]
    vis2err = vis2err[:,idx:]
    visphi = visphi[:,idx:]
    visphierr = visphierr[:,idx:]
    visamp = visamp[:,idx:]
    visamperr = visamperr[:,idx:]

## do polynomial dispersion fit for each visphi measurement
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
## subtract dispersion (FIXME: should fit dispersion at each measurement)
visphi_new = visphi-dispersion

######################################################################
## USER GIVES STARTING VALUES FOR FITS
######################################################################

sep_value=float(input('sep start (mas):'))
pa_value=float(input('PA start (deg):'))
grid_size = float(input('search grid size (mas): '))
steps = int(input('steps in grid: '))
a3 = float(input('flux ratio (f1/f2): '))

a4 = float(input('UD1 (mas): '))
a5 = float(input('UD2 (mas): '))
a6 = float(input('bw smearing (1/R): '))
#a4 = 0.5
#a5 = 0.5
#a6 = 0.005

vary_ratio = input('vary fratio on grid? (y/n) ')
plot_grid = input('plot grid (y/n)? ')
#bootstrap_errors = input('bootstrap errors? (y/n) ')

dra = -sep_value*np.cos((90+pa_value)*np.pi/180)
ddec = sep_value*np.sin((90+pa_value)*np.pi/180)

ra_grid = np.linspace(dra-grid_size,dra+grid_size,steps)
dec_grid = np.linspace(ddec-grid_size,ddec+grid_size,steps)

######################################################################
## CHI2 MAPS
######################################################################

chi_sq = []
ra_results = []
dec_results = []
ratio_results = []
ud1_results = []
ud2_results = []
bw_results = []

## draw plot -- for now just show one grid map being plotted for a check (combined phases)
if plot_grid=='y':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

for ra_try in tqdm(ra_grid):
    for dec_try in dec_grid:

        if vary_ratio=='y':
            ## lmfit for varying params (slower)
            params = Parameters()
            params.add('ra',   value= ra_try, vary=False)
            params.add('dec', value= dec_try, vary=False)
            params.add('ratio', value= a3, min=1.0)
            params.add('ud1',   value= a4, min=0.0,max=3.0)
            params.add('ud2', value= a5, vary=False)
            params.add('bw', value=a6, vary=False)#min=0.0, max=0.1)
            minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
            result = minner.minimize()
            chi2 = result.chisqr
            ra_result = result.params['ra'].value
            dec_result = result.params['dec'].value
            ratio_result = result.params['ratio'].value
            ud1_result = result.params['ud1'].value
            ud2_result = result.params['ud2'].value
            bw_result = result.params['bw'].value
        else:
            ## fixed params (faster)
            params = [ra_try,dec_try,a3,a4,a5,a6]
            chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
            chi2 = np.nansum(chi**2)
            ra_result = ra_try
            dec_result = dec_try
            ratio_result = a3
            ud1_result = a4
            ud2_result = a5
            bw_result = a6

        chi_sq.append(chi2)
        ra_results.append(ra_result)
        dec_results.append(dec_result)
        ratio_results.append(ratio_result)
        ud1_results.append(ud1_result)
        ud2_results.append(ud2_result)
        bw_results.append(bw_result)

        #print(ra_result,dec_result,ratio_result,ud1_result,ud2_result,bw_result,chi2)
    if plot_grid=='y':
        ax.cla()
        ax.set_xlim(min(ra_grid),max(ra_grid))
        ax.set_ylim(min(dec_grid),max(dec_grid))
        ax.set_xlabel('d_RA (mas)')
        ax.set_ylabel('d_DE (mas)')
        ax.scatter(ra_results,dec_results,c=1/np.array(chi_sq),cmap=cm.inferno)
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

index = np.argmin(chi_sq)

## model params
best_params = [ra_results[index],dec_results[index],ratio_results[index],ud1_results[index],ud2_results[index],bw_results[index]]

######################################################################
## DO CHI2 FIT AT BEST POINT ON GRID FOR PLOT
######################################################################

## Do a chi2 fit for phases
params = Parameters()
params.add('ra',   value= best_params[0])
params.add('dec', value= best_params[1])
params.add('ratio', value= best_params[2], min=1.0)
params.add('ud1',   value= best_params[3], vary=False)#min=0.0,max=2.0)
params.add('ud2', value= best_params[4], vary=False)#min=0.0,max=2.0)
params.add('bw', value=best_params[5], min=0.0, max=0.1)

minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
result = minner.minimize()
report_fit(result)

chi_sq_best = result.redchi
ra_best = result.params['ra'].value
dec_best = result.params['dec'].value
ratio_best = result.params['ratio'].value
ud1_best = result.params['ud1'].value
ud2_best = result.params['ud2'].value
bw_best = result.params['bw'].value

######################################################################
## FORM MODEL FROM BEST FIT PARAMS TO PLOT AGAINST DATA
######################################################################

best_params = Parameters()
best_params.add('ra',value=ra_best)
best_params.add('dec',value=dec_best)
best_params.add('ratio',value=ratio_best)
best_params.add('ud1',value=ud1_best)
best_params.add('ud2',value=ud2_best)
best_params.add('bw',value=bw_best)

best_fit = np.around(np.array([ra_best,dec_best,ratio_best,ud1_best,ud2_best,bw_best,chi_sq_best]),decimals=4)

##### Just a grid plot ######
#plt.scatter(ra_results, dec_results, c=1/chi_sq, cmap=cm.inferno)
#plt.colorbar()
#plt.xlabel('d_RA (mas)')
#plt.ylabel('d_DE (mas)')
#plt.title('Best Fit - %s'%np.around(np.array([ra_results[index],dec_results[index]]),decimals=4))
#plt.gca().invert_xaxis()
#plt.axis('equal')
#plt.savefig("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_gridsearch.pdf"%{"1":target_id,"2":date})

complex_vis = cvis_model(best_params,u_coords,v_coords, eff_wave[0])
cp_model = (np.angle(complex_vis[:,:,0])+np.angle(complex_vis[:,:,1])+np.angle(complex_vis[:,:,2]))*180/np.pi
cp_model = np.swapaxes(cp_model,0,1)

complex_vis = cvis_model(best_params,ucoords,vcoords, eff_wave[0])
visibility = np.angle(complex_vis)*180/np.pi
visibility_amp = np.absolute(complex_vis)
if method=='dphase':
    dphase = visibility[1:,:]-visibility[:-1,:]
    dphase = np.insert(dphase,dphase.shape[0],np.nan,axis=0)
#    vamp = visibility_amp[1:,:]-visibility_amp[:-1,:]
#    vamp = np.insert(vamp,vamp.shape[0],np.nan,axis=0)
else:
    dphase = visibility
#    vamp = visibility_amp
vamp = visibility_amp
visphi_model = np.swapaxes(dphase,0,1)
visamp_model = np.swapaxes(vamp,0,1)

complex_vis2 = cvis_model(best_params,ucoords,vcoords,eff_wave[0])
visibility2 = complex_vis2*np.conj(complex_vis2)
vis2_model = visibility2.real
vis2_model = np.swapaxes(vis2_model,0,1)

## plot results
with PdfPages("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_summary.pdf"%{"1":target_id,"2":date}) as pdf:
    
    ## first page - chisq grid
    plt.scatter(ra_results, dec_results, c=1/chi_sq, cmap=cm.inferno)
    plt.colorbar()
    plt.xlabel('d_RA (mas)')
    plt.ylabel('d_DE (mas)')
    plt.title('Best Fit - %s'%np.around(np.array([ra_results[index],dec_results[index]]),decimals=4))
    plt.gca().invert_xaxis()
    plt.axis('equal')
    pdf.savefig()
    plt.close()

    ## regroup data by measurements
    if dtype=='chara' or dtype=='chara_old':
        ntri=20
        nbl=15
    if dtype=='vlti':
        ntri=4
        nbl=6

    n_cp = len(t3phi)/ntri
    n_vp = len(visphi)/nbl
    n_v2 = len(vis2)/nbl
    t3phi_plot = np.array(np.array_split(t3phi,n_cp))
    t3phierr_plot = np.array(np.array_split(t3phierr,n_cp))
    cp_model_plot = np.array_split(cp_model,n_cp)
    visphi_new_plot = np.array_split(visphi_new,n_vp)
    visphierr_plot = np.array_split(visphierr,n_vp)
    visphi_model_plot = np.array_split(visphi_model,n_vp)
    visamp_plot = np.array_split(np.diff(visamp,axis=0),n_vp)
    visamperr_plot = np.array_split(visamperr[:-1,:],n_vp)
    visamp_model_plot = np.array_split(np.diff(visamp_model,axis=0),n_vp)
    vis2_plot = np.array(np.array_split(vis2,n_v2))
    vis2err_plot = np.array(np.array_split(vis2err,n_v2))
    vis2_model_plot = np.array(np.array_split(vis2_model,n_v2))
    
    ## next pages - cp model fits
    index = np.arange(ntri)
    for t3data,t3errdata,modeldata in zip(t3phi_plot,t3phierr_plot,cp_model_plot):

        label_size = 4
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        if dtype=='chara' or dtype=='chara_old':
            fig,axs = plt.subplots(4,5,figsize=(10,7),facecolor='w',edgecolor='k')
        if dtype=='vlti':
            fig,axs = plt.subplots(2,2,figsize=(10,7),facecolor='w',edgecolor='k')
        fig.subplots_adjust(hspace=0.5,wspace=.001)
        axs=axs.ravel()

        for ind,y,yerr,m,tri in zip(index,t3data,t3errdata,modeldata,tels[:ntri]):
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
    index = np.arange(nbl)
    for visdata,viserrdata,modeldata in zip(visphi_new_plot,visphierr_plot,visphi_model_plot):

        label_size = 4
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        if dtype=='chara' or dtype=='chara_old':
            fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
        if dtype=='vlti':
            fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
        fig.subplots_adjust(hspace=0.5,wspace=.001)
        axs=axs.ravel()

        for ind,y,yerr,m,tri in zip(index,visdata,viserrdata,modeldata,vistels[:nbl]):
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
    index = np.arange(nbl)
    for visdata,viserrdata,modeldata in zip(vis2_plot,vis2err_plot,vis2_model_plot):

        label_size = 4
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        if dtype=='chara' or dtype=='chara_old':
            fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
        if dtype=='vlti':
            fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
        fig.subplots_adjust(hspace=0.5,wspace=.001)
        axs=axs.ravel()

        for ind,y,yerr,m,tri in zip(index,visdata,viserrdata,modeldata,vistels[:nbl]):
            x=eff_wave[0]
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
            axs[int(ind)].set_title(str(tri))

        fig.suptitle('%s Vis2'%target_id)
        fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
        fig.text(0.05, 0.5, 'vis2', va='center', rotation='vertical')
        pdf.savefig()
        plt.close()

    ## next pages - visamp fits
    index = np.arange(nbl)
    for visdata,viserrdata,modeldata in zip(visamp_plot,visamperr_plot,visamp_model_plot):

        label_size = 4
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        if dtype=='chara' or dtype=='chara_old':
            fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
        if dtype=='vlti':
            fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
        fig.subplots_adjust(hspace=0.5,wspace=.001)
        axs=axs.ravel()

        for ind,y,yerr,m,tri in zip(index,visdata,viserrdata,modeldata,vistels[:nbl]):
            x=eff_wave[0]
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
            axs[int(ind)].set_title(str(tri))

        fig.suptitle('%s VisAmp'%target_id)
        fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
        fig.text(0.05, 0.5, 'visamp', va='center', rotation='vertical')
        pdf.savefig()
        plt.close()

    ## next page - reduction and fit parameters
    textfig = plt.figure(figsize=(11.69,8.27))
    textfig.clf()
    txt_fit_cp = ('Best fit (ra,dec,ratio,ud1,ud2,bw,redchi2): %s'%best_fit)
    reduction = ('Reduction params (ncoh,nbs,ncs,int):%s'%reduction_params)
    textfig.text(0.5,0.75,txt_fit_cp,size=12,ha="center")
    textfig.text(0.5,0.5,reduction,size=12,ha="center")
    textfig.text(0.5,0.25,flag,size=12,ha='center')
    pdf.savefig()
    plt.close()

###########################################################
### Now do errors
###########################################################
print('Computing errors from CHI2 SURFACE')
size = 0.5
steps = 300
ra_grid = np.linspace(ra_best-size,ra_best+size,steps)
dec_grid = np.linspace(dec_best-size,dec_best+size,steps)
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
        params = [ra_try,dec_try,ratio_best,ud1_best,ud2_best,bw_best]
        #do fit, minimizer uses LM for least square fitting of model to data
        chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
        red_chi2 = np.nansum(chi**2)/(len(np.ndarray.flatten(t3phi))-len(params))
        chi_sq.append(red_chi2)
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
## plot chisq surface grid
plt.scatter(ra_results, dec_results, c=chi_sq, cmap=cm.inferno_r)
plt.colorbar()
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig("/Users/tgardne/ARMADA_epochs/%s/%s_%s_chisq.pdf"%(target_id,target_id,date))
plt.close()
## isolate region where delta_chisq < 1
params = [ra_best,dec_best,ratio_best,ud1_best,ud2_best,bw_best]
chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
chi2_best = np.nansum(chi**2)/(len(np.ndarray.flatten(t3phi))-len(params))

index_err = np.where(chi_sq < (chi2_best+1) )
chi_err = chi_sq[index_err]
ra_err = ra_results[index_err]
dec_err = dec_results[index_err]
## save arrays
#np.save('ra_err',ra_err)
#np.save('dec_err',dec_err)
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
plt.savefig("/Users/tgardne/ARMADA_epochs/%s/%s_%s_ellipse.pdf"%(target_id,target_id,date))
plt.close()

## write results to a txt file
t = np.around(np.nanmedian(time_obs),4)
sep,pa = np.around(cart2pol(best_fit[0],best_fit[1]),decimals=4)
f = open("/Users/tgardne/ARMADA_epochs/%s/%s_%s_chi2err.txt"%(target_id,target_id,date),"w+")
f.write("# date mjd sep(mas) pa(Deg) err_maj(mas) err_min(mas) err_pa(deg)\r\n")
f.write("%s %s %s %s %s %s %s"%(date,t,sep,pa,ellipse_params[0],ellipse_params[1],ellipse_params[2]))
f.close()

#if bootstrap_errors == 'y':
#####################
## Split data by time
#####################
print('Computing errors with BOOTSTRAP')
print('Shape of t3phi = ',t3phi.shape)
print('Shape of vis2 = ',vis2.shape)
num = 20
num2 = 15
t3phi = t3phi.reshape(int(len(t3phi)/num),num,len(t3phi[0]))
t3phierr = t3phierr.reshape(int(len(t3phierr)/num),num,len(t3phierr[0]))
vis2 = vis2.reshape(int(len(vis2)/num2),num2,len(vis2[0]))
vis2err = vis2err.reshape(int(len(vis2err)/num2),num2,len(vis2err[0]))
visphi_new = visphi_new.reshape(int(len(visphi_new)/num2),num2,len(visphi_new[0]))
visphierr = visphierr.reshape(int(len(visphierr)/num2),num2,len(visphierr[0]))
visamp = visamp.reshape(int(len(visamp)/num2),num2,len(visamp[0]))
visamperr = visamperr.reshape(int(len(visamperr)/num2),num2,len(visamperr[0]))
tels = tels.reshape(int(len(tels)/num),num,len(tels[0]))
vistels = vistels.reshape(int(len(vistels)/num2),num2,len(vistels[0]))
u_coords = u_coords.reshape(int(len(u_coords)/num),num,len(u_coords[0]))
v_coords = v_coords.reshape(int(len(v_coords)/num),num,len(v_coords[0]))
ucoords = ucoords.reshape(int(len(ucoords)/num2),num2,1)
vcoords = vcoords.reshape(int(len(vcoords)/num2),num2,1)
print('New shape of t3phi = ',t3phi.shape)
print('New shape of vis2 = ',vis2.shape)
params = Parameters()
params.add('ra',   value= ra_best)
params.add('dec', value= dec_best)
params.add('ratio', value= ratio_best, min=1.0)
params.add('ud1',   value= ud1_best, vary=False)#min=0.0,max=2.0)
params.add('ud2', value= ud2_best, vary=False)#min=0.0,max=2.0)
params.add('bw', value=bw_best, vary=False)#min=0.0, max=0.1)
ra_boot,dec_boot = bootstrap_data(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
ra_mean = np.mean(ra_boot)
dec_mean = np.mean(dec_boot)
a,b,theta = ellipse_hull_fit(ra_boot,dec_boot,ra_mean,dec_mean)
angle = theta*180/np.pi
## want to measure east of north (different than python)
angle_new = 90-angle
if angle_new<0:
    angle_new=360+angle_new
ellipse_params = np.around(np.array([a,b,angle_new]),decimals=4)
ell = Ellipse(xy=(ra_mean,dec_mean),width=2*a,height=2*b,angle=angle,facecolor='lightgrey')
plt.gca().add_patch(ell)
plt.scatter(ra_boot, dec_boot, zorder=2)
plt.title('a,b,thet=%s'%ellipse_params)
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig("/Users/tgardne/ARMADA_epochs/%s/%s_%s_ellipse_boot.pdf"%(target_id,target_id,date))
plt.close()

## write results to a txt file
t = np.around(np.nanmedian(time_obs),4)
sep,pa = np.around(cart2pol(best_fit[0],best_fit[1]),decimals=4)
f = open("/Users/tgardne/ARMADA_epochs/%s/%s_%s_bootstrap.txt"%(target_id,target_id,date),"w+")
f.write("# date mjd sep(mas) pa(Deg) err_maj(mas) err_min(mas) err_pa(deg)\r\n")
f.write("%s %s %s %s %s %s %s"%(date,t,sep,pa,ellipse_params[0],ellipse_params[1],ellipse_params[2]))
f.close()