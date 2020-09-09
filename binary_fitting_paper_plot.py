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
from read_oifits import read_chara,read_vlti
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
from ellipse_fitting import ellipse_bound,ellipse_fitting

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
def combined_minimizer(params,cp,cp_err,vphi,vphierr,v2,v2err,u_coord,v_coord,ucoord,vcoord,wl):

    diff=np.empty((0,len(wl)))

    if 'cphase' in flag:
        complex_vis = cvis_model(params,u_coord,v_coord,wl)
        cp_model = np.angle(complex_vis[:,:,0])+np.angle(complex_vis[:,:,1])+np.angle(complex_vis[:,:,2])
        cp_model = np.swapaxes(cp_model,0,1)
        cp = cp*np.pi/180
        cp_err = cp_err*np.pi/180
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
        vphi_err[vphi_err==0]=100
        if absolute=='y':
            vphi_diff = np.arctan2(np.sin(abs(vphi_data)-abs(vis_model)),np.cos(abs(vphi_data)-abs(vis_model))) / vphi_err
        else:
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

#method=input('VISPHI METHOD (dphase or visphi): ')
if dtype=='chara':
    method = 'dphase'
else:
    method = 'visphi'

interact = input('interactive session with data? (y/n): ')
exclude = input('exclude a telescope (e.g. E1): ')

reduction_params = input('ncoh int_time (notes): ').split(' ')
flag = input('fit to: vis2,cphase,dphase (separate with spaces): ').split(' ')
absolute = input('use absolute phase value (y/n)?')
#absolute='n'

## check directory exists for save files
save_dir="/Users/tgardne/ARMADA_epochs/%s/"%target_id
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

## get information from fits file
if dtype=='chara':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara(dir,target_id,interact,exclude)
if dtype=='vlti':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_vlti(dir,interact)
########################################################

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
steps = float(input('steps in grid: '))
a3 = float(input('flux ratio (f1/f2): '))
a4 = float(input('UD1 (mas): '))
a5 = float(input('UD2 (mas): '))
a6 = float(input('bw smearing (1/R): '))
vary_ratio = input('vary fratio on grid? (y/n) ')
plot_grid = input('plot grid (y/n)? ')

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
            params.add('ud1',   value= a4, vary=False)#min=0.0,max=2.0)
            params.add('ud2', value= a5, vary=False)
            params.add('bw', value=a6, vary=False)#min=0.0, max=0.1)
            minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
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
            chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
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
params.add('ud1',   value= best_params[3], min=0.0,max=2.0)
params.add('ud2', value= best_params[4], min=0.0,max=2.0)
params.add('bw', value=best_params[5], min=0.0, max=0.1)

minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
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

complex_vis = cvis_model(best_params,u_coords,v_coords, eff_wave[0])
cp_model = (np.angle(complex_vis[:,:,0])+np.angle(complex_vis[:,:,1])+np.angle(complex_vis[:,:,2]))*180/np.pi
cp_model = np.swapaxes(cp_model,0,1)

complex_vis = cvis_model(best_params,ucoords,vcoords, eff_wave[0])
visibility = np.angle(complex_vis)*180/np.pi
if method=='dphase':
    dphase = visibility[1:,:]-visibility[:-1,:]
    dphase = np.insert(dphase,dphase.shape[0],np.nan,axis=0)
else:
    dphase = visibility
visphi_model = np.swapaxes(dphase,0,1)

complex_vis2 = cvis_model(best_params,ucoords,vcoords,eff_wave[0])
visibility2 = complex_vis2*np.conj(complex_vis2)
vis2_model = visibility2.real
vis2_model = np.swapaxes(vis2_model,0,1)

## plot results    
## first page - chisq grid
plt.scatter(ra_results, dec_results, c=1/chi_sq, cmap=cm.inferno)
plt.colorbar()
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.title('Best Fit - %s'%np.around(np.array([ra_results[index],dec_results[index]]),decimals=4))
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_chi2map.pdf"%{"1":target_id,"2":date})
plt.close()

for t3,t3err,model,uc,vc in zip(t3phi,t3phierr,cp_model,u_coords,v_coords):
    b1 = np.sqrt(uc[0]**2+vc[0]**2)
    b2 = np.sqrt(uc[1]**2+vc[1]**2)
    b3 = np.sqrt(uc[2]**2+vc[2]**2)
    bl = max(b1,b2,b3) / eff_wave[0]
    plt.errorbar(bl,t3,yerr=t3err,fmt='.',zorder=1)
    plt.plot(bl,model,'+',color='r',zorder=2)
plt.xlabel('B/$\lambda$ x 10$^{-6}$')
plt.ylabel('T3PHI (deg)')
plt.savefig("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_t3phi.pdf"%{"1":target_id,"2":date})
plt.close()

### cp and dphase plots
### regroup data by measurements
#if dtype=='chara':
#    ntri=20
#    nbl=15
#if dtype=='vlti':
#    ntri=4
#    nbl=6
#
#n_cp = len(t3phi)/ntri
#n_vp = len(visphi)/nbl
#n_v2 = len(vis2)/nbl
#t3phi_plot = np.array(np.array_split(t3phi,n_cp))
#t3phierr_plot = np.array(np.array_split(t3phierr,n_cp))
#cp_model_plot = np.array_split(cp_model,n_cp)
#visphi_new_plot = np.array_split(visphi_new,n_vp)
#visphierr_plot = np.array_split(visphierr,n_vp)
#visphi_model_plot = np.array_split(visphi_model,n_vp)
#vis2_plot = np.array(np.array_split(vis2,n_v2))
#vis2err_plot = np.array(np.array_split(vis2err,n_v2))
#vis2_model_plot = np.array(np.array_split(vis2_model,n_v2))
#
### next pages - cp model fits
#index = np.arange(ntri)
#label_size = 4
#mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['ytick.labelsize'] = label_size
#if dtype=='chara':
#    fig,axs = plt.subplots(4,5,figsize=(10,7),facecolor='w',edgecolor='k')
#if dtype=='vlti':
#    fig,axs = plt.subplots(2,2,figsize=(10,7),facecolor='w',edgecolor='k')
#fig.subplots_adjust(hspace=0.5,wspace=.001)
#axs=axs.ravel()
#for t3data,t3errdata,modeldata in zip(t3phi_plot,t3phierr_plot,cp_model_plot):
#    for ind,y,yerr,m,tri in zip(index,t3data,t3errdata,modeldata,tels[:ntri]):
#        x=eff_wave[0]
#        axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
#        axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
#        axs[int(ind)].set_title(tri)
#fig.suptitle('%s Closure Phase'%target_id)
#fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
#fig.text(0.05, 0.5, 'CP (deg)', va='center', rotation='vertical')
#plt.savefig("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_t3phi.pdf"%{"1":target_id,"2":date})
#plt.close()
#
### next pages - visphi fits
#index = np.arange(nbl)
#label_size = 4
#mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['ytick.labelsize'] = label_size
#if dtype=='chara':
#    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
#if dtype=='vlti':
#    fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
#fig.subplots_adjust(hspace=0.5,wspace=.001)
#axs=axs.ravel()
#for visdata,viserrdata,modeldata in zip(visphi_new_plot,visphierr_plot,visphi_model_plot):
#    for ind,y,yerr,m,tri in zip(index,visdata,viserrdata,modeldata,vistels[:nbl]):
#        x=eff_wave[0]
#        axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
#        axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
#        axs[int(ind)].set_title(tri)
#fig.suptitle('%s VisPhi'%target_id)
#fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
#fig.text(0.05, 0.5, 'visphi (deg)', va='center', rotation='vertical')
#plt.savefig("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_visphi.pdf"%{"1":target_id,"2":date})
#plt.close()
#
### next pages - vis2 fits
#index = np.arange(nbl)
#label_size = 4
#mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['ytick.labelsize'] = label_size
#if dtype=='chara':
#    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
#if dtype=='vlti':
#    fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
#fig.subplots_adjust(hspace=0.5,wspace=.001)
#axs=axs.ravel()
#for visdata,viserrdata,modeldata in zip(vis2_plot,vis2err_plot,vis2_model_plot):
#    for ind,y,yerr,m,tri in zip(index,visdata,viserrdata,modeldata,vistels[:nbl]):
#        x=eff_wave[0]
#        axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
#        axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
#        axs[int(ind)].set_title(tri)
#fig.suptitle('%s Vis2'%target_id)
#fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
#fig.text(0.05, 0.5, 'vis2', va='center', rotation='vertical')
#plt.savefig("/Users/tgardne/ARMADA_epochs/%(1)s/%(1)s_%(2)s_vis2.pdf"%{"1":target_id,"2":date})
#plt.close()