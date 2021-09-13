######################################################################
## Tyler Gardner
##
## Bootstrap across time
## Dropping tel by tel
## (to search for systematic / pupil errors)
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

######################################################################
## LOAD DATA
######################################################################

## Ask the user which file contains the closure phases
#dtype = input('chara/vlti/chara_old? ')
dtype='chara'
date=input('Date for saved files (e.g. 2018Jul19):')
dir=input('Path to oifits directory:')
target_id=input('Target ID (e.g. HD_206901): ')

#method=input('VISPHI METHOD (dphase or visphi): ')
method='dphase'
#if dtype=='chara' or dtype==:
#    method = 'dphase'
#else:
#    method = 'visphi'

interact = input('interactive session with data? (y/n): ')
#interact = 'n'
#exclude = input('exclude a telescope (e.g. E1): ')
exclude = ''

#reduction_params = input('ncoh int_time (notes): ').split(' ')
reduction_params = '10 60'
#flag = input('fit to: vis2,cphase,dphase,vamp (separate with spaces): ').split(' ')
flag = ['cphase','dphase']
#absolute = input('use absolute phase value (y/n)?')
absolute='n'

## check directory exists for save files
save_dir="/Users/tgardne/ARMADA_epochs/bootstrap_quick_results/%s/"%target_id
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

## do polynomial dispersion fit for each visphi measurement
if method=='dphase':
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
else:
    visphi_new = visphi

######################################################################
## USER GIVES STARTING VALUES FOR FITS
######################################################################

sep_value=float(input('sep start (mas):'))
pa_value=float(input('PA start (deg):'))
a3 = float(input('flux ratio (f1/f2): '))

#a4 = float(input('UD1 (mas): '))
#a5 = float(input('UD2 (mas): '))
#a6 = float(input('bw smearing (1/R): '))
a4 = 0.5
a5 = 0.5
a6 = 0.005

#bootstrap_errors = input('bootstrap errors? (y/n) ')

dra = -sep_value*np.cos((90+pa_value)*np.pi/180)
ddec = sep_value*np.sin((90+pa_value)*np.pi/180)

######################################################################
## DO CHI2 FIT AT BEST POINT ON GRID FOR PLOT
######################################################################

## Do a chi2 fit for phases
params = Parameters()
params.add('ra',   value= dra)
params.add('dec', value= ddec)
params.add('ratio', value= a3, min=1.0)
params.add('ud1',   value= a4, vary=False)#min=0.0,max=2.0)
params.add('ud2', value= a5, vary=False)#min=0.0,max=2.0)
params.add('bw', value=a6, min=0.0, max=0.1)

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
with PdfPages("/Users/tgardne/ARMADA_epochs/bootstrap_quick_results/%(1)s/%(1)s_%(2)s_summary.pdf"%{"1":target_id,"2":date}) as pdf:

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

#if bootstrap_errors == 'y':
#############################################
## drop tel by tel
#############################################
xfit = []
yfit = []

for exclude in ['none','E1','W2','W1','S2','S1','E2']:

    # get information from fits file
    if dtype=='chara':
        t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara(dir,target_id,interact,exclude,bl_drop)
    if dtype=='vlti':
        t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_vlti(dir,interact)
    if dtype=='chara_old':
        t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara_old(dir,interact,exclude)
    ########################################################

    ## do polynomial dispersion fit for each visphi measurement
    if method=='dphase':
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
    else:
        visphi_new = visphi

    params = Parameters()
    params.add('ra',   value= ra_best)
    params.add('dec', value= dec_best)
    params.add('ratio', value= ratio_best, min=1.0)
    params.add('ud1',   value= ud1_best, vary=False)#min=0.0,max=2.0)
    params.add('ud2', value= ud2_best, vary=False)#min=0.0,max=2.0)
    params.add('bw', value=bw_best, vary=False)#min=0.0, max=0.1)

    minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0]),nan_policy='omit')
    result = minner.minimize()
    report_fit(result)

    ## write results to a txt file
    t = np.around(np.nanmedian(time_obs),4)
    sep,pa = np.around(cart2pol(result.params['ra'].value,result.params['dec'].value),decimals=4)
    f = open("/Users/tgardne/ARMADA_epochs/bootstrap_quick_results/%s/%s_%s_bootstrap_no_%s.txt"%(target_id,target_id,date,exclude),"w+")
    f.write("# date mjd sep(mas) pa(Deg) err_maj(mas) err_min(mas) err_pa(deg)\r\n")
    f.write("%s %s %s %s %s %s %s"%(date,t,sep,pa))
    f.close()

    xfit.append(result.params['ra'].value)
    yfit.append(result.params['dec'].value)

for x,y,label in zip(xfit,yfit,['none','E1','W2','W1','S2','S1','E2']):
    plt.scatter(x, y, zorder=2, label=label)
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.legend()
plt.savefig("/Users/tgardne/ARMADA_epochs/bootstrap_quick_results/%s/%s_%s_ellipse_boot.pdf"%(target_id,target_id,date))
plt.close()