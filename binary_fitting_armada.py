######################################################################
## Tyler Gardner
##
## Do a grid search to fit binary model
## Designed for fitting to MIRCX/GRAVITY data
## Can fit to closure phase, dphase, vis2, visamp
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
from scipy.signal import medfilt

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
def combined_minimizer(params,cp,cp_err,vphi,vphierr,v2,v2err,vamp,vamperr,u_coord,v_coord,ucoord,vcoord,wl,vtels,flag):

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
        dphase = visibility[1:,:]-visibility[:-1,:]
        vis_model = np.insert(dphase,dphase.shape[0],np.nan,axis=0)
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
    
    if 'visphi' in flag:
        complex_vis = cvis_model(params,ucoord,vcoord,wl)
        visibility = np.angle(complex_vis)
        vis_model = np.swapaxes(visibility,0,1)

        vphi_data = vphi*np.pi/180
        vphi_err = vphierr*np.pi/180
        vphi_data[vphi_err==0]=np.nan
        if absolute=='y':
            vphi_diff = np.arctan2(np.sin(abs(vphi_data)-abs(vis_model)),np.cos(abs(vphi_data)-abs(vis_model)))# / vphi_err
        else:
            vphi_diff = np.arctan2(np.sin(vphi_data-vis_model),np.cos(vphi_data-vis_model))# / vphi_err

        ## Subtract order 3 poly coefficients from residuals
        for nmeas in np.arange(vphi_diff.shape[0]):
            xdata = (wl-np.median(wl)) / (np.max(wl-np.min(wl)))
            idx = np.where(np.isfinite(vphi_diff[nmeas,:]))
            unit = xdata[idx]*0+1
            A = np.array([unit,xdata[idx],xdata[idx]**2,xdata[idx]**3])

            result = np.linalg.lstsq(A.T,vphi_diff[nmeas,:][idx],rcond=-1)
            vphi_diff[nmeas,:] -= (result[0][0]+result[0][1]*xdata+result[0][2]*xdata**2+result[0][3]*xdata**3)

        vphi_diff/=vphi_err
        diff = np.append(diff,vphi_diff,axis=0)

    if 'visamp' in flag:

        complex_vis = cvis_model(params,ucoord,vcoord,wl)
        visibility = np.absolute(complex_vis)

        ## correct for slope and offset
        try:    
            v1 = params['v1']
            v2 = params['v2']
            v3 = params['v3']
            v4 = params['v4']
            v5 = params['v5']
            v6 = params['v6'] 
            f1 = params['f1']
            f2 = params['f2']
            f3 = params['f3']
            f4 = params['f4']
            f5 = params['f5']
            f6 = params['f6']
        except: 
            v1 = params[6]
            v2 = params[7]
            v3 = params[8]
            v4 = params[9]
            v5 = params[10]
            v6 = params[11] 
            f1 = params[12]
            f2 = params[13]
            f3 = params[14]
            f4 = params[15]
            f5 = params[16]
            f6 = params[17]

        vis_model = visibility
        vis_model = np.swapaxes(vis_model,0,1)
        vis_model=np.array(vis_model)
        #print(vis_model.shape)

        idx1 = np.where(np.all(vtels==vtels[0],axis=1))
        idx2 = np.where(np.all(vtels==vtels[1],axis=1))
        idx3 = np.where(np.all(vtels==vtels[2],axis=1))
        idx4 = np.where(np.all(vtels==vtels[3],axis=1))
        idx5 = np.where(np.all(vtels==vtels[4],axis=1))
        idx6 = np.where(np.all(vtels==vtels[5],axis=1))

        vis_model[idx1]*=(v1+f1*((wl-np.median(wl))/(max(wl)-min(wl))))
        vis_model[idx2]*=(v2+f2*((wl-np.median(wl))/(max(wl)-min(wl))))
        vis_model[idx3]*=(v3+f3*((wl-np.median(wl))/(max(wl)-min(wl))))
        vis_model[idx4]*=(v4+f4*((wl-np.median(wl))/(max(wl)-min(wl))))
        vis_model[idx5]*=(v5+f5*((wl-np.median(wl))/(max(wl)-min(wl))))
        vis_model[idx6]*=(v6+f6*((wl-np.median(wl))/(max(wl)-min(wl))))

        vamp[vamperr==0]=np.nan
        vamp_diff = (vamp - vis_model) / vamperr
        diff = np.append(diff,vamp_diff,axis=0)
    
    diff = np.array(diff)
    return diff

## function for bootstrap
def bootstrap_data(params,t3,t3err,vp,vperr,v2,v2err,vamp,vamperr,ucc,vcc,uc,vc,wl):
    ra_results=[]
    dec_results=[]

    #ra_start = params['ra'].value
    #dec_start = params['dec'].value

    for i in tqdm(np.arange(1000)):

        #print(params['ra'].value,params['dec'].value)

        ## vary starting position slightly
        #params['ra'].value = np.random.uniform(0.999*ra_start,1.001*ra_start)
        #params['dec'].value = np.random.uniform(0.999*dec_start,1.001*dec_start)
    
        #print(params['ra'].value,params['dec'].value)

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
        minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi_boot,t3phierr_boot,visphi_boot,visphierr_boot,vis2_boot,vis2err_boot,visamp_boot,visamperr_boot,u_coords_boot,v_coords_boot,ucoords_boot,vcoords_boot,wl,fitting_vars),
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

interact = input('interactive session with data? (y/n): ')
exclude = input('exclude a telescope (e.g. E1): ')

reduction_params = input('ncoh int_time (notes): ').split(' ')
fitting_vars = input('fit to: vis2,cphase,dphase,visphi,visamp (separate with spaces): ').split(' ')

## UNCOMMENT THIS FOR 1:1 binaries with FLIPS
absolute = input('use absolute phase value (y/n)?')
#absolute='n'

## check directory exists for save files
save_dir="/Users/tgardner/ARMADA_epochs/%s/"%target_id
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

## Uncomment this if you want to drop LONG baselines
#bl_drop = input('Drop long baselines? (y/n): ')
bl_drop='n'

## get information from fits file
if dtype=='chara':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,az = read_chara(dir,target_id,interact,exclude,bl_drop)
if dtype=='vlti':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,flux,fluxerr = read_vlti(dir,interact)
if dtype=='chara_old':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara_old(dir,interact,exclude)
########################################################
print("T3phi shape = ", t3phi.shape)
print("Vis2 shape = ", visphi.shape)

edge_purge = input("Purge edge channels (y / [n]): ")
if edge_purge == 'y':
    print("PURGING EDGE CHANNELS")
    t3phi = t3phi[:,1:-1]
    t3phierr = t3phierr[:,1:-1]
    vis2 = vis2[:,1:-1]
    vis2err = vis2err[:,1:-1]
    visphi = visphi[:,1:-1]
    visphierr = visphierr[:,1:-1]
    visamp = visamp[:,1:-1]
    visamperr = visamperr[:,1:-1]
    eff_wave = eff_wave[:,1:-1]
    print("New T3phi shape = ", t3phi.shape)
    print("New Vis2 shape = ", visphi.shape)

### Split spectrum in half
##side = input('red or blue? ')
#side = ''
#if side=='blue':
#    idx = int(eff_wave.shape[-1]/2)
#    eff_wave = eff_wave[:,:idx]
#    t3phi = t3phi[:,:idx]
#    t3phierr = t3phierr[:,:idx]
#    vis2 = vis2[:,:idx]
#    vis2err = vis2err[:,:idx]
#    visphi = visphi[:,:idx]
#    visphierr = visphierr[:,:idx]
#    visamp = visamp[:,:idx]
#    visamperr = visamperr[:,:idx]
#if side=='red':
#    idx = int(eff_wave.shape[-1]/2)
#    eff_wave = eff_wave[:,idx:]
#    t3phi = t3phi[:,idx:]
#    t3phierr = t3phierr[:,idx:]
#    vis2 = vis2[:,idx:]
#    vis2err = vis2err[:,idx:]
#    visphi = visphi[:,idx:]
#    visphierr = visphierr[:,idx:]
#    visamp = visamp[:,idx:]
#    visamperr = visamperr[:,idx:]

## do polynomial dispersion fit for each dphase measurement
if 'dphase' in fitting_vars:
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

correct = input('Correct t3phi? (y/[n]): ')
if correct == 'y':
    correction_file = input('File with corrected t3phi: ')
    t3phi_corrected = np.load(correction_file)
    print(t3phi.shape)
    print(t3phi_corrected.shape)
    print('Using corrected t3phi')
    t3phi = t3phi_corrected

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
#if dtype == "chara":
#    a6 = 0.005
#else:
#    a6 = 0.00001
a6 = float(input('bw smearing (1/R): '))

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

## medfilt HR VLTI to get rid of outliers
if dtype=='vlti':
    nloop = 0
    while nloop<3:
        t3phi_filtered = medfilt(t3phi,(1,11))
        vis2_filtered = medfilt(vis2,(1,11))
        visphi_new_filtered = medfilt(visphi_new,(1,11))
        visamp_filtered = medfilt(visamp,(1,11))

        t3phi_resid = t3phi-t3phi_filtered
        vis2_resid = vis2-vis2_filtered
        visphi_new_resid = visphi_new-visphi_new_filtered
        visamp_resid = visamp-visamp_filtered

        std_t3phi = np.nanstd(t3phi_resid)
        std_vis2 = np.nanstd(vis2_resid)
        std_visphi_new = np.nanstd(visphi_new_resid)
        std_visamp = np.nanstd(visamp_resid)

        idx_t3phi = np.where(abs(t3phi_resid)>(3*std_t3phi))
        idx_vis2 = np.where(abs(vis2_resid)>(3*std_vis2))
        idx_visphi_new = np.where(abs(visphi_new_resid)>(3*std_visphi_new))
        idx_visamp = np.where(abs(visamp_resid)>(3*std_visamp))

        t3phi[idx_t3phi]=np.nan
        vis2[idx_vis2]=np.nan
        visphi_new[idx_visphi_new]=np.nan
        visamp[idx_visamp]=np.nan

        nloop+=1

chi_sq = []
redchi_sq = []
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

telnames = vistels[0:6]
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

            params.add('v1',value=1,vary=False)
            params.add('v2',value=1,vary=False)
            params.add('v3',value=1,vary=False)
            params.add('v4',value=1,vary=False)
            params.add('v5',value=1,vary=False)
            params.add('v6',value=1,vary=False)
            params.add('f1',value=0,vary=False)
            params.add('f2',value=0,vary=False)
            params.add('f3',value=0,vary=False)
            params.add('f4',value=0,vary=False)
            params.add('f5',value=0,vary=False)
            params.add('f6',value=0,vary=False)

            if dtype=='vlti':
                minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi[:,::50],t3phierr[:,::50],visphi_new[:,::50],visphierr[:,::50],vis2[:,::50],vis2err[:,::50],visamp[:,::50],visamperr[:,::50],u_coords,v_coords,ucoords,vcoords,eff_wave[0][::50],vistels,fitting_vars),nan_policy='omit')
                result = minner.minimize()
            else:
                minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels,fitting_vars),nan_policy='omit')
                result = minner.minimize()

            chi2 = result.chisqr
            redchi2 = result.redchi
            ra_result = result.params['ra'].value
            dec_result = result.params['dec'].value
            ratio_result = result.params['ratio'].value
            ud1_result = result.params['ud1'].value
            ud2_result = result.params['ud2'].value
            bw_result = result.params['bw'].value
        else:
            ## fixed params (faster)
            params = [ra_try,dec_try,a3,a4,a5,a6,1,1,1,1,1,1,0,0,0,0,0,0]
            if dtype=='vlti':
                chi = combined_minimizer(params,t3phi[:,::50],t3phierr[:,::50],visphi_new[:,::50],visphierr[:,::50],vis2[:,::50],vis2err[:,::50],visamp[:,::50],visamperr[:,::50],u_coords,v_coords,ucoords,vcoords,eff_wave[0][::50],vistels,fitting_vars)
            else:
                chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels,fitting_vars)
            chi2 = np.nansum(chi**2)
            redchi2 = np.nansum(chi**2) / (len(np.ndarray.flatten(t3phi))-len(params))
            #print(chi2)
            ra_result = ra_try
            dec_result = dec_try
            ratio_result = a3
            ud1_result = a4
            ud2_result = a5
            bw_result = a6

        chi_sq.append(chi2)
        redchi_sq.append(redchi2)
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
ratio_results = np.array(ratio_results)
chi_sq = np.array(chi_sq)
redchi_sq = np.array(redchi_sq)

np.save("/Users/tgardner/ARMADA_epochs/%s/%s_%s_ra.npy"%(target_id,target_id,date),ra_results)
np.save("/Users/tgardner/ARMADA_epochs/%s/%s_%s_dec.npy"%(target_id,target_id,date),dec_results)
np.save("/Users/tgardner/ARMADA_epochs/%s/%s_%s_ratio.npy"%(target_id,target_id,date),ratio_results)
np.save("/Users/tgardner/ARMADA_epochs/%s/%s_%s_chisq.npy"%(target_id,target_id,date),chi_sq)
np.save("/Users/tgardner/ARMADA_epochs/%s/%s_%s_redchisq.npy"%(target_id,target_id,date),redchi_sq)

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

if 'visamp' in fitting_vars:
    params.add('v1',value=0)
    params.add('v2',value=0)
    params.add('v3',value=0)
    params.add('v4',value=0)
    params.add('v5',value=0)
    params.add('v6',value=0)
    params.add('f1',value=0)#,vary=False)
    params.add('f2',value=0)#,vary=False)
    params.add('f3',value=0)#,vary=False)
    params.add('f4',value=0)#,vary=False)
    params.add('f5',value=0)#,vary=False)
    params.add('f6',value=0)#,vary=False)
else:
    params.add('v1',value=1,vary=False)
    params.add('v2',value=1,vary=False)
    params.add('v3',value=1,vary=False)
    params.add('v4',value=1,vary=False)
    params.add('v5',value=1,vary=False)
    params.add('v6',value=1,vary=False)
    params.add('f1',value=0,vary=False)
    params.add('f2',value=0,vary=False)
    params.add('f3',value=0,vary=False)
    params.add('f4',value=0,vary=False)
    params.add('f5',value=0,vary=False)
    params.add('f6',value=0,vary=False)

minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels,fitting_vars),nan_policy='omit')
result = minner.minimize()
report_fit(result)

redchi_sq_best = result.redchi
chi_sq_best = result.chisqr
nfree = result.nfree
ra_best = result.params['ra'].value
dec_best = result.params['dec'].value
ratio_best = result.params['ratio'].value
ud1_best = result.params['ud1'].value
ud2_best = result.params['ud2'].value
bw_best = result.params['bw'].value

v1_best = result.params['v1'].value
v2_best = result.params['v2'].value
v3_best = result.params['v3'].value
v4_best = result.params['v4'].value
v5_best = result.params['v5'].value
v6_best = result.params['v6'].value
f1_best = result.params['f1'].value
f2_best = result.params['f2'].value
f3_best = result.params['f3'].value
f4_best = result.params['f4'].value
f5_best = result.params['f5'].value
f6_best = result.params['f6'].value

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

best_params.add('v1',value=v1_best)
best_params.add('v2',value=v2_best)
best_params.add('v3',value=v3_best)
best_params.add('v4',value=v4_best)
best_params.add('v5',value=v5_best)
best_params.add('v6',value=v6_best)
best_params.add('f1',value=f1_best)
best_params.add('f2',value=f2_best)
best_params.add('f3',value=f3_best)
best_params.add('f4',value=f4_best)
best_params.add('f5',value=f5_best)
best_params.add('f6',value=f6_best)

best_fit = np.around(np.array([ra_best,dec_best,ratio_best,ud1_best,ud2_best,bw_best,redchi_sq_best]),decimals=4)

##### Just a grid plot ######
#plt.scatter(ra_results, dec_results, c=1/chi_sq, cmap=cm.inferno)
#plt.colorbar()
#plt.xlabel('d_RA (mas)')
#plt.ylabel('d_DE (mas)')
#plt.title('Best Fit - %s'%np.around(np.array([ra_results[index],dec_results[index]]),decimals=4))
#plt.gca().invert_xaxis()
#plt.axis('equal')
#plt.savefig("/Users/tgardner/ARMADA_epochs/%(1)s/%(1)s_%(2)s_gridsearch.pdf"%{"1":target_id,"2":date})

complex_vis = cvis_model(best_params,u_coords,v_coords, eff_wave[0])
cp_model = (np.angle(complex_vis[:,:,0])+np.angle(complex_vis[:,:,1])+np.angle(complex_vis[:,:,2]))*180/np.pi
cp_model = np.swapaxes(cp_model,0,1)

complex_vis = cvis_model(best_params,ucoords,vcoords, eff_wave[0])
visibility = np.angle(complex_vis)*180/np.pi
visibility_amp = np.absolute(complex_vis)
if 'dphase' in fitting_vars:
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

## add dispersion back into visphi
#if 'visphi' in fitting_vars:
    #if absolute=='y':
    #    vphi_diff = np.arctan2(np.sin((abs(visphi_new)-abs(visphi_model))*np.pi/180),np.cos((abs(visphi_new)-abs(visphi_model))*np.pi/180))
    #else:
    #    vphi_diff = np.arctan2(np.sin((visphi_new-visphi_model)*np.pi/180),np.cos((visphi_new-visphi_model)*np.pi/180))
    #visphi_model+=add_dispersion(eff_wave[0],vphi_diff)
    #ll = (eff_wave[0]-np.median(eff_wave[0]))/(max(eff_wave[0])-min(eff_wave[0]))
    #for idx in np.arange(visphi_model.shape[0]):
    #    visphi_model[idx,:]+=(vplist[idx] + fplist[idx]*ll + ffplist[idx]*ll**2 + fffplist[idx]*ll**3)

## Add order 3 poly coefficients from residuals to model
if 'visphi' in fitting_vars:
    if absolute=='y':
        vphi_diff = np.arctan2(np.sin((abs(visphi_new)-abs(visphi_model))*np.pi/180),np.cos((abs(visphi_new)-abs(visphi_model))*np.pi/180))*180/np.pi
    else:
        vphi_diff = np.arctan2(np.sin((visphi_new-visphi_model)*np.pi/180),np.cos((visphi_new-visphi_model)*np.pi/180))*180/np.pi

    for nmeas in np.arange(vphi_diff.shape[0]):
        xdata = (eff_wave[0]-np.median(eff_wave[0])) / (np.max(eff_wave[0]-np.min(eff_wave[0])))
        idx = np.where(np.isfinite(vphi_diff[nmeas,:]))
        unit = xdata[idx]*0+1
        A = np.array([unit,xdata[idx],xdata[idx]**2,xdata[idx]**3])
        result = np.linalg.lstsq(A.T,vphi_diff[nmeas,:][idx],rcond=-1)
        visphi_model[nmeas,:] += (result[0][0]+result[0][1]*xdata+result[0][2]*xdata**2+result[0][3]*xdata**3)

#print(vis_model.shape)
idx1 = np.where(np.all(vistels==vistels[0],axis=1))
idx2 = np.where(np.all(vistels==vistels[1],axis=1))
idx3 = np.where(np.all(vistels==vistels[2],axis=1))
idx4 = np.where(np.all(vistels==vistels[3],axis=1))
idx5 = np.where(np.all(vistels==vistels[4],axis=1))
idx6 = np.where(np.all(vistels==vistels[5],axis=1))
ll = (eff_wave[0]-np.median(eff_wave[0]))/(max(eff_wave[0])-min(eff_wave[0]))
visamp_model[idx1]*=(v1_best+f1_best*ll)
visamp_model[idx2]*=(v2_best+f2_best*ll)
visamp_model[idx3]*=(v3_best+f3_best*ll)
visamp_model[idx4]*=(v4_best+f4_best*ll)
visamp_model[idx5]*=(v5_best+f5_best*ll)
visamp_model[idx6]*=(v6_best+f6_best*ll)

complex_vis2 = cvis_model(best_params,ucoords,vcoords,eff_wave[0])
visibility2 = complex_vis2*np.conj(complex_vis2)
vis2_model = visibility2.real
vis2_model = np.swapaxes(vis2_model,0,1)

#############################################
## drop tel by tel and refit -- test pupil
#############################################
xfit = []
yfit = []
for exclude in ['none','E1','W2','W1','S2','S1','E2']:
    # get information from fits file
    if dtype=='chara':
        t3phi_d,t3phierr_d,vis2_d,vis2err_d,visphi_d,visphierr_d,visamp_d,visamperr_d,u_coords_d,v_coords_d,ucoords_d,vcoords_d,eff_wave_d,tels_d,vistels_d,time_obs_d,az_d = read_chara(dir,target_id,interact,exclude,bl_drop)
    #if dtype=='vlti':
    #    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_vlti(dir,interact)
    #if dtype=='chara_old':
    #    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara_old(dir,interact,exclude)
    else:
        xfit.append(0)
        yfit.append(0)
        continue
    ########################################################
    ## do polynomial dispersion fit for each visphi measurement
    if 'dphase' in fitting_vars:
        dispersion=[]
        for vis in visphi_d:
            if np.count_nonzero(~np.isnan(vis))>0:
                y=vis
                x=eff_wave_d[0]
                idx = np.isfinite(x) & np.isfinite(y)
                z=np.polyfit(x[idx],y[idx],2)
                p = np.poly1d(z)
                dispersion.append(p(x))
            else:
                dispersion.append(vis)
        dispersion=np.array(dispersion)
        ## subtract dispersion (FIXME: should fit dispersion at each measurement)
        visphi_new_d = visphi_d-dispersion
    else:
        visphi_new_d = visphi_d

    print('Computing best fit -- dropping %s'%exclude)

    ra_start = np.random.uniform(ra_best-0.05,ra_best+0.05)
    dec_start = np.random.uniform(dec_best-0.05,dec_best+0.05)

    params = Parameters()
    params.add('ra',   value= ra_start)
    params.add('dec', value= dec_start)
    params.add('ratio', value= ratio_best, min=1.0)
    params.add('ud1',   value= ud1_best, vary=False)#min=0.0,max=2.0)
    params.add('ud2', value= ud2_best, vary=False)#min=0.0,max=2.0)
    params.add('bw', value=bw_best, min=0.0, max=0.1)

    minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi_d,t3phierr_d,visphi_new_d,visphierr_d,vis2_d,vis2err_d,visamp_d,visamperr_d,u_coords_d,v_coords_d,ucoords_d,vcoords_d,eff_wave_d[0],vistels_d,fitting_vars),nan_policy='omit')
    result = minner.minimize()

    xfit.append(result.params['ra'].value)
    yfit.append(result.params['dec'].value)
xfit=np.array(xfit)
yfit=np.array(yfit)

## plot results
full_plot='y'
with PdfPages("/Users/tgardner/ARMADA_epochs/%(1)s/%(1)s_%(2)s_summary.pdf"%{"1":target_id,"2":date}) as pdf:
    
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

    if full_plot=='n':
        ## next page - reduction and fit parameters
        textfig = plt.figure(figsize=(11.69,8.27))
        textfig.clf()
        txt_fit_cp = ('Best fit (ra,dec,ratio,ud1,ud2,bw,redchi2): %s'%best_fit)
        reduction = ('Reduction params (ncoh,nbs,ncs,int):%s'%reduction_params)
        textfig.text(0.5,0.75,txt_fit_cp,size=12,ha="center")
        textfig.text(0.5,0.5,reduction,size=12,ha="center")
        textfig.text(0.5,0.25,reduction_params,size=12,ha='center')
        pdf.savefig()
        plt.close()

    else:

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
        visamp_plot = np.array_split(visamp,n_vp)
        visamperr_plot = np.array_split(visamperr,n_vp)
        visamp_model_plot = np.array_split(visamp_model,n_vp)
        vis2_plot = np.array(np.array_split(vis2,n_v2))
        vis2err_plot = np.array(np.array_split(vis2err,n_v2))
        vis2_model_plot = np.array(np.array_split(vis2_model,n_v2))
        if dtype=='vlti':
            flux_plot = np.array(np.array_split(flux,n_v2))
            fluxerr_plot = np.array(np.array_split(fluxerr,n_v2))

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

        if dtype=='vlti':

            plt.figure(figsize=(10, 7))

            y1=np.nanmean(flux_plot,axis=0)[0,:] / np.max(np.nanmean(flux_plot,axis=0))
            y2=np.nanmean(flux_plot,axis=0)[1,:] / np.max(np.nanmean(flux_plot,axis=0))
            y3=np.nanmean(flux_plot,axis=0)[2,:] / np.max(np.nanmean(flux_plot,axis=0))
            y4=np.nanmean(flux_plot,axis=0)[3,:] / np.max(np.nanmean(flux_plot,axis=0))

            plt.plot(x,y1,'.-',label='tel1')
            plt.plot(x,y2,'.-',label='tel2')
            plt.plot(x,y3,'.-',label='tel3')
            plt.plot(x,y4,'.-',label='tel4')

            plt.axvline(x=2.16612e-6)
            plt.xlim(xmin=2.16e-6,xmax=2.17e-6)
            plt.title('Calibration Check')
            plt.xlabel('Wavelength (m)')
            plt.ylabel('Normalized Flux')    
            plt.legend() 
            pdf.savefig()
            plt.close()

        ## next page -- tel drop
        if dtype=='chara':
            spread = np.sqrt((xfit-xfit[0])**2+(yfit-yfit[0])**2)
            for x,y,label in zip(xfit,yfit,['none','E1','W2','W1','S2','S1','E2']):
                if label=='none':
                    plt.scatter(x, y, marker='*', zorder=2, label=label)
                else:
                    plt.scatter(x, y, marker='o', label=label)

            plt.title('STDEV = %s mas'%np.around(np.std(spread),4))
            plt.xlabel('d_RA (mas)')
            plt.ylabel('d_DE (mas)')
            plt.gca().invert_xaxis()
            plt.axis('equal')
            plt.legend()
            pdf.savefig()  
            plt.close()

        ## next page - reduction and fit parameters
        textfig = plt.figure(figsize=(11.69,8.27))
        textfig.clf()
        txt_fit_cp = ('Best fit (ra,dec,ratio,ud1,ud2,bw,redchi2): %s'%best_fit)
        reduction = ('Reduction params (ncoh,nbs,ncs,int):%s'%reduction_params)
        textfig.text(0.5,0.75,txt_fit_cp,size=12,ha="center")
        textfig.text(0.5,0.5,reduction,size=12,ha="center")
        textfig.text(0.5,0.25,reduction_params,size=12,ha='center')
        pdf.savefig()
        plt.close()

###########################################################
### Now do errors
###########################################################
print('Computing errors from CHI2 SURFACE')
if dtype=='vlti':
    size = 1.0
    steps = 100
else:
    size = 0.5
    steps = 250
ra_grid = np.linspace(ra_best-size,ra_best+size,steps)
dec_grid = np.linspace(dec_best-size,dec_best+size,steps)
chi_sq = []
redchi_sq = []
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
        ##create a set of Parameters
        params = [ra_try,dec_try,ratio_best,ud1_best,ud2_best,bw_best]
        ##do fit, minimizer uses LM for least square fitting of model to data
        chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels,fitting_vars)
        red_chi2 = np.nansum(chi**2) / nfree #(len(np.ndarray.flatten(t3phi))-len(params))
        raw_chi2 = np.nansum(chi**2)
        
        #params = Parameters()
        #params.add('ra',   value= ra_try, vary=False)
        #params.add('dec', value= dec_try, vary=False)
        #params.add('ratio', value= ratio_best, min=1.0)
        #params.add('ud1',   value= ud1_best, vary=False)#min=0.0,max=3.0)
        #params.add('ud2', value= ud2_best, vary=False)
        #params.add('bw', value=bw_best, min=0.0, max=0.1)

        #params.add('v1',value=v1_best,vary=False)
        #params.add('v2',value=v2_best,vary=False)
        #params.add('v3',value=v3_best,vary=False)
        #params.add('v4',value=v4_best,vary=False)
        #params.add('v5',value=v5_best,vary=False)
        #params.add('v6',value=v6_best,vary=False)
        #params.add('f1',value=f1_best,vary=False)
        #params.add('f2',value=f2_best,vary=False)
        #params.add('f3',value=f3_best,vary=False)
        #params.add('f4',value=f4_best,vary=False)
        #params.add('f5',value=f5_best,vary=False)
        #params.add('f6',value=f6_best,vary=False)

        #minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels,fitting_vars),nan_policy='omit')
        #result = minner.minimize()
        #raw_chi2 = result.chisqr
        #red_chi2 = result.redchi
        
        chi_sq.append(raw_chi2)
        redchi_sq.append(red_chi2)
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
redchi_sq = np.array(redchi_sq)
## plot chisq surface grid
#plt.scatter(ra_results, dec_results, c=chi_sq, cmap=cm.inferno_r)
plt.scatter(ra_results, dec_results, c=redchi_sq, cmap=cm.inferno_r)
plt.colorbar()
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig("/Users/tgardner/ARMADA_epochs/%s/%s_%s_chisq.pdf"%(target_id,target_id,date))
plt.close()
## isolate region where delta_chisq < 2.296
#params = [ra_best,dec_best,ratio_best,ud1_best,ud2_best,bw_best]
#chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
#chi2_best = np.nansum(chi**2)/(len(np.ndarray.flatten(t3phi))-len(params))
#chi2_best = chi_sq_best
chi2_best = redchi_sq_best

#index_err = np.where(chi_sq < (chi2_best+2.296) )
index_err = np.where(redchi_sq < (chi2_best+1) )
#chi_err = chi_sq[index_err]
chi_err = redchi_sq[index_err]
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
#angle_new = 360-angle_new
#angle = 360-angle
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
plt.savefig("/Users/tgardner/ARMADA_epochs/%s/%s_%s_ellipse.pdf"%(target_id,target_id,date))
plt.close()

## write results to a txt file
t = np.around(np.nanmedian(time_obs),4)
sep,pa = np.around(cart2pol(best_fit[0],best_fit[1]),decimals=4)
f = open("/Users/tgardner/ARMADA_epochs/%s/%s_%s_chi2err.txt"%(target_id,target_id,date),"w+")
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
plt.savefig("/Users/tgardner/ARMADA_epochs/%s/%s_%s_ellipse_boot.pdf"%(target_id,target_id,date))
plt.close()

## write results to a txt file
t = np.around(np.nanmedian(time_obs),4)
sep,pa = np.around(cart2pol(best_fit[0],best_fit[1]),decimals=4)
f = open("/Users/tgardner/ARMADA_epochs/%s/%s_%s_bootstrap.txt"%(target_id,target_id,date),"w+")
f.write("# date mjd sep(mas) pa(Deg) err_maj(mas) err_min(mas) err_pa(deg)\r\n")
f.write("%s %s %s %s %s %s %s"%(date,t,sep,pa,ellipse_params[0],ellipse_params[1],ellipse_params[2]))
f.close()
