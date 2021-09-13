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
def combined_minimizer(params,cp,cp_err,vphi,vphierr,v2,v2err,vamp,vamperr,u_coord,v_coord,ucoord,vcoord,wl,vtels):

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
    
    if 'visphi' in flag:
        complex_vis = cvis_model(params,ucoord,vcoord,wl)
        visibility = np.angle(complex_vis)

        ## correct for slope and offset
        try:    
            vp1 = params['vp1']
            vp2 = params['vp2']
            vp3 = params['vp3']
            vp4 = params['vp4']
            vp5 = params['vp5']
            vp6 = params['vp6'] 
            fp1 = params['fp1']
            fp2 = params['fp2']
            fp3 = params['fp3']
            fp4 = params['fp4']
            fp5 = params['fp5']
            fp6 = params['fp6']
            ffp1 = params['ffp1']
            ffp2 = params['ffp2']
            ffp3 = params['ffp3']
            ffp4 = params['ffp4']
            ffp5 = params['ffp5']
            ffp6 = params['ffp6']
            fffp1 = params['fffp1']
            fffp2 = params['fffp2']
            fffp3 = params['fffp3']
            fffp4 = params['fffp4']
            fffp5 = params['fffp5']
            fffp6 = params['fffp6']
        except: 
            vp1 = params[18]
            vp2 = params[19]
            vp3 = params[20]
            vp4 = params[21]
            vp5 = params[22]
            vp6 = params[23] 
            fp1 = params[24]
            fp2 = params[25]
            fp3 = params[26]
            fp4 = params[27]
            fp5 = params[28]
            fp6 = params[29]
            ffp1 = params[30]
            ffp2 = params[31]
            ffp3 = params[32]
            ffp4 = params[33]
            ffp5 = params[34]
            ffp6 = params[35]
            fffp1 = params[36]
            fffp2 = params[37]
            fffp3 = params[38]
            fffp4 = params[39]
            fffp5 = params[40]
            fffp6 = params[41]
        
        dphase = visibility
        vis_model = dphase
        vis_model = np.swapaxes(vis_model,0,1)
        vis_model=np.array(vis_model)

        idx1 = np.where(np.all(vtels==vtels[0],axis=1))
        idx2 = np.where(np.all(vtels==vtels[1],axis=1))
        idx3 = np.where(np.all(vtels==vtels[2],axis=1))
        idx4 = np.where(np.all(vtels==vtels[3],axis=1))
        idx5 = np.where(np.all(vtels==vtels[4],axis=1))
        idx6 = np.where(np.all(vtels==vtels[5],axis=1))

        ll = (wl-np.median(wl))/(max(wl)-min(wl))

        vis_model[idx1]+=(vp1 + fp1*ll + ffp1*ll**2 + fffp1*ll**3)
        vis_model[idx2]+=(vp2 + fp2*ll + ffp2*ll**2 + fffp2*ll**3)
        vis_model[idx3]+=(vp3 + fp3*ll + ffp3*ll**2 + fffp3*ll**3)
        vis_model[idx4]+=(vp4 + fp4*ll + ffp4*ll**2 + fffp4*ll**3)
        vis_model[idx5]+=(vp5 + fp5*ll + ffp5*ll**2 + fffp5*ll**3)
        vis_model[idx6]+=(vp6 + fp6*ll + ffp6*ll**2 + fffp6*ll**3)

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
flag = input('fit to: vis2,cphase,dphase,vamp,visphi (separate with spaces): ').split(' ')
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
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,flux,fluxerr = read_vlti(dir,interact)
if dtype=='chara_old':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs = read_chara_old(dir,interact,exclude)
########################################################
print(t3phi.shape,t3phierr.shape)
print(visphi.shape,visphierr.shape)
print(vis2.shape,vis2err.shape)
print(visamp.shape,visamperr.shape)
print(flux.shape,fluxerr.shape)
print(eff_wave.shape)

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

telnames = vistels[0:6]
for ra_try in tqdm(ra_grid):
    for dec_try in dec_grid:

        if vary_ratio=='y':
            ## lmfit for varying params (slower)
            params = Parameters()
            params.add('ra',   value= ra_try, vary=False)
            params.add('dec', value= dec_try, vary=False)
            params.add('ratio', value= a3, min=1.0)
            params.add('ud1',   value= a4, vary=False)#min=0.0,max=3.0)
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

            params.add('vp1',value=0,vary=False)
            params.add('vp2',value=0,vary=False)
            params.add('vp3',value=0,vary=False)
            params.add('vp4',value=0,vary=False)
            params.add('vp5',value=0,vary=False)
            params.add('vp6',value=0,vary=False)
            params.add('fp1',value=0,vary=False)
            params.add('fp2',value=0,vary=False)
            params.add('fp3',value=0,vary=False)
            params.add('fp4',value=0,vary=False)
            params.add('fp5',value=0,vary=False)
            params.add('fp6',value=0,vary=False)
            params.add('ffp1',value=0,vary=False)
            params.add('ffp2',value=0,vary=False)
            params.add('ffp3',value=0,vary=False)
            params.add('ffp4',value=0,vary=False)
            params.add('ffp5',value=0,vary=False)
            params.add('ffp6',value=0,vary=False)
            params.add('fffp1',value=0,vary=False)
            params.add('fffp2',value=0,vary=False)
            params.add('fffp3',value=0,vary=False)
            params.add('fffp4',value=0,vary=False)
            params.add('fffp5',value=0,vary=False)
            params.add('fffp6',value=0,vary=False)

            minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels),nan_policy='omit')
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
            params = [ra_try,dec_try,a3,a4,a5,a6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels)
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
ratio_results = np.array(ratio_results)
chi_sq = np.array(chi_sq)

np.save("/Users/tgardne/ARMADA_epochs/%s/%s_%s_ra.npy"%(target_id,target_id,date),ra_results)
np.save("/Users/tgardne/ARMADA_epochs/%s/%s_%s_dec.npy"%(target_id,target_id,date),dec_results)
np.save("/Users/tgardne/ARMADA_epochs/%s/%s_%s_ratio.npy"%(target_id,target_id,date),ratio_results)
np.save("/Users/tgardne/ARMADA_epochs/%s/%s_%s_chisq.npy"%(target_id,target_id,date),chi_sq)

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

if 'vamp' in flag:
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
if 'visphi' in flag:
    params.add('vp1',value=0)
    params.add('vp2',value=0)
    params.add('vp3',value=0)
    params.add('vp4',value=0)
    params.add('vp5',value=0)
    params.add('vp6',value=0)
    params.add('fp1',value=0)
    params.add('fp2',value=0)
    params.add('fp3',value=0)
    params.add('fp4',value=0)
    params.add('fp5',value=0)
    params.add('fp6',value=0)
    params.add('ffp1',value=0)
    params.add('ffp2',value=0)
    params.add('ffp3',value=0)
    params.add('ffp4',value=0)
    params.add('ffp5',value=0)
    params.add('ffp6',value=0)
    params.add('fffp1',value=0)
    params.add('fffp2',value=0)
    params.add('fffp3',value=0)
    params.add('fffp4',value=0)
    params.add('fffp5',value=0)
    params.add('fffp6',value=0)
else:
    params.add('vp1',value=0,vary=False)
    params.add('vp2',value=0,vary=False)
    params.add('vp3',value=0,vary=False)
    params.add('vp4',value=0,vary=False)
    params.add('vp5',value=0,vary=False)
    params.add('vp6',value=0,vary=False)
    params.add('fp1',value=0,vary=False)
    params.add('fp2',value=0,vary=False)
    params.add('fp3',value=0,vary=False)
    params.add('fp4',value=0,vary=False)
    params.add('fp5',value=0,vary=False)
    params.add('fp6',value=0,vary=False)
    params.add('ffp1',value=0,vary=False)
    params.add('ffp2',value=0,vary=False)
    params.add('ffp3',value=0,vary=False)
    params.add('ffp4',value=0,vary=False)
    params.add('ffp5',value=0,vary=False)
    params.add('ffp6',value=0,vary=False)
    params.add('fffp1',value=0,vary=False)
    params.add('fffp2',value=0,vary=False)
    params.add('fffp3',value=0,vary=False)
    params.add('fffp4',value=0,vary=False)
    params.add('fffp5',value=0,vary=False)
    params.add('fffp6',value=0,vary=False)

minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels),nan_policy='omit')
result = minner.minimize()
report_fit(result)

chi_sq_best = result.redchi
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

vp1_best = result.params['vp1'].value
vp2_best = result.params['vp2'].value
vp3_best = result.params['vp3'].value
vp4_best = result.params['vp4'].value
vp5_best = result.params['vp5'].value
vp6_best = result.params['vp6'].value
fp1_best = result.params['fp1'].value
fp2_best = result.params['fp2'].value
fp3_best = result.params['fp3'].value
fp4_best = result.params['fp4'].value
fp5_best = result.params['fp5'].value
fp6_best = result.params['fp6'].value
ffp1_best = result.params['ffp1'].value
ffp2_best = result.params['ffp2'].value
ffp3_best = result.params['ffp3'].value
ffp4_best = result.params['ffp4'].value
ffp5_best = result.params['ffp5'].value
ffp6_best = result.params['ffp6'].value
fffp1_best = result.params['fffp1'].value
fffp2_best = result.params['fffp2'].value
fffp3_best = result.params['fffp3'].value
fffp4_best = result.params['fffp4'].value
fffp5_best = result.params['fffp5'].value
fffp6_best = result.params['fffp6'].value

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

best_params.add('vp1',value=vp1_best)
best_params.add('vp2',value=vp2_best)
best_params.add('vp3',value=vp3_best)
best_params.add('vp4',value=vp4_best)
best_params.add('vp5',value=vp5_best)
best_params.add('vp6',value=vp6_best)
best_params.add('fp1',value=fp1_best)
best_params.add('fp2',value=fp2_best)
best_params.add('fp3',value=fp3_best)
best_params.add('fp4',value=fp4_best)
best_params.add('fp5',value=fp5_best)
best_params.add('fp6',value=fp6_best)
best_params.add('ffp1',value=ffp1_best)
best_params.add('ffp2',value=ffp2_best)
best_params.add('ffp3',value=ffp3_best)
best_params.add('ffp4',value=ffp4_best)
best_params.add('ffp5',value=ffp5_best)
best_params.add('ffp6',value=ffp6_best)
best_params.add('fffp1',value=fffp1_best)
best_params.add('fffp2',value=fffp2_best)
best_params.add('fffp3',value=fffp3_best)
best_params.add('fffp4',value=fffp4_best)
best_params.add('fffp5',value=fffp5_best)
best_params.add('fffp6',value=fffp6_best)

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
visibility = np.angle(complex_vis)#*180/np.pi
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

visphi_model[idx1]+=(vp1_best + fp1_best*ll + ffp1_best*ll**2 + fffp1_best*ll**3)
visphi_model[idx2]+=(vp2_best + fp2_best*ll + ffp2_best*ll**2 + fffp2_best*ll**3)
visphi_model[idx3]+=(vp3_best + fp3_best*ll + ffp3_best*ll**2 + fffp3_best*ll**3)
visphi_model[idx4]+=(vp4_best + fp4_best*ll + ffp4_best*ll**2 + fffp4_best*ll**3)
visphi_model[idx5]+=(vp5_best + fp5_best*ll + ffp5_best*ll**2 + fffp5_best*ll**3)
visphi_model[idx6]+=(vp6_best + fp6_best*ll + ffp6_best*ll**2 + fffp6_best*ll**3)

visphi_model*=(180/np.pi)

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
    visamp_plot = np.array_split(visamp,n_vp)
    visamperr_plot = np.array_split(visamperr,n_vp)
    visamp_model_plot = np.array_split(visamp_model,n_vp)
    vis2_plot = np.array(np.array_split(vis2,n_v2))
    vis2err_plot = np.array(np.array_split(vis2err,n_v2))
    vis2_model_plot = np.array(np.array_split(vis2_model,n_v2))
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
steps = 200
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
        ##create a set of Parameters
        #params = [ra_try,dec_try,ratio_best,ud1_best,ud2_best,bw_best]
        ##do fit, minimizer uses LM for least square fitting of model to data
        #chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
        #red_chi2 = np.nansum(chi**2)/(len(np.ndarray.flatten(t3phi))-len(params))
        
        params = Parameters()
        params.add('ra',   value= ra_try, vary=False)
        params.add('dec', value= dec_try, vary=False)
        params.add('ratio', value= ratio_best, vary=False)#min=1.0)
        params.add('ud1',   value= ud1_best, vary=False)#min=0.0,max=3.0)
        params.add('ud2', value= ud2_best, vary=False)
        params.add('bw', value=bw_best, vary=False)#min=0.0, max=0.1)

        params.add('v1',value=v1_best,vary=False)
        params.add('v2',value=v2_best,vary=False)
        params.add('v3',value=v3_best,vary=False)
        params.add('v4',value=v4_best,vary=False)
        params.add('v5',value=v5_best,vary=False)
        params.add('v6',value=v6_best,vary=False)
        params.add('f1',value=f1_best,vary=False)
        params.add('f2',value=f2_best,vary=False)
        params.add('f3',value=f3_best,vary=False)
        params.add('f4',value=f4_best,vary=False)
        params.add('f5',value=f5_best,vary=False)
        params.add('f6',value=f6_best,vary=False)

        minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0],vistels),nan_policy='omit')
        result = minner.minimize()
        red_chi2 = result.redchi
        
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
#params = [ra_best,dec_best,ratio_best,ud1_best,ud2_best,bw_best]
#chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
#chi2_best = np.nansum(chi**2)/(len(np.ndarray.flatten(t3phi))-len(params))
chi2_best = chi_sq_best

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