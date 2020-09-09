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
#absolute = input('use absolute phase value (y/n)?')
absolute='n'

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
a3 = float(input('flux ratio (f1/f2): '))
a4 = float(input('UD1 (mas): '))
a5 = float(input('UD2 (mas): '))
a6 = float(input('bw smearing (1/R): '))

dra = -sep_value*np.cos((90+pa_value)*np.pi/180)
ddec = sep_value*np.sin((90+pa_value)*np.pi/180)

######################################################################
## DO CHI2 FIT 
######################################################################

## Do a chi2 fit for phases
params = Parameters()
params.add('ra',   value= dra)
params.add('dec', value= ddec)
params.add('ratio', value= a3, min=1.0)
params.add('ud1',   value= a4, vary=False)#min=0.0,max=2.0)
params.add('ud2', value= a5, vary=False)#min=0.0,max=2.0)
params.add('bw', value=a6, min=0.0, max=0.1)

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

##########################################################
## Re-shape data to bootstrap in TIME
##########################################################

## regroup data by measurements
if dtype=='chara':
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
u_coords_plot = np.array(np.array_split(u_coords,n_cp))
v_coords_plot = np.array(np.array_split(v_coords,n_cp))

visphi_new_plot = np.array(np.array_split(visphi_new,n_vp))
visphierr_plot = np.array(np.array_split(visphierr,n_vp))
vis2_plot = np.array(np.array_split(vis2,n_v2))
vis2err_plot = np.array(np.array_split(vis2err,n_v2))
ucoords_plot = np.array(np.array_split(ucoords,n_v2))
vcoords_plot = np.array(np.array_split(vcoords,n_v2))

##########################################################
## Now do error ellipses
##########################################################

chi_sq=[]
ra_results=[]
dec_results=[]
ratio_results=[]
ud1_results=[]
ud2_results=[]
bw_results=[]

print('Computing error ellipses with bootstrap')
index=np.arange(1000)
for i in tqdm(index):
    
    #r_t3 = np.random.randint(t3phi.shape[0],size=len(t3phi))
    #r_v2 = np.random.randint(vis2.shape[0],size=len(vis2))
    #r_vp = np.random.randint(visphi_new.shape[0],size=len(visphi_new))

    #t3phi_boot = t3phi[r_t3,:]
    #t3phierr_boot = t3phierr[r_t3,:]
    #u_coords_boot = u_coords[r_t3,:]
    #v_coords_boot = v_coords[r_t3,:]

    #vis2_boot = vis2[r_v2,:]
    #vis2err_boot = vis2err[r_v2,:]
    #ucoords_boot = ucoords[r_v2]
    #vcoords_boot = vcoords[r_v2]
 
    #visphi_boot = visphi_new[r_vp,:]
    #visphierr_boot = visphierr[r_vp,:]

    r = np.random.randint(t3phi_plot.shape[0],size=len(t3phi_plot))

    t3phi_boot = t3phi_plot[r,:]
    t3phierr_boot = t3phierr_plot[r,:]
    u_coords_boot = u_coords_plot[r,:]
    v_coords_boot = v_coords_plot[r,:]

    vis2_boot = vis2_plot[r,:]
    vis2err_boot = vis2err_plot[r,:]
    ucoords_boot = ucoords_plot[r]
    vcoords_boot = vcoords_plot[r]
 
    visphi_boot = visphi_new_plot[r,:]
    visphierr_boot = visphierr_plot[r,:]

    ## Back to correct shape for fitting
    t3phi_boot = t3phi_boot.reshape(int(t3phi_boot.shape[0])*int(t3phi_boot.shape[1]),t3phi_boot.shape[2])
    t3phierr_boot = t3phierr_boot.reshape(int(t3phierr_boot.shape[0])*int(t3phierr_boot.shape[1]),t3phierr_boot.shape[2])
    u_coords_boot = u_coords_boot.reshape(int(u_coords_boot.shape[0])*int(u_coords_boot.shape[1]),u_coords_boot.shape[2])
    v_coords_boot = v_coords_boot.reshape(int(v_coords_boot.shape[0])*int(v_coords_boot.shape[1]),v_coords_boot.shape[2])
    vis2_boot = vis2_boot.reshape(int(vis2_boot.shape[0])*int(vis2_boot.shape[1]),vis2_boot.shape[2])
    vis2err_boot = vis2err_boot.reshape(int(vis2err_boot.shape[0])*int(vis2err_boot.shape[1]),vis2err_boot.shape[2])
    ucoords_boot = ucoords_boot.reshape(int(ucoords_boot.shape[0])*int(ucoords_boot.shape[1]))
    vcoords_boot = vcoords_boot.reshape(int(vcoords_boot.shape[0])*int(vcoords_boot.shape[1]))
    visphi_boot = visphi_boot.reshape(int(visphi_boot.shape[0])*int(visphi_boot.shape[1]),visphi_boot.shape[2])
    visphierr_boot = visphierr_boot.reshape(int(visphierr_boot.shape[0])*int(visphierr_boot.shape[1]),visphierr_boot.shape[2])

    ## Do a chi2 fit for phases
    params = Parameters()
    params.add('ra',   value= ra_best)
    params.add('dec', value= dec_best)
    params.add('ratio', value= ratio_best, vary=False)#min=1.0)
    params.add('ud1',   value= ud1_best, vary=False)#min=0.0,max=2.0)
    params.add('ud2', value= ud2_best, vary=False)#min=0.0,max=2.0)
    params.add('bw', value=bw_best, vary=False)#min=0.0, max=0.1)

    minner = Minimizer(combined_minimizer, params, fcn_args=(t3phi_boot,t3phierr_boot,visphi_boot,visphierr_boot,vis2_boot,vis2err_boot,u_coords_boot,v_coords_boot,ucoords_boot,vcoords_boot,eff_wave[0]),nan_policy='omit')
    result = minner.minimize()

    chi_sq.append(result.redchi)
    ra_results.append(result.params['ra'].value)
    dec_results.append(result.params['dec'].value)
    ratio_results.append(result.params['ratio'].value)
    ud1_results.append(result.params['ud1'].value)
    ud2_results.append(result.params['ud2'].value)
    bw_results.append(result.params['bw'].value)

chi_sq=np.array(chi_sq)
ra_results=np.array(ra_results)
dec_results=np.array(dec_results)
ratio_results=np.array(ratio_results)
ud1_results=np.array(ud1_results)
ud2_results=np.array(ud2_results)
bw_results=np.array(bw_results)

## fit an ellipse to the data
a,b,angle = ellipse_fitting(ra_results,dec_results)
angle = angle*180/np.pi
ra_mean = np.mean(ra_results)
dec_mean = np.mean(dec_results)

## want to measure east of north (different than python)
angle_new = 90-angle
if angle_new<0:
    angle_new=360+angle_new
ellipse_params = np.around(np.array([a,b,angle_new]),decimals=4)

ell = Ellipse(xy=(ra_mean,dec_mean),width=2*a,height=2*b,angle=angle,facecolor='lightgrey')
plt.gca().add_patch(ell)
plt.scatter(ra_results, dec_results, zorder=2)#, c=chi_err, cmap=cm.inferno_r,zorder=2)
plt.title('a,b,thet=%s'%ellipse_params)
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig("/Users/tgardne/ARMADA_epochs/%s/%s_%s_bootstrap.pdf"%(target_id,target_id,date))
plt.close()