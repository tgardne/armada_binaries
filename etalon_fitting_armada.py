######################################################################
## Tyler Gardner
##
## Algorithm for fitting to etalon data
##
## free params alpha, n (for glass) ; si-sj (etalon thickness) ; sep vector,ratio (binary)
######################################################################

from chara_uvcalc import uv_calc
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
eachindex = lambda lst: range(len(lst))
from tqdm import tqdm
#import numpy_indexed as npi
import os
import matplotlib as mpl
from scipy import special
import random
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from read_oifits import read_chara
from itertools import combinations

####################################################
## the etalon model
def mas2rad(mas):
    rad=mas/1000.
    rad=rad/3600.
    rad=rad*np.pi/180.
    return(rad)

def etalon(u,v,ratio,ud,delta_s,lamb,bw):
    lamb2 = lamb*1e6
    n = np.sqrt(0.6961663*lamb2**2/(lamb2**2-0.0684043**2)+0.4079426*lamb2**2/(lamb2**2-0.1162414**2)+0.8974794*lamb2**2/(lamb2**2-9.896161**2)+1)

    #alpha = ratio + slope2*(lamb-1.4e-6)

    secondary_flux = 1/(1+ratio+1/ratio)
    primary_flux = 1/(1+1/ratio+1/(ratio*ratio))
    tertiary_flux = 1 - primary_flux - secondary_flux

    diameter = mas2rad(ud)
    x = np.pi*diameter*np.sqrt(u**2+v**2)
    
    ## bessel 1st order
    f1 = 2*special.jn(1,x)/x
    ## bandwidth smearing
    vis_bw = np.sinc(bw*(n*delta_s/lamb))
        
    complex_vis = primary_flux*f1 + vis_bw*secondary_flux*f1*np.exp(-2*np.pi*1j*n*delta_s/lamb) + vis_bw*tertiary_flux*f1*np.exp(-4*np.pi*1j*n*delta_s/lamb)

    return complex_vis
######################################################

## function which returns complex vis given sep, pa, flux ratio, HA, dec, UD1, UD2, wavelength
def cvis_model(params, delta_s, u, v, wl):
    
    ratio = params['ratio']
    ud = params['ud']
    bw = params['bw']

    ul=u/wl
    vl=v/wl
    
    vis=etalon(ul,vl,ratio,ud,delta_s,wl,bw)
    return vis

## for closure phase
def cvis_model_cp(params, delta_s1, delta_s2, delta_s3, u, v, wl):
    
    ratio = params['ratio']
    ud = params['ud']
    bw = params['bw']

    ul=u/wl
    vl=v/wl

    delta_s=np.array([delta_s1,delta_s2,delta_s3])
    
    vis=etalon(ul,vl,ratio,ud,delta_s,wl,bw)
    return vis

## function which returns residual of model and data to be minimized
def vis_minimizer(params,v2,v2err,vphi,vphierr,t3phi,t3phierr,ucoords,vcoords,u_coord,v_coord,wl,vistel,tel):
    visphi_model = []
    vis_model = []
    dispersion_model = []

    ## correct for wavelength
    lbd_ref = 1.60736e-06
    eff_wave_cor = eff_wave[0]*(1+params['alpha']*(eff_wave[0]-lbd_ref))

    for item1,item2,item4 in zip(ucoords,vcoords,vistel):
        
        ## need to define etalon seps for each triangle
        if str(item4)==str(vistels[0]):
            delta_s = -2*(params['s1']-params['s2'])
            a = params['a1']
            b = params['b1']
            c = params['c1']
        if str(item4)==str(vistels[1]):
            delta_s = -2*(params['s1']-params['s3'])
            a = params['a2']
            b = params['b2']
            c = params['c2']
        if str(item4)==str(vistels[2]):
            delta_s = -2*(params['s1']-params['s4'])
            a = params['a3']
            b = params['b3']
            c = params['c3']
        if str(item4)==str(vistels[3]):
            delta_s = -2*(params['s1']-params['s5'])
            a = params['a4']
            b = params['b4']
            c = params['c4']
        if str(item4)==str(vistels[4]):
            delta_s = -2*(params['s1']-params['s6'])
            a = params['a5']
            b = params['b5']
            c = params['c5']
        if str(item4)==str(vistels[5]):
            delta_s = -2*(params['s2']-params['s3'])
            a = params['a6']
            b = params['b6']
            c = params['c6']
        if str(item4)==str(vistels[6]):
            delta_s = -2*(params['s2']-params['s4'])
            a = params['a7']
            b = params['b7']
            c = params['c7']
        if str(item4)==str(vistels[7]):
            delta_s = -2*(params['s2']-params['s5'])
            a = params['a8']
            b = params['b8']
            c = params['c8']
        if str(item4)==str(vistels[8]):
            delta_s = -2*(params['s2']-params['s6'])
            a = params['a9']
            b = params['b9']
            c = params['c9']
        if str(item4)==str(vistels[9]):
            delta_s = -2*(params['s3']-params['s4'])
            a = params['a10']
            b = params['b10']
            c = params['c10']
        if str(item4)==str(vistels[10]):
            delta_s = -2*(params['s3']-params['s5'])
            a = params['a11']
            b = params['b11']
            c = params['c11']
        if str(item4)==str(vistels[11]):
            delta_s = -2*(params['s3']-params['s6'])
            a = params['a12']
            b = params['b12']
            c = params['c12']
        if str(item4)==str(vistels[12]):
            delta_s = -2*(params['s4']-params['s5'])
            a = params['a13']
            b = params['b13']
            c = params['c13']
        if str(item4)==str(vistels[13]):
            delta_s = -2*(params['s4']-params['s6'])
            a = params['a14']
            b = params['b14']
            c = params['c14']
        if str(item4)==str(vistels[14]):
            delta_s = -2*(params['s5']-params['s6'])
            a = params['a15']
            b = params['b15']
            c = params['c15']

        p = np.poly1d([a,b,c])
        dispersion = p(eff_wave_cor)
        dispersion_model.append(dispersion)

        visphi_arr=[]
        for w in eff_wave_cor:
            complex_vis = cvis_model(params,delta_s,item1, item2, w)
            visibility = np.angle(complex_vis)
            visphi_arr.append(visibility)
        visphi_arr=np.array(visphi_arr)
        dphase = visphi_arr[1:]-visphi_arr[:-1]
        dphase = np.insert(dphase,len(dphase),np.nan)
        visphi_model.append(dphase)

        vis_arr=[]
        for w in eff_wave_cor:
            complex_vis2 = cvis_model(params,delta_s,item1, item2, w)
            visibility2 = complex_vis2*np.conj(complex_vis2)
            vis_arr.append(visibility2.real)
        vis_model.append(vis_arr)

    vis_model=np.array(vis_model)
    visphi_model=np.array(visphi_model)
    dispersion_model=np.array(dispersion_model)

    cp_model = []
    for item1,item2,item4 in zip(u_coord,v_coord,tel):
        ## need to define etalon seps for each triangle
        if str(item4)==str(tels[0]):
            delta_s1 = -2*(params['s1']-params['s2'])
            delta_s2 = -2*(params['s2']-params['s3'])
            delta_s3 = -2*(params['s3']-params['s1'])
        if str(item4)==str(tels[1]):
            delta_s1 = -2*(params['s1']-params['s2'])
            delta_s2 = -2*(params['s2']-params['s4'])
            delta_s3 = -2*(params['s4']-params['s1'])
        if str(item4)==str(tels[2]):
            delta_s1 = -2*(params['s1']-params['s2'])
            delta_s2 = -2*(params['s2']-params['s5'])
            delta_s3 = -2*(params['s5']-params['s1'])
        if str(item4)==str(tels[3]):
            delta_s1 = -2*(params['s1']-params['s2'])
            delta_s2 = -2*(params['s2']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s1'])
        if str(item4)==str(tels[4]):
            delta_s1 = -2*(params['s1']-params['s3'])
            delta_s2 = -2*(params['s3']-params['s4'])
            delta_s3 = -2*(params['s4']-params['s1'])
        if str(item4)==str(tels[5]):
            delta_s1 = -2*(params['s1']-params['s3'])
            delta_s2 = -2*(params['s3']-params['s5'])
            delta_s3 = -2*(params['s5']-params['s1'])
        if str(item4)==str(tels[6]):
            delta_s1 = -2*(params['s1']-params['s3'])
            delta_s2 = -2*(params['s3']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s1'])
        if str(item4)==str(tels[7]):
            delta_s1 = -2*(params['s1']-params['s4'])
            delta_s2 = -2*(params['s4']-params['s5'])
            delta_s3 = -2*(params['s5']-params['s1'])
        if str(item4)==str(tels[8]):
            delta_s1 = -2*(params['s1']-params['s4'])
            delta_s2 = -2*(params['s4']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s1'])
        if str(item4)==str(tels[9]):
            delta_s1 = -2*(params['s1']-params['s5'])
            delta_s2 = -2*(params['s5']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s1'])
        if str(item4)==str(tels[10]):
            delta_s1 = -2*(params['s2']-params['s3'])
            delta_s2 = -2*(params['s3']-params['s4'])
            delta_s3 = -2*(params['s4']-params['s2'])
        if str(item4)==str(tels[11]):
            delta_s1 = -2*(params['s2']-params['s3'])
            delta_s2 = -2*(params['s3']-params['s5'])
            delta_s3 = -2*(params['s5']-params['s2'])
        if str(item4)==str(tels[12]):
            delta_s1 = -2*(params['s2']-params['s3'])
            delta_s2 = -2*(params['s3']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s2'])
        if str(item4)==str(tels[13]):
            delta_s1 = -2*(params['s2']-params['s4'])
            delta_s2 = -2*(params['s4']-params['s5'])
            delta_s3 = -2*(params['s5']-params['s2'])
        if str(item4)==str(tels[14]):
            delta_s1 = -2*(params['s2']-params['s4'])
            delta_s2 = -2*(params['s4']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s2'])
        if str(item4)==str(tels[15]):
            delta_s1 = -2*(params['s2']-params['s5'])
            delta_s2 = -2*(params['s5']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s2'])
        if str(item4)==str(tels[16]):
            delta_s1 = -2*(params['s3']-params['s4'])
            delta_s2 = -2*(params['s4']-params['s5'])
            delta_s3 = -2*(params['s5']-params['s3'])
        if str(item4)==str(tels[17]):
            delta_s1 = -2*(params['s3']-params['s4'])
            delta_s2 = -2*(params['s4']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s3'])
        if str(item4)==str(tels[18]):
            delta_s1 = -2*(params['s3']-params['s5'])
            delta_s2 = -2*(params['s5']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s3'])
        if str(item4)==str(tels[19]):
            delta_s1 = -2*(params['s4']-params['s5'])
            delta_s2 = -2*(params['s5']-params['s6'])
            delta_s3 = -2*(params['s6']-params['s4'])
        
        phase_arr=[]
        for w in eff_wave_cor:
            complex_vis = cvis_model_cp(params,delta_s1,delta_s2,delta_s3,item1,item2,w)
            phase = (np.angle(complex_vis[0])+np.angle(complex_vis[1])+np.angle(complex_vis[2]))
            phase_arr.append(phase)
        cp_model.append(phase_arr)
    cp_model=np.array(cp_model)

    ## subtract dispersion from data
    vphi_new = vphi-dispersion_model

    ## need to do an "angle difference" for phases
    vphi_data = vphi_new*np.pi/180
    vphi_err = vphierr*np.pi/180
    cp_data = t3phi*np.pi/180
    cp_err = t3phierr*np.pi/180

    vis_diff = (v2 - vis_model) / v2err
    visphi_diff = np.arctan2(np.sin(vphi_data-visphi_model),np.cos(vphi_data-visphi_model)) / vphi_err
    cp_diff = np.arctan2(np.sin(cp_data-cp_model),np.cos(cp_data-cp_model)) / cp_err

    if flag=='dphase':
        diff = visphi_diff
    if flag=='cphase':
        diff = cp_diff
    if flag=='vis2':
        diff = vis_diff
    if flag=='phases':
        diff = np.concatenate([visphi_diff,cp_diff])
    if flag=='vis':
        diff = np.concatenate([vis_diff,visphi_diff])
    if flag=='ALL':
        diff = np.concatenate([vis_diff,visphi_diff,cp_diff])

    return diff

## Ask the user which file contains the closure phases
dir=input('Path to oifits directory:')
target_id=input('Target ID (e.g. HD_206901): ')
date=input('Date for saved files (e.g. 2018Jul19):')

interact = input('interactive session with data? (y/n): ')
#exclude = input('exclude a telescope (e.g. E1): ')
#interact = 'n'
exclude = ''
## Exclude for 5T data?
exclude_t = input('exclude for 5T at end? (y/n): ')

## get information from fits file
t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,az = read_chara(dir,target_id,interact,exclude)
print('number measurements = %s'%str(int(t3phi.shape[0]/20)))
########################################################

### Split spectrum in half
#side = input('red or blue? ')
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

## do initial polynomial fit for each visphi measurement
a_guess=[]
b_guess=[]
c_guess=[]
polyfit=[]
for vis in visphi[:15]:
    if np.count_nonzero(~np.isnan(vis))>0:
        y=vis
        x=eff_wave[0]
        idx = np.isfinite(x) & np.isfinite(y)
        z=np.polyfit(x[idx],y[idx],2)
        p = np.poly1d(z)
        polyfit.append(p(x))
        a_guess.append(p[2])
        b_guess.append(p[1])
        c_guess.append(p[0])
    else:
        a_guess.append(0)
        b_guess.append(0)
        c_guess.append(0)
polyfit=np.array(polyfit)
a_guess=np.array(a_guess)
b_guess=np.array(b_guess)
c_guess=np.array(c_guess)

## subtract out the best fit polynomials
#visphi_subtracted = visphi-polyfit

###########
## Now do least-squares minimization -- using lmfit python package
###########

flag = input('vis2,cphase,dphase,phases,vis,ALL:')

#a1 = float(input('ratio: '))
a1 = 4.0
a2 = float(input('UD (mas): '))

s1_guess = 6.6e-6
s2_guess = 30.3e-6
s3_guess = 0
s4_guess = 18.7e-6
s5_guess = 24.8e-6
s6_guess = 12.6e-6

#create a set of Parameters, choose starting value and range for search
params = Parameters()

## star params
params.add('ratio', value= a1, min=1.0)
params.add('ud',   value= a2, vary=False)#min=0.0,max=1.0)
params.add('bw',value=1/190,min=0)

alpha_guess = float(input('alpha = '))
params.add('alpha',value=alpha_guess,vary=False)

## etalon params
params.add('s1',value=s1_guess)
params.add('s2',value=s2_guess)
params.add('s3',value=s3_guess, vary=False)
params.add('s4',value=s4_guess)
params.add('s5',value=s5_guess)
params.add('s6',value=s6_guess)

## dispersion params
if flag=='vis2' or flag=='cphase':
    params.add('a1',value=a_guess[0],vary=False)
    params.add('b1',value=b_guess[0],vary=False)
    params.add('c1',value=c_guess[0],vary=False)
    params.add('a2',value=a_guess[1],vary=False)
    params.add('b2',value=b_guess[1],vary=False)
    params.add('c2',value=c_guess[1],vary=False)
    params.add('a3',value=a_guess[2],vary=False)
    params.add('b3',value=b_guess[2],vary=False)
    params.add('c3',value=c_guess[2],vary=False)
    params.add('a4',value=a_guess[3],vary=False)
    params.add('b4',value=b_guess[3],vary=False)
    params.add('c4',value=c_guess[3],vary=False)
    params.add('a5',value=a_guess[4],vary=False)
    params.add('b5',value=b_guess[4],vary=False)
    params.add('c5',value=c_guess[4],vary=False)
    params.add('a6',value=a_guess[5],vary=False)
    params.add('b6',value=b_guess[5],vary=False)
    params.add('c6',value=c_guess[5],vary=False)
    params.add('a7',value=a_guess[6],vary=False)
    params.add('b7',value=b_guess[6],vary=False)
    params.add('c7',value=c_guess[6],vary=False)
    params.add('a8',value=a_guess[7],vary=False)
    params.add('b8',value=b_guess[7],vary=False)
    params.add('c8',value=c_guess[7],vary=False)
    params.add('a9',value=a_guess[8],vary=False)
    params.add('b9',value=b_guess[8],vary=False)
    params.add('c9',value=c_guess[8],vary=False)
    params.add('a10',value=a_guess[9],vary=False)
    params.add('b10',value=b_guess[9],vary=False)
    params.add('c10',value=c_guess[9],vary=False)
    params.add('a11',value=a_guess[10],vary=False)
    params.add('b11',value=b_guess[10],vary=False)
    params.add('c11',value=c_guess[10],vary=False)
    params.add('a12',value=a_guess[11],vary=False)
    params.add('b12',value=b_guess[11],vary=False)
    params.add('c12',value=c_guess[11],vary=False)
    params.add('a13',value=a_guess[12],vary=False)
    params.add('b13',value=b_guess[12],vary=False)
    params.add('c13',value=c_guess[12],vary=False)
    params.add('a14',value=a_guess[13],vary=False)
    params.add('b14',value=b_guess[13],vary=False)
    params.add('c14',value=c_guess[13],vary=False)
    params.add('a15',value=a_guess[14],vary=False)
    params.add('b15',value=b_guess[14],vary=False)
    params.add('c15',value=c_guess[14],vary=False)
else:
    params.add('a1',value=a_guess[0])
    params.add('b1',value=b_guess[0])
    params.add('c1',value=c_guess[0])
    params.add('a2',value=a_guess[1])
    params.add('b2',value=b_guess[1])
    params.add('c2',value=c_guess[1])
    params.add('a3',value=a_guess[2])
    params.add('b3',value=b_guess[2])
    params.add('c3',value=c_guess[2])
    params.add('a4',value=a_guess[3])
    params.add('b4',value=b_guess[3])
    params.add('c4',value=c_guess[3])
    params.add('a5',value=a_guess[4])
    params.add('b5',value=b_guess[4])
    params.add('c5',value=c_guess[4])
    params.add('a6',value=a_guess[5])
    params.add('b6',value=b_guess[5])
    params.add('c6',value=c_guess[5])
    params.add('a7',value=a_guess[6])
    params.add('b7',value=b_guess[6])
    params.add('c7',value=c_guess[6])
    params.add('a8',value=a_guess[7])
    params.add('b8',value=b_guess[7])
    params.add('c8',value=c_guess[7])
    params.add('a9',value=a_guess[8])
    params.add('b9',value=b_guess[8])
    params.add('c9',value=c_guess[8])
    params.add('a10',value=a_guess[9])
    params.add('b10',value=b_guess[9])
    params.add('c10',value=c_guess[9])
    params.add('a11',value=a_guess[10])
    params.add('b11',value=b_guess[10])
    params.add('c11',value=c_guess[10])
    params.add('a12',value=a_guess[11])
    params.add('b12',value=b_guess[11])
    params.add('c12',value=c_guess[11])
    params.add('a13',value=a_guess[12])
    params.add('b13',value=b_guess[12])
    params.add('c13',value=c_guess[12])
    params.add('a14',value=a_guess[13])
    params.add('b14',value=b_guess[13])
    params.add('c14',value=c_guess[13])
    params.add('a15',value=a_guess[14])
    params.add('b15',value=b_guess[14])
    params.add('c15',value=c_guess[14])

#do fit, minimizer uses LM for least square fitting of model to data
minner = Minimizer(vis_minimizer, params, fcn_args=(vis2,vis2err,visphi,visphierr,t3phi,t3phierr,ucoords,vcoords,u_coords,v_coords,eff_wave,vistels,tels),nan_policy='omit')
result = minner.minimize()
report_fit(result)

chi_sq = result.redchi
ratio_result = result.params['ratio'].value
ud_result = result.params['ud'].value
bw_result = result.params['bw'].value
alpha_result = result.params['alpha'].value

s1_result = result.params['s1'].value
s2_result = result.params['s2'].value
s3_result = result.params['s3'].value
s4_result = result.params['s4'].value
s5_result = result.params['s5'].value
s6_result = result.params['s6'].value

a1_result = result.params['a1'].value
a2_result = result.params['a2'].value
a3_result = result.params['a3'].value
a4_result = result.params['a4'].value
a5_result = result.params['a5'].value
a6_result = result.params['a6'].value
a7_result = result.params['a7'].value
a8_result = result.params['a8'].value
a9_result = result.params['a9'].value
a10_result = result.params['a10'].value
a11_result = result.params['a11'].value
a12_result = result.params['a12'].value
a13_result = result.params['a13'].value
a14_result = result.params['a14'].value
a15_result = result.params['a15'].value

b1_result = result.params['b1'].value
b2_result = result.params['b2'].value
b3_result = result.params['b3'].value
b4_result = result.params['b4'].value
b5_result = result.params['b5'].value
b6_result = result.params['b6'].value
b7_result = result.params['b7'].value
b8_result = result.params['b8'].value
b9_result = result.params['b9'].value
b10_result = result.params['b10'].value
b11_result = result.params['b11'].value
b12_result = result.params['b12'].value
b13_result = result.params['b13'].value
b14_result = result.params['b14'].value
b15_result = result.params['b15'].value

c1_result = result.params['c1'].value
c2_result = result.params['c2'].value
c3_result = result.params['c3'].value
c4_result = result.params['c4'].value
c5_result = result.params['c5'].value
c6_result = result.params['c6'].value
c7_result = result.params['c7'].value
c8_result = result.params['c8'].value
c9_result = result.params['c9'].value
c10_result = result.params['c10'].value
c11_result = result.params['c11'].value
c12_result = result.params['c12'].value
c13_result = result.params['c13'].value
c14_result = result.params['c14'].value
c15_result = result.params['c15'].value

## plot data with best fit model
best_params = Parameters()
best_params.add('ratio',value=ratio_result)
best_params.add('ud',value=ud_result)
best_params.add('bw',value=bw_result)
best_params.add('alpha',value=alpha_result)

best_params.add('s1',value=s1_result)
best_params.add('s2',value=s2_result)
best_params.add('s3',value=s3_result)
best_params.add('s4',value=s4_result)
best_params.add('s5',value=s5_result)
best_params.add('s6',value=s6_result)

best_params.add('a1',value=a1_result)
best_params.add('b1',value=b1_result)
best_params.add('c1',value=c1_result)
best_params.add('a2',value=a2_result)
best_params.add('b2',value=b2_result)
best_params.add('c2',value=c2_result)
best_params.add('a3',value=a3_result)
best_params.add('b3',value=b3_result)
best_params.add('c3',value=c3_result)
best_params.add('a4',value=a4_result)
best_params.add('b4',value=b4_result)
best_params.add('c4',value=c4_result)
best_params.add('a5',value=a5_result)
best_params.add('b5',value=b5_result)
best_params.add('c5',value=c5_result)
best_params.add('a6',value=a6_result)
best_params.add('b6',value=b6_result)
best_params.add('c6',value=c6_result)
best_params.add('a7',value=a7_result)
best_params.add('b7',value=b7_result)
best_params.add('c7',value=c7_result)
best_params.add('a8',value=a8_result)
best_params.add('b8',value=b8_result)
best_params.add('c8',value=c8_result)
best_params.add('a9',value=a9_result)
best_params.add('b9',value=b9_result)
best_params.add('c9',value=c9_result)
best_params.add('a10',value=a10_result)
best_params.add('b10',value=b10_result)
best_params.add('c10',value=c10_result)
best_params.add('a11',value=a11_result)
best_params.add('b11',value=b11_result)
best_params.add('c11',value=c11_result)
best_params.add('a12',value=a12_result)
best_params.add('b12',value=b12_result)
best_params.add('c12',value=c12_result)
best_params.add('a13',value=a13_result)
best_params.add('b13',value=b13_result)
best_params.add('c13',value=c13_result)
best_params.add('a14',value=a14_result)
best_params.add('b14',value=b14_result)
best_params.add('c14',value=c14_result)
best_params.add('a15',value=a15_result)
best_params.add('b15',value=b15_result)
best_params.add('c15',value=c15_result)

visphi_model = []
vis_model = []
dispersion_model = []

## correct for wavelength
lbd_ref = 1.60736e-06
eff_wave_cor = eff_wave[0]*(1+best_params['alpha']*(eff_wave[0]-lbd_ref))

for item1,item2,item4 in zip(ucoords,vcoords,vistels):    
    ## need to define etalon seps for each triangle
    if str(item4)==str(vistels[0]):
        delta_s = -2*(best_params['s1']-best_params['s2'])
        a = best_params['a1']
        b = best_params['b1']
        c = best_params['c1']
    if str(item4)==str(vistels[1]):
        delta_s = -2*(best_params['s1']-best_params['s3'])
        a = best_params['a2']
        b = best_params['b2']
        c = best_params['c2']
    if str(item4)==str(vistels[2]):
        delta_s = -2*(best_params['s1']-best_params['s4'])
        a = best_params['a3']
        b = best_params['b3']
        c = best_params['c3']
    if str(item4)==str(vistels[3]):
        delta_s = -2*(best_params['s1']-best_params['s5'])
        a = best_params['a4']
        b = best_params['b4']
        c = best_params['c4']
    if str(item4)==str(vistels[4]):
        delta_s = -2*(best_params['s1']-best_params['s6'])
        a = best_params['a5']
        b = best_params['b5']
        c = best_params['c5']
    if str(item4)==str(vistels[5]):
        delta_s = -2*(best_params['s2']-best_params['s3'])
        a = best_params['a6']
        b = best_params['b6']
        c = best_params['c6']
    if str(item4)==str(vistels[6]):
        delta_s = -2*(best_params['s2']-best_params['s4'])
        a = best_params['a7']
        b = best_params['b7']
        c = best_params['c7']
    if str(item4)==str(vistels[7]):
        delta_s = -2*(best_params['s2']-best_params['s5'])
        a = best_params['a8']
        b = best_params['b8']
        c = best_params['c8']
    if str(item4)==str(vistels[8]):
        delta_s = -2*(best_params['s2']-best_params['s6'])
        a = best_params['a9']
        b = best_params['b9']
        c = best_params['c9']
    if str(item4)==str(vistels[9]):
        delta_s = -2*(best_params['s3']-best_params['s4'])
        a = best_params['a10']
        b = best_params['b10']
        c = best_params['c10']
    if str(item4)==str(vistels[10]):
        delta_s = -2*(best_params['s3']-best_params['s5'])
        a = best_params['a11']
        b = best_params['b11']
        c = best_params['c11']
    if str(item4)==str(vistels[11]):
        delta_s = -2*(best_params['s3']-best_params['s6'])
        a = best_params['a12']
        b = best_params['b12']
        c = best_params['c12']
    if str(item4)==str(vistels[12]):
        delta_s = -2*(best_params['s4']-best_params['s5'])
        a = best_params['a13']
        b = best_params['b13']
        c = best_params['c13']
    if str(item4)==str(vistels[13]):
        delta_s = -2*(best_params['s4']-best_params['s6'])
        a = best_params['a14']
        b = best_params['b14']
        c = best_params['c14']
    if str(item4)==str(vistels[14]):
        delta_s = -2*(best_params['s5']-best_params['s6'])
        a = best_params['a15']
        b = best_params['b15']
        c = best_params['c15']
    p = np.poly1d([a,b,c])
    dispersion = p(eff_wave_cor)
    dispersion_model.append(dispersion)
    visphi_arr=[]
    for w in eff_wave_cor:
        complex_vis = cvis_model(best_params,delta_s,item1, item2, w)
        visibility = np.angle(complex_vis)*180/np.pi
        visphi_arr.append(visibility)
    visphi_arr=np.array(visphi_arr)
    dphase = visphi_arr[1:]-visphi_arr[:-1]
    dphase = np.insert(dphase,len(dphase),np.nan)
    visphi_model.append(dphase)
    vis_arr=[]
    for w in eff_wave_cor:
        complex_vis2 = cvis_model(best_params,delta_s,item1, item2, w)
        visibility2 = complex_vis2*np.conj(complex_vis2)
        vis_arr.append(visibility2.real)
    vis_model.append(vis_arr)
vis_model=np.array(vis_model)
visphi_model=np.array(visphi_model)
dispersion_model=np.array(dispersion_model)

cp_model = []
for item1,item2,item4 in zip(u_coords,v_coords,tels):
        
    ## need to define etalon seps for each triangle
    if str(item4)==str(tels[0]):
        delta_s1 = -2*(best_params['s1']-best_params['s2'])
        delta_s2 = -2*(best_params['s2']-best_params['s3'])
        delta_s3 = -2*(best_params['s3']-best_params['s1'])
    if str(item4)==str(tels[1]):
        delta_s1 = -2*(best_params['s1']-best_params['s2'])
        delta_s2 = -2*(best_params['s2']-best_params['s4'])
        delta_s3 = -2*(best_params['s4']-best_params['s1'])
    if str(item4)==str(tels[2]):
        delta_s1 = -2*(best_params['s1']-best_params['s2'])
        delta_s2 = -2*(best_params['s2']-best_params['s5'])
        delta_s3 = -2*(best_params['s5']-best_params['s1'])
    if str(item4)==str(tels[3]):
        delta_s1 = -2*(best_params['s1']-best_params['s2'])
        delta_s2 = -2*(best_params['s2']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s1'])
    if str(item4)==str(tels[4]):
        delta_s1 = -2*(best_params['s1']-best_params['s3'])
        delta_s2 = -2*(best_params['s3']-best_params['s4'])
        delta_s3 = -2*(best_params['s4']-best_params['s1'])
    if str(item4)==str(tels[5]):
        delta_s1 = -2*(best_params['s1']-best_params['s3'])
        delta_s2 = -2*(best_params['s3']-best_params['s5'])
        delta_s3 = -2*(best_params['s5']-best_params['s1'])
    if str(item4)==str(tels[6]):
        delta_s1 = -2*(best_params['s1']-best_params['s3'])
        delta_s2 = -2*(best_params['s3']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s1'])
    if str(item4)==str(tels[7]):
        delta_s1 = -2*(best_params['s1']-best_params['s4'])
        delta_s2 = -2*(best_params['s4']-best_params['s5'])
        delta_s3 = -2*(best_params['s5']-best_params['s1'])
    if str(item4)==str(tels[8]):
        delta_s1 = -2*(best_params['s1']-best_params['s4'])
        delta_s2 = -2*(best_params['s4']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s1'])
    if str(item4)==str(tels[9]):
        delta_s1 = -2*(best_params['s1']-best_params['s5'])
        delta_s2 = -2*(best_params['s5']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s1'])
    if str(item4)==str(tels[10]):
        delta_s1 = -2*(best_params['s2']-best_params['s3'])
        delta_s2 = -2*(best_params['s3']-best_params['s4'])
        delta_s3 = -2*(best_params['s4']-best_params['s2'])
    if str(item4)==str(tels[11]):
        delta_s1 = -2*(best_params['s2']-best_params['s3'])
        delta_s2 = -2*(best_params['s3']-best_params['s5'])
        delta_s3 = -2*(best_params['s5']-best_params['s2'])
    if str(item4)==str(tels[12]):
        delta_s1 = -2*(best_params['s2']-best_params['s3'])
        delta_s2 = -2*(best_params['s3']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s2'])
    if str(item4)==str(tels[13]):
        delta_s1 = -2*(best_params['s2']-best_params['s4'])
        delta_s2 = -2*(best_params['s4']-best_params['s5'])
        delta_s3 = -2*(best_params['s5']-best_params['s2'])
    if str(item4)==str(tels[14]):
        delta_s1 = -2*(best_params['s2']-best_params['s4'])
        delta_s2 = -2*(best_params['s4']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s2'])
    if str(item4)==str(tels[15]):
        delta_s1 = -2*(best_params['s2']-best_params['s5'])
        delta_s2 = -2*(best_params['s5']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s2'])
    if str(item4)==str(tels[16]):
        delta_s1 = -2*(best_params['s3']-best_params['s4'])
        delta_s2 = -2*(best_params['s4']-best_params['s5'])
        delta_s3 = -2*(best_params['s5']-best_params['s3'])
    if str(item4)==str(tels[17]):
        delta_s1 = -2*(best_params['s3']-best_params['s4'])
        delta_s2 = -2*(best_params['s4']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s3'])
    if str(item4)==str(tels[18]):
        delta_s1 = -2*(best_params['s3']-best_params['s5'])
        delta_s2 = -2*(best_params['s5']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s3'])
    if str(item4)==str(tels[19]):
        delta_s1 = -2*(best_params['s4']-best_params['s5'])
        delta_s2 = -2*(best_params['s5']-best_params['s6'])
        delta_s3 = -2*(best_params['s6']-best_params['s4'])
    phase_arr=[]
    for w in eff_wave_cor:
        complex_vis = cvis_model_cp(best_params,delta_s1,delta_s2,delta_s3,item1, item2, w)
        phase = (np.angle(complex_vis[0])+np.angle(complex_vis[1])+np.angle(complex_vis[2]))*180/np.pi
        phase_arr.append(phase)
    cp_model.append(phase_arr)
cp_model=np.array(cp_model)

## plot results
with PdfPages("/Users/tgardner/etalon_epochs/%(1)s_%(2)s_summary_%(3)s.pdf"%{"1":target_id,"2":date,"3":flag}) as pdf:

    ## plot dispersion model
    label_size = 4
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
    fig.subplots_adjust(hspace=0.5,wspace=.001)
    axs=axs.ravel()

    index = np.linspace(0,14,15)
    for ind in index:
        vphidata=[]
        vphierrdata=[]
        modeldata=[]
        uvis=[]
        vvis=[]
        for v,verr,mod,uc,vc,bl in zip(visphi,visphierr,dispersion_model,ucoords,vcoords,vistels):
            if str(bl)==str(vistels[int(ind)]):
                vphidata.append(v)
                vphierrdata.append(verr)
                modeldata.append(mod)
                uvis.append(uc)
                vvis.append(vc)
        vphidata=np.array(vphidata)
        vphierrdata=np.array(vphierrdata)
        modeldata=np.array(modeldata)
        uvis=np.array(uvis)
        vvis=np.array(vvis)

        for y,yerr,m,u,v in zip(vphidata,vphierrdata,modeldata,uvis,vvis):
            #x=np.sqrt(u**2+v**2)/eff_wave[0]
            x=eff_wave_cor
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'--',color='r',zorder=2)
        axs[int(ind)].set_title(str(vistels[int(ind)]))

    fig.suptitle('%s VisPhi'%target_id)
    fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
    fig.text(0.05, 0.5, 'VisPhi (deg)', va='center', rotation='vertical')
    pdf.savefig()
    plt.close()

    ## plot vis2 data
    label_size = 4
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
    fig.subplots_adjust(hspace=0.5,wspace=.001)
    axs=axs.ravel()

    index = np.linspace(0,14,15)
    for ind in index:
        v2data=[]
        v2errdata=[]
        modeldata=[]
        uvis=[]
        vvis=[]
        for v,verr,mod,uc,vc,bl in zip(vis2,vis2err,vis_model,ucoords,vcoords,vistels):
            if str(bl)==str(vistels[int(ind)]):
                v2data.append(v)
                v2errdata.append(verr)
                modeldata.append(mod)
                uvis.append(uc)
                vvis.append(vc)
        v2data=np.array(v2data)
        v2errdata=np.array(v2errdata)
        modeldata=np.array(modeldata)
        uvis=np.array(uvis)
        vvis=np.array(vvis)

        for y,yerr,m,u,v in zip(v2data,v2errdata,modeldata,uvis,vvis):
            #x=np.sqrt(u**2+v**2)/eff_wave[0]
            x=eff_wave_cor
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
        axs[int(ind)].set_title(str(vistels[int(ind)]))

    fig.suptitle('%s Vis2'%target_id)
    fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
    fig.text(0.05, 0.5, 'Vis2', va='center', rotation='vertical')
    pdf.savefig()
    plt.close()

    ## plot visphi data
    label_size = 4
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
    fig.subplots_adjust(hspace=0.5,wspace=.001)
    axs=axs.ravel()

    visphi_new = visphi - dispersion_model

    index = np.linspace(0,14,15)
    for ind in index:
        vphidata=[]
        vphierrdata=[]
        modeldata=[]
        uvis=[]
        vvis=[]
        for v,verr,mod,uc,vc,bl in zip(visphi_new,visphierr,visphi_model,ucoords,vcoords,vistels):
            if str(bl)==str(vistels[int(ind)]):
                vphidata.append(v)
                vphierrdata.append(verr)
                modeldata.append(mod)
                uvis.append(uc)
                vvis.append(vc)
        vphidata=np.array(vphidata)
        vphierrdata=np.array(vphierrdata)
        modeldata=np.array(modeldata)
        uvis=np.array(uvis)
        vvis=np.array(vvis)

        for y,yerr,m,u,v in zip(vphidata,vphierrdata,modeldata,uvis,vvis):
            #x=np.sqrt(u**2+v**2)/eff_wave[0]
            x=eff_wave_cor
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
        axs[int(ind)].set_title(str(vistels[int(ind)]))

    fig.suptitle('%s VisPhi'%target_id)
    fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
    fig.text(0.05, 0.5, 'VisPhi (deg)', va='center', rotation='vertical')
    pdf.savefig()
    plt.close()

    ## plot cp data
    label_size = 4
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    fig,axs = plt.subplots(4,5,figsize=(10,7),facecolor='w',edgecolor='k')
    fig.subplots_adjust(hspace=0.5,wspace=.001)
    axs=axs.ravel()

    index = np.linspace(0,19,20)
    for ind in index:
        t3data=[]
        t3errdata=[]
        modeldata=[]
        for t,terr,mod,tri in zip(t3phi,t3phierr,cp_model,tels):
            if str(tri)==str(tels[int(ind)]):
                t3data.append(t)
                t3errdata.append(terr)
                modeldata.append(mod)
        t3data=np.array(t3data)
        t3errdata=np.array(t3errdata)
        modeldata=np.array(modeldata)

        for y,yerr,m in zip(t3data,t3errdata,modeldata):
            x=eff_wave_cor
            axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-',zorder=1)
            axs[int(ind)].plot(x,m,'+--',color='r',zorder=2)
        axs[int(ind)].set_title(str(tels[int(ind)]))

    fig.suptitle('%s Closure Phase'%target_id)
    fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
    fig.text(0.05, 0.5, 'CP (deg)', va='center', rotation='vertical')
    pdf.savefig()
    plt.close()

## Now save a txt file with the best fit
etalon_fit = np.array([best_params['s1'],best_params['s2'],best_params['s3'],best_params['s4'],best_params['s5'],best_params['s6'],best_params['ratio'],best_params['ud'],best_params['bw']])
f = open("/Users/tgardner/etalon_epochs/%(1)s_fit_%(2)s.txt"%{"1":date,"2":flag},"w+")
f.write("# date mjd\r\n")
f.write("# s1 s2 s3 s4 s5 s6 ratio ud bw\r\n")
f.write("# eff_wave\r\n")
f.write("%s %s\r\n"%(date,np.around(np.nanmedian(time_obs),4)))
f.write("%s %s %s %s %s %s %s %s %s\r\n"%(etalon_fit[0],etalon_fit[1],etalon_fit[2],etalon_fit[3],etalon_fit[4],etalon_fit[5],etalon_fit[6],etalon_fit[7],etalon_fit[8]))
f.write(" ".join(map(str,eff_wave_cor)))
f.close()

#####################
## bootstrap for errors
#####################

## Split data by time
print('Shape of t3phi = ',t3phi.shape)
print('Shape of vis2 = ',vis2.shape)
print('Sahpe of tels = ',tels.shape)
num = 20
num2 = 15
t3phi_t = t3phi.reshape(int(len(t3phi)/num),num,len(t3phi[0]))
t3phierr_t = t3phierr.reshape(int(len(t3phierr)/num),num,len(t3phierr[0]))
tels_t = tels.reshape(int(len(tels)/num),num,len(tels[0]))
vis2_t = vis2.reshape(int(len(vis2)/num2),num2,len(vis2[0]))
vis2err_t = vis2err.reshape(int(len(vis2err)/num2),num2,len(vis2err[0]))
visphi_t = visphi.reshape(int(len(visphi)/num2),num2,len(visphi[0]))
visphierr_t = visphierr.reshape(int(len(visphierr)/num2),num2,len(visphierr[0]))
vistels_t = vistels.reshape(int(len(vistels)/num2),num2,len(vistels[0]))
u_coords_t = u_coords.reshape(int(len(u_coords)/num),num,len(u_coords[0]))
v_coords_t = v_coords.reshape(int(len(v_coords)/num),num,len(v_coords[0]))
ucoords_t = ucoords.reshape(int(len(ucoords)/num2),num2,1)
vcoords_t = vcoords.reshape(int(len(vcoords)/num2),num2,1)
print('New shape of t3phi = ',t3phi_t.shape)
print('New shape of vis2 = ',vis2_t.shape)

## perform bootstrap
s1_results=[]
s2_results=[]
s3_results=[]
s4_results=[]
s5_results=[]
s6_results=[]

for i in tqdm(np.arange(50)):
    r = np.random.randint(t3phi_t.shape[0],size=len(t3phi_t))
    t3phi_boot = t3phi_t[r,:]
    t3phierr_boot = t3phierr_t[r,:]
    u_coords_boot = u_coords_t[r,:]
    v_coords_boot = v_coords_t[r,:]
    tels_boot = tels_t[r,:]
    visphi_boot = visphi_t[r,:]
    visphierr_boot = visphierr_t[r,:]
    ucoords_boot = ucoords_t[r,:]
    vcoords_boot = vcoords_t[r,:]
    vis2_boot = vis2_t[r,:]
    vis2err_boot = vis2err_t[r,:]
    vistels_boot = vistels_t[r,:]

    t3phi_boot = t3phi_boot.reshape(int(t3phi_boot.shape[0])*int(t3phi_boot.shape[1]),t3phi_boot.shape[2])
    t3phierr_boot = t3phierr_boot.reshape(int(t3phierr_boot.shape[0])*int(t3phierr_boot.shape[1]),t3phierr_boot.shape[2])
    u_coords_boot = u_coords_boot.reshape(int(u_coords_boot.shape[0])*int(u_coords_boot.shape[1]),u_coords_boot.shape[2])
    v_coords_boot = v_coords_boot.reshape(int(v_coords_boot.shape[0])*int(v_coords_boot.shape[1]),v_coords_boot.shape[2])
    tels_boot = tels_boot.reshape(int(tels_boot.shape[0])*int(tels_boot.shape[1]),tels_boot.shape[2])
    visphi_boot = visphi_boot.reshape(int(visphi_boot.shape[0])*int(visphi_boot.shape[1]),visphi_boot.shape[2])
    visphierr_boot = visphierr_boot.reshape(int(visphierr_boot.shape[0])*int(visphierr_boot.shape[1]),visphierr_boot.shape[2])
    ucoords_boot = ucoords_boot.reshape(int(ucoords_boot.shape[0])*int(ucoords_boot.shape[1]))
    vcoords_boot = vcoords_boot.reshape(int(vcoords_boot.shape[0])*int(vcoords_boot.shape[1]))
    vis2_boot = vis2_boot.reshape(int(vis2_boot.shape[0])*int(vis2_boot.shape[1]),vis2_boot.shape[2])
    vis2err_boot = vis2err_boot.reshape(int(vis2err_boot.shape[0])*int(vis2err_boot.shape[1]),vis2err_boot.shape[2])
    vistels_boot = vistels_boot.reshape(int(vistels_boot.shape[0])*int(vistels_boot.shape[1]),vistels_boot.shape[2])    

    #do fit, minimizer uses LM for least square fitting of model to data
    minner = Minimizer(vis_minimizer, params, fcn_args=(vis2_boot,vis2err_boot,visphi_boot,visphierr_boot,t3phi_boot,t3phierr_boot,ucoords_boot,vcoords_boot,u_coords_boot,v_coords_boot,eff_wave,vistels_boot,tels_boot),nan_policy='omit')
    result = minner.minimize()
    s1_results.append(result.params['s1'].value)
    s2_results.append(result.params['s2'].value)
    s3_results.append(result.params['s3'].value)
    s4_results.append(result.params['s4'].value)
    s5_results.append(result.params['s5'].value)
    s6_results.append(result.params['s6'].value)

s1_results=np.array(s1_results)
s2_results=np.array(s2_results)
s3_results=np.array(s3_results)
s4_results=np.array(s4_results)
s5_results=np.array(s5_results)
s6_results=np.array(s6_results)

#############################################
## Compute etalon factor with error
#############################################
## gather bootstrap results
thickness = np.array([[i,j,k,l,m,n] for i,j,k,l,m,n in 
                    zip(s1_results,s2_results,s3_results,
                       s4_results,s5_results,s6_results)])
## Set a REFERENCE
#thickness_ref = np.array([6.6,30.3,0,18.7,24.8,12.6])*1e-6
thickness_ref = np.array([6.591269386838166, 30.30573561932599,
                        0.0, 18.725249334224807, 24.832348063211644,
                        12.643059171774237])*1e-6

## Exclude for 5T data?
if exclude_t == 'y':
    et_ind = int(input('Enter index to exclude: '))

## want opds, not thickness
diff=[]
diff_ref=[]

idx = [abs(i-j) for i,j in combinations(thickness[0],2)]
ind = np.argsort(idx)

for item in thickness:
    if exclude_t == 'y':
        item[et_ind] = np.nan
    d = [abs(i-j) for i,j in combinations(item,2)]
    diff.append(np.array(d)[ind]) 
diff=np.array(diff)

d = [abs(i-j) for i,j in combinations(thickness_ref,2)]
diff_ref.append(np.array(d)[ind])
diff_ref=np.array(diff_ref)

## mean ratios and standard errors
etalon_ratio = []
for item in diff:
    ## get rid of nans
    indices = ~np.isnan(item)
    item = item[indices]
    ## measure slope for scale factor
    fit,V = np.polyfit(diff_ref[0][indices],item,1,cov='True')
    etalon_ratio.append(fit[0])
etalon_ratio = np.array(etalon_ratio)

etalon_factor = np.mean(etalon_ratio)
etalon_factor_err = np.std(etalon_ratio)

plt.hist(etalon_ratio)
plt.title('Etalon factor = %s $\pm$ %s'%(np.around(etalon_factor,6),
                                         np.around(etalon_factor_err,6)))
plt.xlabel('Etalon factor')
plt.ylabel('#')
plt.savefig("/Users/tgardner/etalon_epochs/%s_%s_bootstrap_results.pdf"%(date,target_id))
plt.close()

## save etalon factors to txt file
date_mjd = np.mean(time_obs)
f = open("/Users/tgardner/etalon_epochs/%s_%s_factor.txt"%(date,target_id),"w+")
f.write("# date(mjd) etalon_factor error\r\n")
f.write("%s %s %s\r\n"%(date_mjd,etalon_factor,etalon_factor_err))
f.close()
