######################################################################
## Tyler Gardner
##
## Do a grid search on rho, inclination, and bigomega
## Developed to make a detection of hot Jupiter ups And b (also works for binaries)
######################################################################

from chara_uvcalc import uv_calc
from binary_disk_point import binary_disk_point
from binary_disks_vector import binary_disks_vector
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.io.fits as fits
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
eachindex = lambda lst: range(len(lst))
from tqdm import tqdm
import os
import matplotlib.cm as cm
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mp
from PyAstronomy import pyasl
ks=pyasl.MarkleyKESolver()
from scipy.signal import medfilt
from scipy import stats

## function to compute expected sep,PA for each time
def companion_position(params,time):
    P = params['P']
    T = params['T']
    e = params['e']
    w = params['omega']*np.pi/180
    a = params['a']
    inc = params['inc']*np.pi/180
    bigw = params['bigomega']*np.pi/180

    A=a*(np.cos(bigw)*np.cos(w)-np.sin(bigw)*np.cos(inc)*np.sin(w))
    B=a*(np.sin(bigw)*np.cos(w)+np.cos(bigw)*np.cos(inc)*np.sin(w))
    F=a*(-np.cos(bigw)*np.sin(w)-np.sin(bigw)*np.cos(inc)*np.cos(w))
    G=a*(-np.sin(bigw)*np.sin(w)+np.cos(bigw)*np.cos(inc)*np.cos(w))

    M = 2*np.pi/P*(time-T)
    E = ks.getE(M,e)

    X = np.cos(E)-e
    Y = np.sqrt(1-e**2)*np.sin(E)
    dec = A*X+F*Y
    ra = B*X+G*Y
    return ra,dec

def companion_position_circular(params,time):
    P = params['P']
    T = params['T']
    e = 0
    w = params['omega']*np.pi/180
    a = params['a']
    inc = params['inc']*np.pi/180
    bigw = params['bigomega']*np.pi/180

    A=a*(np.cos(bigw)*np.cos(w)-np.sin(bigw)*np.cos(inc)*np.sin(w))
    B=a*(np.sin(bigw)*np.cos(w)+np.cos(bigw)*np.cos(inc)*np.sin(w))
    F=a*(-np.cos(bigw)*np.sin(w)-np.sin(bigw)*np.cos(inc)*np.cos(w))
    G=a*(-np.sin(bigw)*np.sin(w)+np.cos(bigw)*np.cos(inc)*np.cos(w))

    E = 2*np.pi/P*(time-T)

    X = np.cos(E)-e
    Y = np.sqrt(1-e**2)*np.sin(E)
    dec = A*X+F*Y
    ra = B*X+G*Y
    return ra,dec

## function which returns complex vis given sep, pa, flux ratio, HA, dec, UD1, UD2, wavelength
def cvis_model(params, u, v, wl, time,circular='n'):
    if circular=='y':
        ra = companion_position_circular(params,time)[0]
        dec = companion_position_circular(params,time)[1]
    else:
        ra = companion_position(params,time)[0]
        dec = companion_position(params,time)[1]
    ratio = params['ratio']
    bw = params['bw']
    ud = params['ud']
    ud2 = params['ud2']
    
    ul=np.array([u/i for i in wl])
    vl=np.array([v/i for i in wl])
    
    vis=binary_disks_vector().binary2(ul,vl,ra,dec,ratio,ud,ud2,bw)
    return vis

## function which returns residual of model and data to be minimized
def cp_minimizer(params,cp,cp_err,u_coords,v_coords,wl,time,circular='n'):
    model=[]
    for item1,item2,item4,item5 in zip(u_coords,v_coords,time,wl):
        complex_vis = cvis_model(params, item1, item2, item5, item4,circular)
        phase = (np.angle(complex_vis[:,0])+np.angle(complex_vis[:,1])+np.angle(complex_vis[:,2]))
        model.append(phase)
    model=np.array(model)

    ## need to do an "angle difference"
    data = cp*np.pi/180
    err = cp_err*np.pi/180
    
    diff = np.arctan2(np.sin(data-model),np.cos(data-model))
    return diff/err

## Ask the user which directory contains all files
dir=input('Path to oifits directory: ')
target_id=input('Target ID (e.g. HD_206901): ')
date = input('Date: ')
circular = input('Assume circular (y,[n])')

## get information from fits file
t3phi = []
t3phierr = []
u_coords = []
v_coords = []
eff_wave = []
time_obs=[]
#tels = []

ftype = input('chara/chara_old? ')

beam_map = {1:'S1',2:'S2',3:'E1',4:'E2',5:'W1',6:'W2'}

#start = 2
#end = 30
start = 1
end = 11

if ftype=='chara':
    for file in sorted(os.listdir(dir)):
        ## In mircx pipeline, reduced files end with oifits.fit if uncalibrated or viscal.fits if calibrated
        if file.endswith("_oifits.fits") or file.endswith("_viscal.fits") or file.endswith("_uvfix.fits"):

            filename = os.path.join(dir, file)
            hdu = fits.open(filename)
            oi_target = hdu[0].header['OBJECT']

            ## Check if oifits file is your target of interest
            if oi_target==target_id:
                #print(filename)
                oi_mjd = hdu[0].header['MJD-OBS']
                oi_t3 = hdu['OI_T3'].data
                oi_vis2 = hdu['OI_VIS2'].data

                ## t3phi data:
                for i in eachindex(oi_t3):
                    t3 = oi_t3[i]['T3PHI']
                    t3err = oi_t3[i]['T3PHIERR']
                    t3flag = np.where(oi_t3[i].field('FLAG')==True)
                    t3[t3flag] = np.nan
                    t3err[t3flag] = np.nan
                    t3phi.append(t3[start:end])
                    t3phierr.append(t3err[start:end])
                    #tels.append([beam_map[a] for a in oi_t3[i]['STA_INDEX']])
                    u1coord = oi_t3[i]['U1COORD']
                    v1coord = oi_t3[i]['V1COORD']
                    u2coord = oi_t3[i]['U2COORD']
                    v2coord = oi_t3[i]['V2COORD']
                    u3coord = -u1coord - u2coord
                    v3coord = -v1coord - v2coord
                    u_coords.append([u1coord,u2coord,u3coord])
                    v_coords.append([v1coord,v2coord,v3coord])

                    eff_wave.append(hdu['OI_WAVELENGTH'].data['EFF_WAVE'][start:end])
                    time_obs.append(oi_mjd)
            hdu.close()
else:
    try:
        hdu = fits.open(dir)
        for table in hdu:

            #if table.name=='OI_WAVELENGTH':
            #    eff_wave.append(table.data['EFF_WAVE'][1:33])

            ## t3phi data:
            wl_i=1
            if table.name=='OI_T3':
                for i in eachindex(table.data):
                    t3 = table.data[i]['T3PHI']
                    t3err = table.data[i]['T3PHIERR']
                    t3flag = np.where(table.data[i].field('FLAG')==True)
                    t3[t3flag] = np.nan
                    t3err[t3flag] = np.nan
                    t3phi.append(t3[start:end])
                    t3phierr.append(t3err[start:end])
                    #tels.append([beam_map[a] for a in table.data[i]['STA_INDEX']])
                    u1coord = table.data[i]['U1COORD']
                    v1coord = table.data[i]['V1COORD']
                    u2coord = table.data[i]['U2COORD']
                    v2coord = table.data[i]['V2COORD']
                    u3coord = -u1coord - u2coord
                    v3coord = -v1coord - v2coord
                    u_coords.append([u1coord,u2coord,u3coord])
                    v_coords.append([v1coord,v2coord,v3coord])
                    time_obs.append(table.data[i]['MJD'])
                    eff_wave.append(hdu['OI_WAVELENGTH',wl_i].data['EFF_WAVE'][start:end])
                wl_i+=1

        hdu.close()
    except:
        for file in sorted(os.listdir(dir)):
            #print(file)
            ## In mircx pipeline, reduced files end with oifits.fit if uncalibrated or viscal.fits if calibrated
            if file.endswith(".oifits"):

                filename = os.path.join(dir, file)
                print(filename)
                hdu = fits.open(filename)

                for table in hdu:

                    #if table.name=='OI_WAVELENGTH':
                    #    eff_wave.append(table.data['EFF_WAVE'][1:33])
                    ## t3phi data:
                    wl_i=1
                    if table.name=='OI_T3':
                        for i in eachindex(table.data):
                            t3 = table.data[i]['T3PHI']
                            t3err = table.data[i]['T3PHIERR']
                            t3flag = np.where(table.data[i].field('FLAG')==True)
                            t3[t3flag] = np.nan
                            t3err[t3flag] = np.nan
                            t3phi.append(t3[start:end])
                            t3phierr.append(t3err[start:end])
                            #tels.append([beam_map[a] for a in table.data[i]['STA_INDEX']])
                            u1coord = table.data[i]['U1COORD']
                            v1coord = table.data[i]['V1COORD']
                            u2coord = table.data[i]['U2COORD']
                            v2coord = table.data[i]['V2COORD']
                            u3coord = -u1coord - u2coord
                            v3coord = -v1coord - v2coord
                            u_coords.append([u1coord,u2coord,u3coord])
                            v_coords.append([v1coord,v2coord,v3coord])
                            time_obs.append(table.data[i]['MJD'])
                            eff_wave.append(hdu['OI_WAVELENGTH',wl_i].data['EFF_WAVE'][start:end])
                        wl_i+=1

                hdu.close()

t3phi = np.array(t3phi)
t3phierr = np.array(t3phierr)
u_coords = np.array(u_coords)
v_coords = np.array(v_coords)
eff_wave = np.array(eff_wave)
time_obs = np.array(time_obs)


print(t3phi.shape)
print(u_coords.shape)
print(eff_wave.shape)
print(time_obs.shape)

### get rid of crazy values
#idx = np.where(t3phierr>(3*np.nanmean(t3phierr)))
#t3phi[idx]=np.nan
#t3_std = np.nanstd(t3phi)
#print('Standard deviation t3phi = ',t3_std)
#idx = np.where(abs(t3phi)>(5*t3_std))
#t3phi[idx]=np.nan

print(t3phi.shape,t3phierr.shape)
print(u_coords.shape)
print(eff_wave.shape)
print(time_obs.shape)

## check for t3phi corrections based on cals (another script)
#if ftype=='chara' or ftype=='chara_old':
#    t3phi_corrected=np.empty((0,t3phi.shape[-1]))
#    t3phierr_corrected=np.empty((0,t3phierr.shape[-1]))
#    eff_wave_corrected=np.empty((0,eff_wave.shape[-1]))
#    u_coords_corrected=np.empty((0,u_coords.shape[-1]))
#    v_coords_corrected=np.empty((0,v_coords.shape[-1]))
#    time_obs_corrected=np.empty((0))
#    for file in os.listdir(dir):
#        if file.endswith("npy"):
#            filename = os.path.join(dir, file)
#            t3 = np.load(filename,allow_pickle=True)
#            print(t3.shape)
#            t3phi_corrected = np.append(t3phi_corrected,t3,axis=0)
#            t3phierr_corrected = np.append(t3phierr_corrected,t3[1],axis=0)
#            eff_wave_corrected = np.append(eff_wave_corrected,t3[2],axis=0)
#            u_coords_corrected = np.append(u_coords_corrected,t3[3],axis=0)
#            v_coords_corrected = np.append(v_coords_corrected,t3[4],axis=0)
#            time_obs_corrected = np.append(time_obs_corrected,t3[5],axis=0)
#    if t3phi_corrected.shape[0]>0:
#        t3phi = t3phi_corrected
#        t3phierr = t3phierr_corrected
#        eff_wave = eff_wave_corrected
#        u_coords = u_coords_corrected
#        v_coords = v_coords_corrected
#        time_obs = time_obs_corrected
#        t3phi = np.concatenate([t3phi,t3phi_corrected])
#        t3phierr = np.concatenate([t3phierr,t3phierr_corrected])
#        eff_wave = np.concatenate([eff_wave,eff_wave_corrected])
#        u_coords = np.concatenate([u_coords,u_coords_corrected])
#        v_coords = np.concatenate([v_coords,v_coords_corrected])
#        time_obs = np.concatenate([time_obs,time_obs_corrected])
#        print('Using CORRECTED t3phi')
#
#    print(t3phi.shape)
#    print(u_coords.shape)
#    print(eff_wave.shape)
#    print(time_obs.shape)
correct = input('Correct t3phi? (y/[n]): ')
if correct == 'y':
    correction_file = input('File with corrected t3phi: ')
    t3phi_corrected = np.load(correction_file)
    print(t3phi.shape)
    print(t3phi_corrected.shape)
    print('Using corrected t3phi')
    t3phi = t3phi_corrected

else:
    nloop = 0
    while nloop<3:
        t3phi_filtered = medfilt(t3phi,(1,5))
        t3phi_resid = t3phi-t3phi_filtered
        std_t3phi = np.nanstd(t3phi_resid)
        idx_t3phi = np.where(abs(t3phi_resid)>(5*std_t3phi))
        t3phi[idx_t3phi]=np.nan
        nloop+=1
std_t3phi = np.nanstd(t3phi)
print('Standard deviation t3phi = ',std_t3phi)

## average channels
avg_channels = input('Average channels together? (y/n): ')
if avg_channels=='y':
    t3phi_filtered = []
    t3phierr_filtered = []
    eff_wave_filtered = []
    for phi,err,wl in zip(t3phi,t3phierr,eff_wave):
        t3phi_filtered.append(np.nanmean(phi.reshape(-1,4),axis=1))
        #t3phi_filtered.append(np.nanmedian(phi.reshape(-1,4),axis=1))
        t3phierr_filtered.append(np.nanmedian(err.reshape(-1,4),axis=1))
        #t3phierr_filtered.append(np.nanstd(phi.reshape(-1,4),axis=1))
        eff_wave_filtered.append(np.nanmean(wl.reshape(-1,4),axis=1))
    t3phi=np.array(t3phi_filtered)
    t3phierr=np.array(t3phierr_filtered)
    eff_wave=np.array(eff_wave_filtered)

print('Minimum error = %s'%np.nanmin(t3phierr))
print('Maximum error = %s'%np.nanmax(t3phierr))
rid_error = input("Put a floor on low error bars? (y/[n]): ")

if rid_error == 'y':

    #error_val = float(input("Minimum error value (deg): "))
    #idx = np.where(t3phierr<error_val)
    #t3phi[idx]=np.nan

    print('Setting a floor on error bars')
    #t3phi_10 = np.nanpercentile(t3phierr,0.5)
    #t3phi_90 = np.nanpercentile(t3phierr,99.5)

    plt.hist(np.ndarray.flatten(t3phierr),bins=100)
    plt.show()
    error_median = np.nanmedian(t3phierr)
    error_floor = error_median/2
    print('Error floor = ',error_floor)

    idx1 = np.where(t3phierr<error_floor)
    #idx2 = np.where(t3phierr>t3phi_90)
    t3phierr[idx1]=error_floor
    #t3phierr[idx2]=np.nan
    #t3phi[idx1]=np.nan
    #t3phi[idx2]=np.nan

    print('New minimum error = %s'%np.nanmin(t3phierr))
    print('New Maximum error = %s'%np.nanmax(t3phierr))

#### plot t3phi data - 20 closing triangles for 6 telescopes
#for t,terr in zip(t3phi,t3phierr):
#    x=eff_wave[0]*1e6
#    plt.errorbar(x,t,yerr=terr,fmt='.-')
#
#    plt.title('%s Closure Phase'%target_id)
#    plt.show()
#    plt.close('all')

###########
## Now give orbital elements
###########
#P_guess=float(input('period (days):'))
#T_guess=float(input('tper (MJD):'))
#e_guess=float(input('eccentricity:'))
#omega_guess=float(input('omega (deg):'))
#ratio_guess = float(input('flux ratio (f1/f2): '))
#bw_guess = float(input('bw smearing (0.004): '))
#ud_guess = float(input('UD (mas): '))

#P_guess=4.617111
#T_guess=50033.55
#e_guess=0.012
#omega_guess=224.11
#ratio_guess = 5000
#bw_guess = 0.02
#ud_guess = 1.097
#ud2_guess = 0.1

## Tau Boo b
P_guess=3.3124568
T_guess= 59673.15
e_guess=0.011
omega_guess=113.4
ratio_guess = 3000
bw_guess = 0.02
ud_guess = 0.814
ud2_guess = 0.1
## semi-major should be 3.1mas
## inclination 43.5deg (Pelletier et al, 2021)

#P_guess=10.21296
#T_guess=52997.6813
#e_guess=0.00198
#omega_guess=102.427
#ratio_guess = 4.6
#bw_guess = 0.005
#ud_guess = 1.05
#ud2_guess = 0.6

## semi-major should be 4.4mas
## inclination 24deg (Piskorz et al, 2017)

###########
## Now compute chi-sq
###########
i_start = float(input('inclination start (deg): '))
i_end = float(input('inclination end (deg): '))
niter = int(input('Nsteps (e.g. 25): '))

bigw_start = float(input('bigw start (deg): '))
bigw_end = float(input('bigw end (deg): '))
nbigw_guess = int(input('Nsteps (e.g. 75): '))

fix_a = input('Fix semi-major? y/[n]: ')
if fix_a=='y':
    afix = float(input("Semi-major (mas)? "))
    a_grid = [afix]
else:
    a_start = float(input('a start (mas): '))
    a_end = float(input('a end (mas): '))
    na_guess = int(input('Nsteps (e.g. 25): '))
    a_grid = np.linspace(a_start,a_end,na_guess)
    
i_grid = np.linspace(i_start,i_end,niter)
bigomega_try = np.linspace(bigw_start,bigw_end,nbigw_guess)

# test
#nbigw_guess = 1
#bigomega_try = [225]

chi_sq = np.zeros(shape=(int(len(a_grid)*len(i_grid)),nbigw_guess))
a_results = np.zeros(shape=(int(len(a_grid)*len(i_grid)),nbigw_guess))
i_results = np.zeros(shape=(int(len(a_grid)*len(i_grid)),nbigw_guess))
bigw_results = np.zeros(shape=(int(len(a_grid)*len(i_grid)),nbigw_guess))
ratio_results = np.zeros(shape=(int(len(a_grid)*len(i_grid)),nbigw_guess))

#create a set of Parameters, choose starting value and range for search
params = Parameters()
params.add('P', value=P_guess, vary=False)
params.add('T', value=T_guess, vary=False)
if circular=='y':
    params.add('e', value=0.0, vary=False)
else:
    params.add('e', value=e_guess, vary=False)
params.add('omega', value=omega_guess, vary=False)
params.add('bw', value=0, vary=False)
params.add('ud', value=ud_guess, vary=False)
params.add('ud2', value=ud2_guess, vary=False)
params.add('a', value=1, vary=False)#min=0)
params.add('inc', value=1, vary=False)#min=0)
params.add('bigomega', value=1, vary=False)#min=0, max=360)
params.add('ratio', value=ratio_guess, min=1.0)

iternum = 0
for i_try in tqdm(i_grid):
    for a_try in a_grid:

        i_guess = 0

        for bigw_guess in bigomega_try:

            params['bigomega'].value = bigw_guess
            params['a'].value = a_try
            params['inc'].value = i_try

            #do fit, minimizer uses LM for least square fitting of model to data
            minner = Minimizer(cp_minimizer, params, fcn_args=(t3phi,t3phierr,u_coords,v_coords,eff_wave,time_obs,circular),nan_policy='omit')
            #result = minner.minimize()
            result = minner.leastsq(xtol=1e-5,ftol=1e-5)

            chi_sq[iternum,i_guess] = result.redchi
            bigw_results[iternum,i_guess] = result.params['bigomega'].value
            a_results[iternum,i_guess] = result.params['a'].value
            i_results[iternum,i_guess] = result.params['inc'].value
            ratio_results[iternum,i_guess] = result.params['ratio'].value

            i_guess+=1

        iternum+=1

    # write results
    np.save('./orbitsearch_cp/a_results_%s_%s.npy'%(target_id,date), a_results)
    np.save('./orbitsearch_cp/i_results_%s_%s.npy'%(target_id,date), i_results)
    np.save('./orbitsearch_cp/bigw_results_%s_%s.npy'%(target_id,date), bigw_results)
    np.save('./orbitsearch_cp/ratio_results_%s_%s.npy'%(target_id,date), ratio_results)
    np.save('./orbitsearch_cp/chi_sq_%s_%s.npy'%(target_id,date), chi_sq)

a_plot = []
inc_plot = []
bigw_plot = []
chi2_plot = []
ratio_plot = []
for i,j,k,l,m in zip(chi_sq,a_results,i_results,bigw_results,ratio_results):
    idx = np.argmin(i)
    chi2_plot.append(i[idx])
    a_plot.append(j[idx])
    inc_plot.append(k[idx])
    bigw_plot.append(l[idx])
    ratio_plot.append(m[idx])
a_plot = np.array(a_plot)
inc_plot = np.array(inc_plot)
bigw_plot = np.array(bigw_plot)
chi2_plot = np.array(chi2_plot)
ratio_plot = np.array(ratio_plot)

#report_fit(result)
index = np.argmin(chi2_plot)
print('-----RESULTS-------')
print('a = %s'%a_plot[index])
print('i = %s'%inc_plot[index])
print('bigw = %s'%bigw_plot[index])
print('ratio = %s'%ratio_plot[index])
print('redchi = %s'%chi2_plot[index])
print('-------------------')

best_fit = np.around(np.array([a_results[index],i_results[index],bigw_results[index],ratio_results[index]]),decimals=4)


## plot results
with PdfPages("./orbitsearch_cp/%s_%s_orbitsearch.pdf"%(target_id,date)) as pdf:
    ## first page - chisq grid
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(a_plot, inc_plot, c=1/chi2_plot, cmap=cm.inferno, s=150)
    plt.colorbar()
    ax.set_xlim(min(a_plot),max(a_plot))
    ax.set_ylim(min(inc_plot),max(inc_plot))
    plt.xlabel('a (mas)')
    plt.ylabel('i (deg)')
    #plt.title('Best Fit - %s'%best_fit)
    #plt.axis('equal')
    pdf.savefig()
    plt.close()
