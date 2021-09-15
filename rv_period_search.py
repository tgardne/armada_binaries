######################################################################
## Tyler Gardner
##
## Pipeline to fit binary orbits
## and search for additional companions
##
## For binary orbits from MIRCX/GRAVITY
##
######################################################################

from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
import matplotlib.cm as cm
from read_data import read_data,read_rv,read_wds,read_orb6
from astrometry_model import astrometry_model,rv_model,rv_model_circular,triple_model,triple_model_circular,lnlike,lnprior,lnpost,create_init
from orbit_plotting import orbit_model,triple_orbit_model
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import random
from PyAstronomy.pyasl import foldAt
from PyAstronomy import pyasl
ks=pyasl.MarkleyKESolver()

###########################################
## SETUP PATHS
###########################################

if os.getcwd()[7:14] == 'tgardne':
    ## setup paths for user
    path = '/Users/tgardne/ARMADA_orbits'
    path_etalon = '/Users/tgardne/etalon_epochs/etalon_fits/etalon_factors_fit.txt'
    path_wds = '/Users/tgardne/wds_targets'
    path_orb6 = '/Users/tgardne/catalogs/orb6orbits.sql.txt'
    
elif os.getcwd()[7:19] == 'adam.scovera':
    ## Adam's path
    path = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/ARMADA_orbits'
    path_etalon = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/etalon_factors_fit.txt'
    path_wds = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/wds_targets'
    path_orb6 = '/Users/adam.scovera/Documents/UofM/BEPResearch_Data/orb6orbits.sql.txt'

###########################################
## Specify Target
###########################################
target_hd = input('Target HD #: ')
date = input('Date for savefile: ')
#target = input('Target HIP #: ')
#target_wds = input('Target WDS #: ')

directory='%s/HD%s_rv/'%(path,target_hd)
if not os.path.exists(directory):
    os.makedirs(directory)

###########################################
## Read in ARMADA data
###########################################
file=open('%s/HD_%s_rv.txt'%(path,target_hd))
weight=1

t,rv,err = read_rv(file,weight)
file.close()

plt.errorbar(t,rv,err,fmt='o')
plt.xlabel('Time')
plt.ylabel('RV')
plt.show()

##########################################
## Grid Search for Period
##########################################

## New test -- try period spacing from PHASES III paper
time_span = max(t) - min(t)
print('Time span of data = %s days'%time_span)
f = 5
min_per = float(input('minimum period to search (days) = '))
#min_per = 2
max_k = int(2*f*time_span / min_per)
k_range = np.arange(max_k)[:-1] + 1
P2 = 2*f*time_span / k_range
P2 = np.logspace(np.log10(50),np.log10(500),1000)
print('Min/Max period (days) = %s / %s ; %s steps'%(min(P2),max(P2),len(k_range)))

print('Grid Searching over period')
params_inner=[]
chi2 = []
#chi2_noise = []

## randomize orbital elements
niter = 20
T_guess = np.random.uniform(min(t),max(t),niter)
K_guess = np.random.uniform(1,50,niter)
gamma_guess = np.random.uniform(-20,20,niter)

params = Parameters()
params.add('w',   value= 0, vary=False)
params.add('e', value= 0, vary=False)
params.add('P', value= 1, vary=False)
params.add('T', value= 1, min=0)
params.add('K',value=1,min=0)
params.add('gamma',value=1)

iter_num = 0
for period in tqdm(P2):

    params_n=[]
    chi2_n = []
    #chi2_noise_n = []

    for i in np.arange(niter):

        params['T'].value = T_guess[i]
        params['K'].value = K_guess[i]
        params['gamma'].value = gamma_guess[i]
        params['P'].value = period

        #do fit, minimizer uses LM for least square fitting of model to data
        minner = Minimizer(rv_model_circular, params, fcn_args=(rv,t,err),
                          nan_policy='omit')
        result = minner.leastsq(xtol=1e-5,ftol=1e-5)
        params_n.append([period,result.params['e'],result.params['w']
                            ,result.params['T'],result.params['K'],result.params['gamma']])
        #chi2_n.append(result.redchi)
        chi2_n.append(result.chisqr)

    params_n=np.array(params_n)
    chi2_n = np.array(chi2_n)
    #chi2_noise_n = np.array(chi2_noise_n)

    idx = np.argmin(chi2_n)
    #idx_n = np.argmin(chi2_noise_n)
    chi2.append(chi2_n[idx])
    #chi2_noise.append(chi2_noise_n[idx_n])
    params_inner.append(params_n[idx])
    

params_inner=np.array(params_inner)
chi2 = np.array(chi2)
#chi2_noise = np.array(chi2_noise)

idx = np.argmin(chi2)
period_best = params_inner[:,0][idx]

## save parameter arrays
np.save('%s/HD%s_%s_params_rv.npy'%(directory,target_hd,date),params_inner)
np.save('%s/HD%s_%s_chi2_rv.npy'%(directory,target_hd,date),chi2)

#plt.plot(params_inner[:,0],1/chi2_noise,'.--')
plt.plot(params_inner[:,0],1/chi2,'o-')
plt.xscale('log')
plt.xlabel('Period (d)')
plt.ylabel('1/chi2')
plt.title('Best Period = %s'%period_best)
plt.savefig('%s/HD%s_%s_chi2_period_rv.pdf'%(directory,target_hd,date))
plt.close()

print('Best inner period = %s'%period_best)

## Do a fit at best period
params = Parameters()
params.add('w',   value= params_inner[:,2][idx], vary=False)
params.add('e', value= params_inner[:,1][idx], vary=False)
params.add('P', value= params_inner[:,0][idx], min=0)
params.add('T', value= params_inner[:,3][idx], min=0)
params.add('K', value= params_inner[:,4][idx], min=0)
params.add('gamma', value= params_inner[:,5][idx])

#do fit, minimizer uses LM for least square fitting of model to data
minner = Minimizer(rv_model_circular, params, fcn_args=(rv,t,err),
                  nan_policy='omit')
result = minner.minimize()
best = [result.params['P'],result.params['e'],result.params['w']
                    ,result.params['T'],result.params['K'],result.params['gamma']]
try:
    report_fit(result)
except:
    print('-'*10)
    print('Fit FAILED!!!!')
    print('-'*10)

#Plot results:
foldtime= foldAt(t,result.params['P'],T0=result.params['T'])
tt=np.linspace(result.params['T'],result.params['T']+result.params['P'],100)
MM=[]
for i in tt:
    mm_anom=2*np.pi/result.params['P']*(i-result.params['T'])
    MM.append(mm_anom)
MM=np.asarray(MM)
EE=[]
for j in MM:
    #ee_anom=keplerE(j,a1[1])
    ee_anom=ks.getE(j,result.params['e'])
    EE.append(ee_anom)
EE=np.asarray(EE)
f=2*np.arctan(np.sqrt((1+result.params['e'])/(1-result.params['e']))*np.tan(EE/2))
y1=result.params['K']*(np.cos(result.params['w']+f)+result.params['e']*np.cos(result.params['w']))+result.params['gamma']
tt_fold=foldAt(tt,result.params['P'],T0=result.params['T'])

plt.errorbar(foldtime,rv,yerr=err,fmt='o')
plt.plot(tt_fold,y1,'--')
plt.ylabel('RV(km/s)')
plt.xlabel('Phase')
plt.title('RV Curve')
plt.savefig('%s/HD%s_%s_orbit_rv.pdf'%(directory,target_hd,date))
plt.close()