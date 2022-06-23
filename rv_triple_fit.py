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
from astrometry_model import rv_model,rv_model_circular,rv_triple_model
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

period_guess = float(input('Period guess 1 (days) = '))
period2_guess = float(input('Period guess 2 (days) = '))

## randomize orbital elements
niter = 50
T_guess = np.random.uniform(min(t),max(t),niter)
K_guess = np.random.uniform(-30,30,niter)
gamma_guess = np.random.uniform(-20,20,niter)

T2_guess = np.random.uniform(min(t),max(t),niter)
K2_guess = np.random.uniform(-30,30,niter)
gamma2_guess = np.random.uniform(-20,20,niter)

circular = input("Circular? ([y]/n): ")
if circular=='n':
    w_guess = np.random.uniform(0,360,niter)
    e_guess = np.random.uniform(0,0.5,niter)
    w2_guess = np.random.uniform(0,360,niter)
    e2_guess = np.random.uniform(0,0.5,niter)

params = Parameters()
if circular=='n':
    params.add('w',   value= 1, min=0,max=360)
    params.add('e', value= 0.1, min=0,max=0.99)
    params.add('w2',   value= 1, min=0,max=360)
    params.add('e2', value= 0.1, min=0,max=0.99)
else:
    params.add('w',   value= 0, vary=False)
    params.add('e', value= 0, vary=False)
    params.add('w2',   value= 0, vary=False)
    params.add('e2', value= 0, vary=False)

params.add('P', value= period_guess, min=0)
params.add('T', value= 1, min=0)
params.add('K',value=1)
params.add('gamma',value=1)

params.add('P2', value= period2_guess, min=0)
params.add('T2', value= 1, min=0)
params.add('K2',value=1)

params_inner=[]
chi2 = []
iter_num = 0
for i in np.arange(niter):

    params['T'].value = T_guess[i]
    params['K'].value = K_guess[i]
    params['T2'].value = T2_guess[i]
    params['K2'].value = K2_guess[i]
    params['gamma'].value = gamma_guess[i]

    if circular=='n':
        params['w'].value = w_guess[i]
        params['e'].value = e_guess[i]
        params['w2'].value = w2_guess[i]
        params['e2'].value = e2_guess[i]
    #do fit, minimizer uses LM for least square fitting of model to data
    minner = Minimizer(rv_triple_model, params, fcn_args=(rv,t,err),
                    nan_policy='omit')
    result = minner.leastsq(xtol=1e-5,ftol=1e-5)
    
    params_inner.append([result.params['P'],result.params['e'],result.params['w']
                        ,result.params['T'],result.params['K'],result.params['gamma'],
                        result.params['P2'],result.params['e2'],result.params['w2']
                        ,result.params['T2'],result.params['K2']])
    chi2.append(result.chisqr)
    

params_inner=np.array(params_inner)
chi2 = np.array(chi2)
#chi2_noise = np.array(chi2_noise)

idx = np.argmin(chi2)
period_best = params_inner[:,0][idx]
period2_best = params_inner[:,6][idx]

## save parameter arrays
np.save('%s/HD%s_%s_params_rv.npy'%(directory,target_hd,date),params_inner)
np.save('%s/HD%s_%s_chi2_rv.npy'%(directory,target_hd,date),chi2)

print('Best inner periods = %s, %s'%(period_best,period2_best))

## Do a fit at best period
params = Parameters()
if circular=='n':
    params.add('w',   value= params_inner[:,2][idx], min=0,max=360)
    params.add('e', value= params_inner[:,1][idx], min=0,max=0.99)
    params.add('w2',   value= params_inner[:,8][idx], min=0,max=360)
    params.add('e2', value= params_inner[:,7][idx], min=0,max=0.99)
else:
    params.add('w',   value= params_inner[:,2][idx], vary=False)
    params.add('e', value= params_inner[:,1][idx], vary=False)
    params.add('w2',   value= params_inner[:,8][idx], vary=False)
    params.add('e2', value= params_inner[:,7][idx], vary=False)
params.add('P', value= params_inner[:,0][idx], min=0)
params.add('T', value= params_inner[:,3][idx], min=0)
params.add('K', value= params_inner[:,4][idx])
params.add('P2', value= params_inner[:,6][idx], min=0)
params.add('T2', value= params_inner[:,9][idx], min=0)
params.add('K2', value= params_inner[:,10][idx])
params.add('gamma', value= params_inner[:,5][idx])

#do fit, minimizer uses LM for least square fitting of model to data
minner = Minimizer(rv_triple_model, params, fcn_args=(rv,t,err),
                nan_policy='omit')
result = minner.minimize()
    
best = [result.params['P'],result.params['e'],result.params['w']
                    ,result.params['T'],result.params['K'],result.params['gamma'],
                    result.params['P2'],result.params['e2'],result.params['w2']
                    ,result.params['T2'],result.params['K2']]
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

MM2=[]
for i in tt:
    mm_anom=2*np.pi/result.params['P2']*(i-result.params['T2'])
    MM2.append(mm_anom)
MM2=np.asarray(MM2)
EE2=[]
for j in MM2:
    #ee_anom=keplerE(j,a1[1])
    ee_anom=ks.getE(j,result.params['e2'])
    EE2.append(ee_anom)
EE2=np.asarray(EE2)

f=2*np.arctan(np.sqrt((1+result.params['e'])/(1-result.params['e']))*np.tan(EE/2))
f2=2*np.arctan(np.sqrt((1+result.params['e2'])/(1-result.params['e2']))*np.tan(EE2/2))
y1=result.params['K']*(np.cos(result.params['w']+f)+result.params['e']*np.cos(result.params['w']))+result.params['K2']*(np.cos(result.params['w2']+f2)+result.params['e2']*np.cos(result.params['w2']))+result.params['gamma']
tt_fold=foldAt(tt,result.params['P'],T0=result.params['T'])

plt.errorbar(foldtime,rv,yerr=err,fmt='o')
plt.plot(tt_fold,y1,'--')
plt.ylabel('RV(km/s)')
plt.xlabel('Phase')
plt.title('RV Curve')
plt.savefig('%s/HD%s_%s_orbit_triple_rv.pdf'%(directory,target_hd,date))
plt.close()