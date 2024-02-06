from scipy import stats
from scipy import integrate
from scipy import interpolate
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy import unravel_index
import pymultinest
from pymultinest import watch
from datetime import date
from datetime import datetime
import time
import json
from astropy.io import fits

########################################################################################
############ LOAD DETECTION LIMIT ######################################################
########################################################################################
now0 = datetime.now()

# Load combined detection limit map (sum of contrast curves of sample over separations and mass ratios)
detection_limit=fits.open('/nfs/turbo/lsa-mrmeyer/lsa-defurio/CANDID_1.0.5/CANDID-master/secondruntouse/detectionmap_secondrun.fits')
detection_limit=detection_limit[0].data

# Load properties of detections (sep, mass ratio)
observed_a, observed_q=np.loadtxt('/nfs/turbo/lsa-mrmeyer/lsa-defurio/MultiNest_companion_population/Sep6/paper_all_detections.dat', unpack=True, usecols=[0,1])
observed_q=np.array(observed_q)
observed_a=np.array(observed_a)
log_observed_a=np.log10(observed_a)

# Functional form of mass ratio distribution (power law)
def q_dist(q, qpow):
	if qpow < 0: f = (1-q)**(-qpow)
	elif qpow >= 0: f = q**qpow
	return f


# Functional form of separation distribution (log-normal)
def separation_distribution_log(a, logao = 1.0, sigmaao = 1.0):
	in_exp=-(((a-logao)**2)/(2*(sigmaao)**2))
	coeff=1/(np.sqrt(2*np.pi*sigmaao**2))
	return coeff*np.exp(in_exp)


sampling_size=10**4 # size of artificial companion population
detect=len(observed_q) # number of real detections
sample_size=54.0 # number of sources in sample

# Generate artificial companion population from sampled parameters
def model(alpha, loga, sigmaa):
# Mass Ratio
	alpha = alpha
	x = np.linspace(0.0,1,10**4)
	f = q_dist(x, alpha)
	qdist_sum=f/np.sum(f)
	qsamp=np.random.choice(x, size=sampling_size, p=qdist_sum)
# Separation Distribution
	a = np.linspace(0.01,16,10**4)
	a = np.log10(a)
	loga = np.log10(loga)
	sigmaa = sigmaa
	a_dist = separation_distribution_log(a, logao = loga, sigmaao = sigmaa)
	adist_sum=a_dist/np.sum(a_dist)
	asamp=np.random.choice(a, size=sampling_size, p=adist_sum)
	return qsamp, asamp

# Define priors for parameters that define the companion population
def prior(cube, ndim, nparams):
    cube[0] = cube[0]*10.0 - 5.0 # mass ratio power law index
    cube[1] = cube[1]*1.0 # Companion frequency
    cube[2] = cube[2]*9.9 + 0.1 # mean separation 
    cube[3] = cube[3]*2.0 + 0.1 # sigma of log-normal separation distribution
    return


# Define likelihood
def loglike(cube, ndim, nparams):
	alpha = cube[0]
	freq = cube[1]
	loga = cube[2]
	sigmaa = cube[3]

	loga = np.log10(loga)
	# Make a population weighted in the mass ratio distribution and the separation distribution
	# to be placed into the likelihood space given the detection limits
	q_pop, a_pop = model(alpha, loga, sigmaa)  # a_pop is in log space

	# Generate the separation and mass ratio distribution grid to construct the joint probability estimation
	a_log = np.linspace(-3,2,101) # in log(au)
	a_dist = separation_distribution_log(a_log, logao = loga, sigmaao = sigmaa)
	adist_sum=a_dist/np.sum(a_dist)
	q_range = np.linspace(0.0,1.0,21)
	q_distr = q_dist(q_range, alpha)
	qdist_sum = q_distr/np.sum(q_distr)

	# Set up a 3D data cube with separation, mass ratio, and probability
	AP, QP = np.meshgrid(adist_sum, qdist_sum)
	ZZ=AP*QP

	# Create joint probability by multiplying this grid by the detection limit map
	# where x (AP) corresponds to separation in log(au) and y (QP) corresponds to mass ratio
	joint = ZZ*detection_limit

	# Now iterate through true detections in terms of separation and mass ratio and
	# assign a probability to each from joint probability (model x detectionlimit)

	prob_n=0 # probability to change the likelihood function
	for i in range(len(observed_q)):
		binn=0
		for j in range(len(q_range)-1):
			for k in range(len(a_log)-1):
				# conditional statement that identifies the bin where the detection exists, gives joint probability of detecting that companion
				if  q_range[j] < observed_q[i] and observed_q[i] <= q_range[j+1] and a_log[k] < log_observed_a[i] and log_observed_a[i] <= a_log[k+1]:
					if joint[j,k] == 0.0: prob_n+=0.0
					else: prob_n+=np.log10(joint[j,k])
					binn+=1
					if binn != 1: print('THIS IS A PROBLEM YOU NEED TO EDIT THE CODE ########################')


	# Iterate through the artificial companion population and determine what is detected
	kkk=0.0
	for i in range(len(q_pop)):
		for j in range(len(q_range)-1):
			for k in range(len(a_log)-1):
				# conditional statement that identifies the bin where the detection exists, gives probability of detecting that companion
				if q_range[j] < q_pop[i] and q_pop[i] <= q_range[j+1] and a_log[k] < a_pop[i] and a_pop[i] <= a_log[k+1]:
					if joint[j,k] == 0.0: kkk+=0.0
					else: kkk += detection_limit[j,k]*((freq*sample_size)/sampling_size)
	if kkk == 0.0: poisson_likelihood=0
	else: 
		poisson_likelihood=(kkk**detect)*np.exp(-kkk)/np.math.factorial(detect)
		poisson_likelihood=np.log10(poisson_likelihood)
	loglikelihood = prob_n+poisson_likelihood
	return loglikelihood

# Setup and outputs for MultiNest
datafile='./Sep6/lognorma_powerq/outputs_chara_secondrun_lin/'
parameters = ["alpha", "freq", "loga", "sigmaa"]
n_params = len(parameters)

json.dump(parameters, open(datafile + '_params.json', 'w')) # save parameter names

# Run MultiNest
iterations=5000
now = datetime.now()
pymultinest.run(loglike, prior, n_params, importance_nested_sampling = True, resume = False, verbose = True, sampling_efficiency = 'model', max_iter=iterations, n_live_points = 400, outputfiles_basename=datafile + '_')

# Get MultiNest outputs
aa = pymultinest.Analyzer(outputfiles_basename=datafile + '_', n_params = n_params)
a_lnZ = aa.get_stats()['global evidence']
bestfit = aa.get_best_fit()
bestfit_parameters = np.array(bestfit['parameters'])
bestfit_loglike = bestfit['log_likelihood']
print(bestfit_parameters)
print(bestfit_loglike)
print('************************')
print('MAIN RESULT: Evidence Z ')
print('************************')
print('  log Z for model with 1 line = %.3f' % (a_lnZ))
print()
now1 = datetime.now()
time_to_run=now1-now0
print('Time to run code: ', time_to_run)
