######################################################################
## Tyler Gardner
##
## Plot mass limit vs period
## Requires a run of inject planet
## Need distance, m1, m2
##
######################################################################

import numpy as np
import os
import matplotlib.pyplot as plt

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
date = input('Note for savefile: ')
distance = float(input('Distance (pc): '))
mass_star1 = float(input('Mass Star 1 (Msun): '))
mass_star2 = float(input('Mass Star 2 (Msun): '))

###########################################
## Read in planet injection files
###########################################
detection_period = input('Period detection limit file: ')
semi_period = input('Period semi limit file: ')

detection_period = np.load(detection_period)
semi_period = np.load(semi_period)

abin1 = ((detection_period/365.25)**2*mass_star1)**(1/3)
abin2 = ((detection_period/365.25)**2*mass_star2)**(1/3)
mass_planet1 = mass_star1/(abin1-semi_period*distance/1000)*(semi_period*distance/1000)/0.0009546
mass_planet2 = mass_star2/(abin2-semi_period*distance/1000)*(semi_period*distance/1000)/0.0009546

plt.plot(detection_period,mass_planet1,'o-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Period (d)')
plt.ylabel('Mass (MJ)')
plt.title('Mass1 HD %s'%target_hd)
plt.savefig('%s/mass_limits/HD%s_%s_masslimit1.pdf'%(path,target_hd,date))
plt.close()

plt.plot(detection_period,mass_planet2,'o-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Period (d)')
plt.ylabel('Mass (MJ)')
plt.title('Mass2 HD %s'%target_hd)
plt.savefig('%s/mass_limits/HD%s_%s_masslimit2.pdf'%(path,target_hd,date))
plt.close()