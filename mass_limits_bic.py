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
semi_period = input('Semi limit file: ')
percentage_recovered = input('Percent recovered filed: ')

detection_period = np.load(detection_period)
semi_period = np.load(semi_period)
percentage_recovered= np.load(percentage_recovered)

semi_25 = np.full(len(detection_period),0.025)
semi_50 = np.full(len(detection_period),0.05)
semi_100 = np.full(len(detection_period),0.1)

abin1 = ((detection_period/365.25)**2*mass_star1)**(1/3)
abin2 = ((detection_period/365.25)**2*mass_star2)**(1/3)
mass_planet1_25 = mass_star1/(abin1-semi_25*distance/1000)*(semi_25*distance/1000)/0.0009546
mass_planet1_50 = mass_star1/(abin1-semi_50*distance/1000)*(semi_50*distance/1000)/0.0009546
mass_planet1_100 = mass_star1/(abin1-semi_100*distance/1000)*(semi_100*distance/1000)/0.0009546
mass_planet2_25 = mass_star2/(abin2-semi_25*distance/1000)*(semi_25*distance/1000)/0.0009546
mass_planet2_50 = mass_star2/(abin2-semi_50*distance/1000)*(semi_50*distance/1000)/0.0009546
mass_planet2_100 = mass_star2/(abin2-semi_100*distance/1000)*(semi_100*distance/1000)/0.0009546

X,Y = np.meshgrid(detection_period,semi_period)
Z = np.swapaxes(percentage_recovered,0,1)

abin1 = ((detection_period/365.25)**2*mass_star1)**(1/3)
abin2 = ((detection_period/365.25)**2*mass_star2)**(1/3)
mass_planet1 = mass_star1/(abin1-Y*distance/1000)*(Y*distance/1000)/0.0009546
mass_planet2 = mass_star2/(abin2-Y*distance/1000)*(Y*distance/1000)/0.0009546

contour =  plt.contour(X,mass_planet1,Z,levels=[10,60,99],colors='grey')
plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet1_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('planet period (days)')
plt.ylabel('planet mass ($M_J$)')
plt.ylim(1,1000)
#plt.axis('equal')
plt.title('HD%s Primary Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
#plt.legend()
plt.savefig('%s/mass_limits/HD%s_%s_bic_masslimit1.pdf'%(path,target_hd,date))
plt.close()

contour =  plt.contour(X,mass_planet2,Z,levels=[10,60,99],colors='grey')
plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet2_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet2_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet2_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('planet period (days)')
plt.ylabel('planet mass ($M_J$)')
plt.ylim(1,1000)
#plt.axis('equal')
plt.title('HD%s Secondary Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
#plt.legend()
plt.savefig('%s/mass_limits/HD%s_%s_bic_masslimit2.pdf'%(path,target_hd,date))