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
    path = '/Users/tgardner/ARMADA_orbits'
    path_etalon = '/Users/tgardner/etalon_epochs/etalon_fits/etalon_factors_fit.txt'
    path_wds = '/Users/tgardner/wds_targets'
    path_orb6 = '/Users/tgardner/catalogs/orb6orbits.sql.txt'
    
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
#mass_planet1_25 = mass_star1/(abin1-semi_25*distance/1000)*(semi_25*distance/1000)/0.0009546
#mass_planet1_50 = mass_star1/(abin1-semi_50*distance/1000)*(semi_50*distance/1000)/0.0009546
#mass_planet1_100 = mass_star1/(abin1-semi_100*distance/1000)*(semi_100*distance/1000)/0.0009546
#mass_planet2_25 = mass_star2/(abin2-semi_25*distance/1000)*(semi_25*distance/1000)/0.0009546
#mass_planet2_50 = mass_star2/(abin2-semi_50*distance/1000)*(semi_50*distance/1000)/0.0009546
#mass_planet2_100 = mass_star2/(abin2-semi_100*distance/1000)*(semi_100*distance/1000)/0.0009546

X1,Y1 = np.meshgrid(abin1,semi_period)
X2,Y2 = np.meshgrid(abin2,semi_period)
Z = np.swapaxes(percentage_recovered,0,1)

mass_planet1 = mass_star1/(abin1-Y1*distance/1000)*(Y1*distance/1000)/0.0009546
mass_planet2 = mass_star2/(abin2-Y2*distance/1000)*(Y2*distance/1000)/0.0009546

ratio1 = mass_planet1 * 0.0009546 / mass_star1
ratio2 = mass_planet2 * 0.0009546 / mass_star2

contour =  plt.contour(X1,mass_planet1,Z,levels=[10,60,99],colors='grey')
plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet1_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('semi major (au)')
plt.ylabel('planet mass ($M_J$)')
plt.ylim(1,np.max(mass_planet1))
plt.xlim(np.min(abin1),np.max(abin1))
#plt.axis('equal')
plt.title('HD%s Primary Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
#plt.legend()
plt.savefig('%s/mass_limits/HD%s/HD%s_%s_masslimit1.pdf'%(path,target_hd,target_hd,date))
plt.close()

contour =  plt.contour(X1,ratio1,Z,levels=[10,60,99],colors='grey',extend='max')
plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet1_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('semi major (au)')
plt.ylabel('mass ratio')
#plt.ylim(np.min(ratio1),np.max(ratio1))
plt.ylim(np.min(np.abs(ratio1)),1)
plt.xlim(np.min(abin1),np.max(abin1))
#plt.axis('equal')
plt.title('HD%s Primary Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
#plt.legend()
plt.savefig('%s/mass_limits/HD%s/HD%s_%s_ratiolimit1.pdf'%(path,target_hd,target_hd,date))
plt.close()

contour =  plt.contour(X2,mass_planet2,Z,levels=[10,60,99],colors='grey')
plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet1_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('semi major (au)')
plt.ylabel('planet mass ($M_J$)')
plt.ylim(1,np.max(mass_planet2))
plt.xlim(np.min(abin2),np.max(abin2))
#plt.axis('equal')
plt.title('HD%s Secondary Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
#plt.legend()
plt.savefig('%s/mass_limits/HD%s/HD%s_%s_masslimit2.pdf'%(path,target_hd,target_hd,date))
plt.close()

contour =  plt.contour(X2,ratio2,Z,levels=[10,60,99],colors='grey',extend='max')
plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet1_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('semi major (au)')
plt.ylabel('mass ratio')
#plt.ylim(np.min(ratio2),np.max(ratio2))
plt.ylim(np.min(np.abs(ratio2)),1)
plt.xlim(np.min(abin2),np.max(abin2))
#plt.axis('equal')
plt.title('HD%s Secondary Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
#plt.legend()
plt.savefig('%s/mass_limits/HD%s/HD%s_%s_ratiolimit2.pdf'%(path,target_hd,target_hd,date))
plt.close()

## save parameter arrays
np.save('%s/mass_limits/HD%s/HD%s_%s_mass1.npy'%(path,target_hd,target_hd,date),mass_planet1)
np.save('%s/mass_limits/HD%s/HD%s_%s_mass2.npy'%(path,target_hd,target_hd,date),mass_planet2)
np.save('%s/mass_limits/HD%s/HD%s_%s_ratio1.npy'%(path,target_hd,target_hd,date),ratio1)
np.save('%s/mass_limits/HD%s/HD%s_%s_ratio2.npy'%(path,target_hd,target_hd,date),ratio2)
np.save('%s/mass_limits/HD%s/HD%s_%s_a1.npy'%(path,target_hd,target_hd,date),abin1)
np.save('%s/mass_limits/HD%s/HD%s_%s_a2.npy'%(path,target_hd,target_hd,date),abin2)