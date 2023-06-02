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
import pandas as pd
import matplotlib

matplotlib.rcParams['figure.figsize'] = (8, 5)
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['lines.markerfacecolor'] = 'black'
matplotlib.rcParams['lines.linewidth'] = 2.2
matplotlib.rcParams['patch.facecolor'] = 'None'


###########################################
## SETUP PATHS
###########################################

'''
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
'''
## Setup paths
dirpath = "/Users/suzutsuki-ch/Work/ARMADA/"
save_directory = "/Users/suzutsuki-ch/Work/ARMADA/Targets" ## path for saved files
armada_file = "/Users/suzutsuki-ch/Work/ARMADA/full_target_list.csv" ## path to csv target file
note = 'Chi' ## note for saved files (e.g. 'hip' for hipparcos distance, or 'gaia')

target_hd = '6456'
df = pd.read_csv(armada_file,dtype=object)

directory = "%s/HD_%s/"%(save_directory,target_hd)
if not os.path.exists(directory):
    print("Creating directory")
    os.makedirs(directory)

###########################################
## Specify Target
###########################################
"""
target_hd = input('Target HD #: ')
date = input('Note for savefile: ')
distance = float(input('Distance (pc): '))
mass_star1 = float(input('Mass Star 1 (Msun): '))
mass_star2 = float(input('Mass Star 2 (Msun): '))
"""

target_hd = "6456"
distance = 87.9
mass_star1 = 2.388
mass_star2 = 1.935

###########################################
## Read in planet injection files
###########################################
#detection_period = input('Period detection limit file: ')
#semi_period = input('Semi limit file: ')
#percentage_recovered = input('Percent recovered filed: ')

detection_period = np.load('/Users/suzutsuki-ch/Work/ARMADA/Targets/HD_6456/HD6456__bic_period.npy')
semi_period = np.load('/Users/suzutsuki-ch/Work/ARMADA/Targets/HD_6456/HD6456__bic_semi.npy')
percentage_recovered= np.load('/Users/suzutsuki-ch/Work/ARMADA/Targets/HD_6456/HD6456__bic_percent.npy')

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

plt.figure(facecolor="None")


plt.axhline(y=1000, color="#9A3324", lw=1.5, ls="--", label=r"$1M_\odot$")
plt.axhline(y=13, color="#9A3324", lw=1.5, ls="-.", label=r"$13M_J$")
plt.axhline(y=577, color="black", lw=1.5, ls="-", label=r"Mass Difference")
plt.contourf(X,mass_planet1,Z,levels=[0,10,68,99], extend='min', cmap='cividis')
plt.text(x=600,y=100,s="Unstable\nOrbit", fontsize=15, color="#D86018")
#plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet1_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet1_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.colorbar()
plt.axvspan(547, 6000, alpha=0.5, color='#D86018')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('planet period (days)')
plt.ylabel('planet mass ($M_J$)')
# plt.ylim(1,np.max(mass_planet1))
plt.ylim(1,2*10**3)
plt.xlim(np.min(detection_period),np.max(detection_period))
#plt.axis('equal')
plt.title('HD%s Primary Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
plt.legend()
plt.savefig("%s/HD_%s_%s_bis_masslimit1.png"%(directory,target_hd,note), dpi=240, bbox_inches="tight")
plt.show()
plt.close()

contour =  plt.contour(X,mass_planet2,Z,levels=[10,60,99],colors='#00274C')
plt.contour(X,mass_planet1,Z,levels=[10,60,99], color='#00B2A9')
plt.clabel(contour,inline=False,fmt='%1.0f',colors='black')
#plt.plot(detection_period,mass_planet2_25,'--',linewidth=2,color='black',label="25 $\mu$as wobble")
#plt.plot(detection_period,mass_planet2_50,'--',linewidth=2,color='black',label="50 $\mu$as wobble")
#plt.plot(detection_period,mass_planet2_100,'--',linewidth=2,color='black',label="100 $\mu$as wobble")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('planet period (days)')
plt.ylabel('planet mass ($M_J$)')
plt.ylim(1,np.max(mass_planet2))
plt.xlim(np.min(detection_period),np.max(detection_period))
#plt.axis('equal')
plt.title('HD%s Detection Limits'%target_hd)
#plt.colorbar(ticks=[20,40,60,80,100],label="planets recovered").set_ticklabels(['20%','40%','60%','80%','100%'])
plt.savefig("%s/HD_%s_%s_bic_masslimit2.pdf"%(directory,target_hd,note), dpi=240, bbox_inches="tight")
#plt.show()
