######################################################################
## Tyler Gardner
##
## Plot t3phi vs time for a channel
##
######################################################################

import numpy as np
import matplotlib.pyplot as plt
eachindex = lambda lst: range(len(lst))
import os
import matplotlib as mpl
from read_oifits import read_chara,read_vlti,read_chara_old
from astropy.io import fits

## select night, target
dir=input('Path to oifits directory: ')

dtype = input('chara / vlti / chara_old? ')
target_id=input('Target (e.g. HD_206901): ')

if dtype=='chara' or dtype=='chara_old':
    exclude=input('exclude a telescope (e.g. E1): ')

interact=input('interact with data? (y/n): ')

## get information from fits file
if dtype=='chara':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoord,vcoord,eff_wave,tels,vistels,time_obs,az = read_chara(dir,target_id,interact,exclude)
if dtype=='vlti':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoord,vcoord,eff_wave,tels,vistels,time_obs,flux_data,flux_err = read_vlti(dir,interact)
if dtype=='chara_old':
    t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoord,vcoord,eff_wave,tels,vistels,time_obs = read_chara_old(dir,interact,exclude)

print(t3phi.shape)
print(eff_wave.shape)
unique_tels = np.unique(tels,axis=0)
unique_vistels = np.unique(vistels,axis=0)
## plot t3phi data
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

print(t3phi.shape)
channel = input('Channel to plot (enter a to plot all): ')

if channel=='a':

    if dtype=='chara' or dtype=='chara_old':
        fig,axs = plt.subplots(4,5,figsize=(15,10),facecolor='w',edgecolor='k')
        index = np.arange(len(unique_tels))
    if dtype=='vlti':
        fig,axs = plt.subplots(2,2,figsize=(15,10),facecolor='w',edgecolor='k')
        index = np.arange(len(unique_tels))

    fig.subplots_adjust(hspace=0.5,wspace=.001)
    axs=axs.ravel()
    ymax = np.nanmax(t3phi)
    ymin = np.nanmin(t3phi)

    for channel in np.arange(t3phi.shape[-1]):

        for ind in index:
            t3data=[]
            t3errdata=[]
            tdata=[]
            for t,terr,tri,tt in zip(t3phi,t3phierr,tels,time_obs):
                if str(tri)==str(unique_tels[int(ind)]):
                    t3data.append(t)
                    t3errdata.append(terr)
                    tdata.append(tt)
            t3data=np.array(t3data)
            t3errdata=np.array(t3errdata)
            tdata=np.array(tdata)

            if channel==0 or channel==(t3phi.shape[-1]-1):
                axs[int(ind)].errorbar(tdata,t3data[:,channel],yerr=t3errdata[:,channel],fmt='.--',color='black')
            else:
                axs[int(ind)].errorbar(tdata,t3data[:,channel],yerr=t3errdata[:,channel],fmt='.--',color='lightgrey',zorder=0)
            axs[int(ind)].set_title(str(unique_tels[int(ind)]))
            axs[int(ind)].set_ylim(ymin,ymax)

    fig.suptitle('%s Closure Phase, Channel %s'%(target_id,channel))
    fig.text(0.5, 0.05, 'Time (mjd)', ha='center')
    fig.text(0.05, 0.5, 'CP (deg)', va='center', rotation='vertical')
    #plt.savefig('%s_phases.pdf'%target_id)
    plt.show()
else:
    
    if dtype=='chara' or dtype=='chara_old':
        fig,axs = plt.subplots(4,5,figsize=(10,7),facecolor='w',edgecolor='k')
        index = np.arange(len(unique_tels))
    if dtype=='vlti':
        fig,axs = plt.subplots(2,2,figsize=(10,7),facecolor='w',edgecolor='k')
        index = np.arange(len(unique_tels))

    fig.subplots_adjust(hspace=0.5,wspace=.001)
    axs=axs.ravel()

    ymax = np.nanmax(t3phi[:,int(channel)])
    ymin = np.nanmin(t3phi[:,int(channel)])

    for ind in index:
        t3data=[]
        t3errdata=[]
        tdata=[]
        for t,terr,tri,tt in zip(t3phi,t3phierr,tels,time_obs):
            if str(tri)==str(unique_tels[int(ind)]):
                t3data.append(t)
                t3errdata.append(terr)
                tdata.append(tt)
        t3data=np.array(t3data)
        t3errdata=np.array(t3errdata)  
        tdata=np.array(tdata)
        
        axs[int(ind)].errorbar(tdata,t3data[:,int(channel)],yerr=t3errdata[:,int(channel)],fmt='o--')
        axs[int(ind)].set_title(str(unique_tels[int(ind)]))
        axs[int(ind)].set_ylim(ymin,ymax)

    fig.suptitle('%s Closure Phase, Channel %s'%(target_id,channel))
    fig.text(0.5, 0.05, 'Time (mjd)', ha='center')
    fig.text(0.05, 0.5, 'CP (deg)', va='center', rotation='vertical')
    #plt.savefig('%s_phases.pdf'%target_id)
    plt.show()