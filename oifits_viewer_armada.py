######################################################################
## Tyler Gardner
##
## Plot vis2/t3/dphase data given oifits file
## Written for files produced by MIRCX or GRAVITY pipelines
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
label_size = 4
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

if dtype=='chara' or dtype=='chara_old':
    fig,axs = plt.subplots(4,5,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_tels))
if dtype=='vlti':
    fig,axs = plt.subplots(2,2,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_tels))

fig.subplots_adjust(hspace=0.5,wspace=.001)
axs=axs.ravel()

for ind in index:
    t3data=[]
    t3errdata=[]
    for t,terr,tri in zip(t3phi,t3phierr,tels):
        if str(tri)==str(unique_tels[int(ind)]):
            t3data.append(t)
            t3errdata.append(terr)
    t3data=np.array(t3data)
    t3errdata=np.array(t3errdata)

    for y,yerr in zip(t3data,t3errdata):
        x=eff_wave[0]
        axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-')
    axs[int(ind)].set_title(str(unique_tels[int(ind)]))

fig.suptitle('%s Closure Phase'%target_id)
fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
fig.text(0.05, 0.5, 'CP (deg)', va='center', rotation='vertical')
#plt.savefig('%s_phases.pdf'%target_id)
plt.show()

## plot vis2 data
if dtype=='chara' or dtype=='chara_old':
    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_vistels))
if dtype=='vlti':
    fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_vistels))

fig.subplots_adjust(hspace=0.5,wspace=.001)
axs=axs.ravel()

for ind in index:
    v2data=[]
    v2errdata=[]
    uvis=[]
    vvis=[]
    for v,verr,uc,vc,bl in zip(vis2,vis2err,ucoord,vcoord,vistels):
        if str(bl)==str(unique_vistels[int(ind)]):
            v2data.append(v)
            v2errdata.append(verr)
            uvis.append(uc)
            vvis.append(vc)
    v2data=np.array(v2data)
    v2errdata=np.array(v2errdata)
    uvis=np.array(uvis)
    vvis=np.array(vvis)

    for y,yerr,u,v in zip(v2data,v2errdata,uvis,vvis):
        #x=np.sqrt(u**2+v**2)/eff_wave[0]
        x=eff_wave[0]
        axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-')
    axs[int(ind)].set_title(str(unique_vistels[int(ind)]))

fig.suptitle('%s Vis2'%target_id)
fig.text(0.5, 0.05, 'B/$\lambda$', ha='center')
fig.text(0.05, 0.5, 'Vis^2', va='center', rotation='vertical')
plt.show()
#plt.savefig('%s_vis2.pdf'%target_id)

## plot dphase data
if dtype=='chara' or dtype=='chara_old':
    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_vistels))
if dtype=='vlti':
    fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_vistels))

fig.subplots_adjust(hspace=0.5,wspace=.001)
axs=axs.ravel()

for ind in index:
    v2data=[]
    v2errdata=[]
    uvis=[]
    vvis=[]
    for v,verr,uc,vc,bl in zip(visphi,visphierr,ucoord,vcoord,vistels):
        if str(bl)==str(unique_vistels[int(ind)]):
            v2data.append(v)
            v2errdata.append(verr)
            uvis.append(uc)
            vvis.append(vc)
    v2data=np.array(v2data)
    v2errdata=np.array(v2errdata)
    uvis=np.array(uvis)
    vvis=np.array(vvis)

    for y,yerr,u,v in zip(v2data,v2errdata,uvis,vvis):
        #x=np.sqrt(u**2+v**2)/eff_wave[0]
        x=eff_wave[0]
        axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-')
    axs[int(ind)].set_title(str(unique_vistels[int(ind)]))
fig.suptitle('%s Diff Phase'%target_id)
fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
fig.text(0.05, 0.5, 'Diff Phase', va='center', rotation='vertical')
plt.show()
#plt.savefig('%s_dphase.pdf'%target_id)

## plot visamp data
if dtype=='chara' or dtype=='chara_old':
    fig,axs = plt.subplots(3,5,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_vistels))
if dtype=='vlti':
    fig,axs = plt.subplots(2,3,figsize=(10,7),facecolor='w',edgecolor='k')
    index = np.arange(len(unique_vistels))

fig.subplots_adjust(hspace=0.5,wspace=.001)
axs=axs.ravel()

for ind in index:
    v2data=[]
    v2errdata=[]
    uvis=[]
    vvis=[]
    for v,verr,uc,vc,bl in zip(visamp,visamperr,ucoord,vcoord,vistels):
        if str(bl)==str(unique_vistels[int(ind)]):
            v2data.append(v)
            v2errdata.append(verr)
            uvis.append(uc)
            vvis.append(vc)
    v2data=np.array(v2data)
    v2errdata=np.array(v2errdata)
    uvis=np.array(uvis)
    vvis=np.array(vvis)

    for y,yerr,u,v in zip(v2data,v2errdata,uvis,vvis):
        #x=np.sqrt(u**2+v**2)/eff_wave[0]
        x=eff_wave[0]
        axs[int(ind)].errorbar(x,y,yerr=yerr,fmt='.-')
    axs[int(ind)].set_title(str(unique_vistels[int(ind)]))
fig.suptitle('%s VisAmp'%target_id)
fig.text(0.5, 0.05, 'Wavelength (m)', ha='center')
fig.text(0.05, 0.5, 'VisAmp', va='center', rotation='vertical')
plt.show()
#plt.savefig('%s_dphase.pdf'%target_id)