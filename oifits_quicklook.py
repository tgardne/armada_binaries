######################################################################
## Tyler Gardner
##
## Plot vis2/t3 data vs baseline
## Written for files produced by MIRCX pipeline
##
######################################################################

import numpy as np
eachindex = lambda lst: range(len(lst))
import os
import matplotlib.pyplot as plt
import astropy.io.fits as fits

def read_chara(dir,target_id):
    
    ## get information from fits file
    eff_wave = []
    ucoords = []
    vcoords = []
    vis2=[]
    vis2err=[]
    baseline=[]
    t3phi=[]
    t3phierr=[]
    u_coords=[]
    v_coords=[]
    baseline_t3=[]

    for file in os.listdir(dir):
        if file.endswith("fits"):

            filename = os.path.join(dir, file)
            hdu = fits.open(filename)
            oi_target = hdu[0].header['OBJECT']

            if oi_target==target_id:
                oi_vis2 = hdu['OI_VIS2'].data
                oi_t3 = hdu['OI_T3'].data

                wave = hdu['OI_WAVELENGTH'].data['EFF_WAVE'][1:-1]
                eff_wave.append(wave)

                for i in eachindex(oi_vis2):
                    v2 = oi_vis2[i]['VIS2DATA']
                    v2err = oi_vis2[i]['VIS2ERR']
                    v2flag = np.where(oi_vis2[i].field('FLAG')==True)
                    v2[v2flag] = np.nan
                    v2err[v2flag] = np.nan
                    vis2.append(v2[1:-1])
                    vis2err.append(v2err[1:-1])
                    ucoords.append(oi_vis2[i]['UCOORD'])
                    vcoords.append(oi_vis2[i]['VCOORD'])

                    bl = np.sqrt(oi_vis2[i]['UCOORD']**2+oi_vis2[i]['VCOORD']**2)/wave
                    baseline.append(bl)

                for i in eachindex(oi_t3):
                    t3 = oi_t3[i]['T3PHI']
                    t3err = oi_t3[i]['T3PHIERR']
                    t3flag = np.where(oi_t3[i].field('FLAG')==True)
                    t3[t3flag] = np.nan
                    t3err[t3flag] = np.nan
                    t3phi.append(t3[1:-1])
                    t3phierr.append(t3err[1:-1])
                    u1coord = oi_t3[i]['U1COORD']
                    v1coord = oi_t3[i]['V1COORD']
                    u2coord = oi_t3[i]['U2COORD']
                    v2coord = oi_t3[i]['V2COORD']
                    u3coord = -u1coord - u2coord
                    v3coord = -v1coord - v2coord
                    u_coords.append([u1coord,u2coord,u3coord])
                    v_coords.append([v1coord,v2coord,v3coord])
                    bl1 = np.sqrt(u1coord**2+v1coord**2)
                    bl2 = np.sqrt(u2coord**2+v2coord**2)
                    bl3 = np.sqrt(u3coord**2+u3coord**2)
                    baseline_t3.append(max(bl1,bl2,bl3)/wave)

            hdu.close()

    vis2=np.array(vis2)
    vis2err=np.array(vis2err)
    ucoords = np.array(ucoords)
    vcoords = np.array(vcoords)
    baseline = np.array(baseline)
    t3phi=np.array(t3phi)
    t3phierr=np.array(t3phierr)
    u_coords=np.array(u_coords)
    v_coords=np.array(v_coords)
    baseline_t3 = np.array(baseline_t3)
    
    return vis2,vis2err,t3phi,t3phierr,ucoords,vcoords,u_coords,v_coords,baseline,baseline_t3,eff_wave[0]

## select night, target
dir=input('Path to oifits directory: ')
target=input('Target (e.g. HD_206901): ')

vis2,vis2err,t3phi,t3phierr,ucoords,vcoords,u_coords,v_coords,baseline,baseline_t3,eff_wave = read_chara(dir,target)

## plot vis2 and t3phi data
fig,ax = plt.subplots(2)

fig.suptitle('%s'%target)

nmeasure = np.arange(vis2.shape[-1])
for i in nmeasure:
    ax[0].errorbar(baseline[:,i],vis2[:,i],vis2err[:,i],fmt='.')
ax[0].set_xlabel('B/$\lambda$ x 10$^{-6}$')
ax[0].set_ylabel('vis2')

for i in nmeasure:
    ax[1].errorbar(baseline_t3[:,i],t3phi[:,i],t3phierr[:,i],fmt='.')
ax[1].set_xlabel('B/$\lambda$ x 10$^{-6}$')
ax[1].set_ylabel('t3phi (deg)')

plt.show()