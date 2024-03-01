######################################################################
## Tyler Gardner
## 
## Read oifits file for t3,vis2,visphi
## arguments: directory of oifits, target name, y/n for interactive session
## returns: t3phi, vis2, visphi (with errors)
######################################################################

import numpy as np
eachindex = lambda lst: range(len(lst))
import os
import matplotlib.pyplot as plt
import astropy.io.fits as fits

def read_chara(dir,target_id,interact='n',exclude='',bl_drop='n'):

    beam_map = {1:'S1',2:'S2',3:'E1',4:'E2',5:'W1',6:'W2'}
    #beam_map = {0:'S1',1:'S2',2:'E1',3:'E2',4:'W1',5:'W2'}

    ## get information from fits file
    t3phi=[]
    t3phierr=[]
    eff_wave = []
    time_obs = []
    az = []
    tels = []
    u_coords = []
    v_coords = []
    vis2=[]
    vis2err=[]
    visphi=[]
    visphierr=[]
    visamp=[]
    visamperr=[]
    vistels=[]
    ucoords=[]
    vcoords=[]

    #start = 1
    #end = -1
    for file in os.listdir(dir):
        if file.endswith("_oifits.fits") or file.endswith("_viscal.fits") or file.endswith("_uvfix.fits"):

            filename = os.path.join(dir, file)
            hdu = fits.open(filename)
            oi_target = hdu[0].header['OBJECT']

            if oi_target==target_id:
                print(filename)
                oi_mjd = hdu[0].header['MJD-OBS']
                oi_t3 = hdu['OI_T3'].data
                oi_vis2 = hdu['OI_VIS2'].data
                oi_vis = hdu['OI_VIS'].data

                eff_wave.append(hdu['OI_WAVELENGTH'].data['EFF_WAVE'])
                time_obs.append(oi_mjd)

                az.append(hdu[0].header['AZ'])

                for i in eachindex(oi_t3):
                    ## use this to exclude a telescope
                    stations = [beam_map[a] for a in oi_t3[i]['STA_INDEX']]
                    if exclude in stations:
                        phases_empty = np.empty(oi_t3[i]['T3PHI'].shape)
                        phases_empty[:] = np.nan
                        t3phi.append(phases_empty)
                        t3phierr.append(phases_empty)
                        tels.append([beam_map[a] for a in oi_t3[i]['STA_INDEX']])
                        u1coord = np.nan
                        v1coord = np.nan
                        u2coord = np.nan
                        v2coord = np.nan
                        u3coord = np.nan
                        v3coord = np.nan
                        u_coords.append([u1coord,u2coord,u3coord])
                        v_coords.append([v1coord,v2coord,v3coord])
                        continue
                    u1coord = oi_t3[i]['U1COORD']
                    v1coord = oi_t3[i]['V1COORD']
                    u2coord = oi_t3[i]['U2COORD']
                    v2coord = oi_t3[i]['V2COORD']
                    u3coord = -u1coord - u2coord
                    v3coord = -v1coord - v2coord
                    ## drop baselines beyond given length:
                    if bl_drop=='y':
                        bl = max(np.sqrt(u1coord**2+v1coord**2),
                                    np.sqrt(u2coord**2+v2coord**2),
                                    np.sqrt(u3coord**2+v2coord**2))
                        if bl>250:
                            phases_empty = np.empty(oi_t3[i]['T3PHI'].shape)
                            phases_empty[:] = np.nan
                            t3phi.append(phases_empty)
                            t3phierr.append(phases_empty)
                            tels.append([beam_map[a] for a in oi_t3[i]['STA_INDEX']])
                            u1coord = np.nan
                            v1coord = np.nan
                            u2coord = np.nan
                            v2coord = np.nan
                            u3coord = np.nan
                            v3coord = np.nan
                            u_coords.append([u1coord,u2coord,u3coord])
                            v_coords.append([v1coord,v2coord,v3coord])
                            continue

                    t3 = oi_t3[i]['T3PHI']
                    t3err = oi_t3[i]['T3PHIERR']
                    t3flag = np.where(oi_t3[i].field('FLAG')==True)
                    t3[t3flag] = np.nan
                    t3err[t3flag] = np.nan
                    t3phi.append(t3)
                    t3phierr.append(t3err)
                    tels.append([beam_map[a] for a in oi_t3[i]['STA_INDEX']])
                    u_coords.append([u1coord,u2coord,u3coord])
                    v_coords.append([v1coord,v2coord,v3coord])

                for i in eachindex(oi_vis2):
                    ## use this to exclude a telescope
                    stations = [beam_map[a] for a in oi_vis2[i]['STA_INDEX']]
                    if exclude in stations:
                        phases_empty = np.empty(oi_vis2[i]['VIS2DATA'].shape)
                        phases_empty[:] = np.nan
                        vis2.append(phases_empty)
                        vis2err.append(phases_empty)
                        continue
                    if bl_drop=='y':
                        uc = oi_vis2[i]['UCOORD']
                        vc = oi_vis2[i]['VCOORD']
                        if np.sqrt(uc**2+vc**2)>250:
                            phases_empty = np.empty(oi_vis2[i]['VIS2DATA'].shape)
                            phases_empty[:] = np.nan
                            vis2.append(phases_empty)
                            vis2err.append(phases_empty)
                            continue
                    v2 = oi_vis2[i]['VIS2DATA']
                    v2err = oi_vis2[i]['VIS2ERR']
                    v2flag = np.where(oi_vis2[i].field('FLAG')==True)
                    v2[v2flag] = np.nan
                    v2err[v2flag] = np.nan
                    vis2.append(v2)
                    vis2err.append(v2err)

                for i in eachindex(oi_vis):
                    ## use this to exclude a telescope
                    stations = [beam_map[a] for a in oi_vis[i]['STA_INDEX']]
                    if exclude in stations:
                        phases_empty = np.empty(oi_vis[i]['VISPHI'].shape)
                        phases_empty[:] = np.nan
                        visphi.append(phases_empty)
                        visphierr.append(phases_empty)
                        visamp.append(phases_empty)
                        visamperr.append(phases_empty)
                        vistels.append([beam_map[a] for a in oi_vis[i]['STA_INDEX']])
                        ucoords.append(np.nan)
                        vcoords.append(np.nan)
                        continue
                    if bl_drop=='y':
                        uc = oi_vis[i]['UCOORD']
                        vc = oi_vis[i]['VCOORD']
                        if np.sqrt(uc**2+vc**2)>250:
                            phases_empty = np.empty(oi_vis[i]['VISPHI'].shape)
                            phases_empty[:] = np.nan
                            visphi.append(phases_empty)
                            visphierr.append(phases_empty)
                            visamp.append(phases_empty)
                            visamperr.append(phases_empty)
                            vistels.append([beam_map[a] for a in oi_vis[i]['STA_INDEX']])
                            ucoords.append(np.nan)
                            vcoords.append(np.nan)
                            continue
                    vis = oi_vis[i]['VISPHI']
                    viserr = oi_vis[i]['VISPHIERR']
                    vamp = oi_vis[i]['VISAMP']
                    vamperr = oi_vis[i]['VISAMPERR']
                    visflag = np.where(oi_vis[i].field('FLAG')==True)
                    vis[visflag] = np.nan
                    viserr[visflag] = np.nan
                    vamp[visflag] = np.nan
                    vamperr[visflag] = np.nan
                    visphi.append(vis)
                    visphierr.append(viserr)
                    visamp.append(vamp)
                    visamperr.append(vamperr)
                    vistels.append([beam_map[a] for a in oi_vis[i]['STA_INDEX']])
                    ucoords.append(oi_vis[i]['UCOORD'])
                    vcoords.append(oi_vis[i]['VCOORD'])
            hdu.close()

    t3phi = np.array(t3phi)
    t3phierr = np.array(t3phierr)
    u_coords = np.array(u_coords)
    v_coords = np.array(v_coords)
    eff_wave = np.array(eff_wave)
    time_obs = np.array(time_obs)
    az = np.array(az)
    tels = np.array(tels)

    vis2=np.array(vis2)
    vis2err=np.array(vis2err)
    visphi = np.array(visphi)
    visphierr = np.array(visphierr)
    visamp = np.array(visamp)
    visamperr = np.array(visamperr)
    ucoords = np.array(ucoords)
    vcoords = np.array(vcoords)
    vistels = np.array(vistels)

    if interact=='y':

        ################################################
        ## Interactive Session to filter data
        ################################################

        ## regroup data by measurements
        n_cp = int(len(t3phi)/20)
        n_vp = int(len(visphi)/15)
        n_v2 = int(len(vis2)/15)
        n_wl = t3phi.shape[-1]

        t3phi = t3phi.reshape((20,n_cp,n_wl))
        t3phierr = t3phierr.reshape((20,n_cp,n_wl))
        visphi = visphi.reshape((15,n_vp,n_wl))
        visphierr = visphierr.reshape((15,n_vp,n_wl))
        vis2 = vis2.reshape((15,n_v2,n_wl))
        vis2err = vis2err.reshape((15,n_v2,n_wl))

        # Functions for interactive session
        def on_click(event):
            bad_x = event.xdata
            bad_y = event.ydata
            diff = np.sqrt((x-bad_x)**2+(y-bad_y)**2)
            idx = np.nanargmin(diff)
            idx2 = np.where(m1==y[idx])
            m1[idx2] = np.nan
            y[idx] = np.nan

            ax.cla()
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            plt.draw()

        ## t3phi
        t3phi_new = []
        index = np.linspace(0,19,20)
        for m1,m2,ind in zip(t3phi,t3phierr,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            t3phi_new.append(m1)
        t3phi_new = np.array(t3phi_new)

        ## vis2
        vis2_new = []
        index = np.linspace(0,14,15)
        for m1,m2,ind in zip(vis2,vis2err,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            vis2_new.append(m1)
        vis2_new = np.array(vis2_new)

        ## visphi
        visphi_new = []
        index = np.linspace(0,14,15)
        for m1,m2,ind in zip(visphi,visphierr,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            visphi_new.append(m1)
        visphi_new = np.array(visphi_new)

        ## group data back to normal for fitting
        t3phi = t3phi_new.reshape((20*n_cp,n_wl))
        t3phierr = t3phierr.reshape((20*n_cp,n_wl))
        visphi = visphi_new.reshape((15*n_vp,n_wl))
        visphierr = visphierr.reshape((15*n_vp,n_wl))
        vis2 = vis2_new.reshape((15*n_v2,n_wl))
        vis2err = vis2err.reshape((15*n_v2,n_wl))

    ########################################################

    return t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,az

def read_vlti(dir,interact='n',exclude=''):
    t3phi=[]
    t3phierr=[]
    eff_wave = []
    time_obs = []
    tels = []
    u_coords = []
    v_coords = []
    vis2=[]
    vis2err=[]
    visphi=[]
    visphierr=[]
    visamp=[]
    visamperr=[]
    vistels=[]
    ucoords=[]
    vcoords=[]
    flux=[]
    fluxerr=[]

    for file in os.listdir(dir):
        if file.endswith("singlescivis.fits") or file.endswith("singlesciviscalibrated.fits"):
            filename = os.path.join(dir, file)
            oifile = fits.open(filename,quiet=True)
            oi_t3 = oifile['OI_T3',10].data
            oi_vis2 = oifile['OI_VIS2',10].data
            oi_vis = oifile['OI_VIS',10].data
            oi_flux = oifile['OI_FLUX',10].data

            nwave = oifile['OI_WAVELENGTH',10].data.field('EFF_WAVE').shape[0]
            cut = int(nwave*0.05)

            eff_wave.append(oifile['OI_WAVELENGTH',10].data.field('EFF_WAVE')[cut:-cut])
            time_obs.append(oifile[0].header['MJD-OBS'])

            for i in eachindex(oi_t3):
                t3 = oi_t3[i]['T3PHI']
                t3err = oi_t3[i]['T3PHIERR']
                idx = np.where(t3err==0.)
                t3[idx] = np.nan
                t3err[idx] = np.nan
                t3flag = np.where(oi_t3[i].field('FLAG')==True)
                t3[t3flag] = np.nan
                t3err[t3flag] = np.nan

                t3phi.append(t3[cut:-cut])
                t3phierr.append(t3err[cut:-cut])
                tels.append(oi_t3[i].field('STA_INDEX'))

                u1coord = oi_t3[i]['U1COORD']
                v1coord = oi_t3[i]['V1COORD']
                u2coord = oi_t3[i]['U2COORD']
                v2coord = oi_t3[i]['V2COORD']
                u3coord = -u1coord - u2coord
                v3coord = -v1coord - v2coord
                u_coords.append([u1coord,u2coord,u3coord])
                v_coords.append([v1coord,v2coord,v3coord])

            for i in eachindex(oi_vis2):
                v2 = oi_vis2[i]['VIS2DATA']
                v2err = oi_vis2[i]['VIS2ERR']
                idx = np.where(v2err==0.)
                v2[idx] = np.nan
                v2err[idx] = np.nan
                v2flag = np.where(oi_vis2[i].field('FLAG')==True)
                v2[v2flag] = np.nan
                v2err[v2flag] = np.nan

                vis2.append(v2[cut:-cut])
                vis2err.append(v2err[cut:-cut])
                vistels.append(oi_vis2[i].field('STA_INDEX'))
                ucoords.append(oi_vis2[i]['UCOORD'])
                vcoords.append(oi_vis2[i]['VCOORD'])

            for i in eachindex(oi_vis):
                vphi = oi_vis[i]['VISPHI']
                vphierr = oi_vis[i]['VISPHIERR']
                idx = np.where(vphierr==0.)
                vphi[idx] = np.nan
                vphierr[idx] = np.nan
                vphiflag = np.where(oi_vis[i].field('FLAG')==True)
                vphi[vphiflag] = np.nan
                vphierr[vphiflag] = np.nan

                visphi.append(vphi[cut:-cut])
                visphierr.append(vphierr[cut:-cut])

                vamp = oi_vis[i]['VISAMP']
                vamperr = oi_vis[i]['VISAMPERR']
                vamp[vphiflag] = np.nan
                vamperr[vphiflag] = np.nan

                visamp.append(vamp[cut:-cut])
                visamperr.append(vamperr[cut:-cut])

            for i in eachindex(oi_flux):
                fluxdata = oi_flux[i]['FLUX']
                fluxdataerr = oi_flux[i]['FLUXERR']

                flux.append(fluxdata[cut:-cut])
                fluxerr.append(fluxdataerr[cut:-cut])

            oifile.close()

    t3phi=np.array(t3phi)
    t3phierr=np.array(t3phierr)
    eff_wave=np.array(eff_wave)
    tels=np.array(tels)
    u_coords=np.array(u_coords)
    v_coords=np.array(v_coords)
    time_obs=np.array(time_obs)
    vis2=np.array(vis2)
    vis2err=np.array(vis2err)
    visphi=np.array(visphi)
    visphierr=np.array(visphierr)
    visamp=np.array(visamp)
    visamperr=np.array(visamperr)
    vistels=np.array(vistels)
    ucoords=np.array(ucoords)
    vcoords=np.array(vcoords)
    flux=np.array(flux)
    fluxerr=np.array(fluxerr)

    if interact=='y':

        ################################################
        ## Interactive Session to filter data
        ################################################

        ## regroup data by measurements
        n_cp = int(len(t3phi)/4)
        n_vp = int(len(visphi)/6)
        n_v2 = int(len(vis2)/6)
        n_wl = t3phi.shape[-1]

        t3phi = t3phi.reshape((4,n_cp,n_wl))
        t3phierr = t3phierr.reshape((4,n_cp,n_wl))
        visphi = visphi.reshape((6,n_vp,n_wl))
        visphierr = visphierr.reshape((6,n_vp,n_wl))
        vis2 = vis2.reshape((6,n_v2,n_wl))
        vis2err = vis2err.reshape((6,n_v2,n_wl))

        # Functions for interactive session
        def on_click(event):
            bad_x = event.xdata
            bad_y = event.ydata
            diff = np.sqrt((x-bad_x)**2+(y-bad_y)**2)
            idx = np.nanargmin(diff)
            idx2 = np.where(m1==y[idx])
            m1[idx2] = np.nan
            y[idx] = np.nan

            ax.cla()
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            plt.draw()

        ## t3phi
        t3phi_new = []
        index = np.linspace(0,3,4)
        for m1,m2,ind in zip(t3phi,t3phierr,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            t3phi_new.append(m1)
        t3phi_new = np.array(t3phi_new)

        ## vis2
        vis2_new = []
        index = np.linspace(0,5,6)
        for m1,m2,ind in zip(vis2,vis2err,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            vis2_new.append(m1)
        vis2_new = np.array(vis2_new)

        ## visphi
        visphi_new = []
        index = np.linspace(0,5,6)
        for m1,m2,ind in zip(visphi,visphierr,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            visphi_new.append(m1)
        visphi_new = np.array(visphi_new)

        ## group data back to normal for fitting
        t3phi = t3phi_new.reshape((4*n_cp,n_wl))
        t3phierr = t3phierr.reshape((4*n_cp,n_wl))
        visphi = visphi_new.reshape((6*n_vp,n_wl))
        visphierr = visphierr.reshape((6*n_vp,n_wl))
        vis2 = vis2_new.reshape((6*n_v2,n_wl))
        vis2err = vis2err.reshape((6*n_v2,n_wl))

    ########################################################
    return t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,flux,fluxerr

def read_chara_old(file,interact='n',exclude=''):

    beam_map = {1:'S1',2:'S2',3:'E1',4:'E2',5:'W1',6:'W2'}
    #beam_map = {0:'S1',1:'S2',2:'E1',3:'E2',4:'W1',5:'W2'}

    ## get information from fits file
    t3phi=[]
    t3phierr=[]
    eff_wave = []
    time_obs = []
    tels = []
    u_coords = []
    v_coords = []
    vis2=[]
    vis2err=[]
    visphi=[]
    visphierr=[]
    visamp=[]
    visamperr=[]
    vistels=[]
    ucoords=[]
    vcoords=[]

    print(file)
    hdu = fits.open(file)

    #start = 1
    #end = -1
    for table in hdu:

        #print(table.name)
        wl_i=1
        if table.name=='OI_T3':
            for i in eachindex(table.data):
                ## use this to exclude a telescope
                stations = [beam_map[a] for a in table.data[i]['STA_INDEX']]
                if exclude in stations:
                    phases_empty = np.empty(table.data[i]['T3PHI'].shape)
                    phases_empty[:] = np.nan
                    t3phi.append(phases_empty)
                    t3phierr.append(phases_empty)
                    tels.append([beam_map[a] for a in table.data[i]['STA_INDEX']])
                    u1coord = np.nan
                    v1coord = np.nan
                    u2coord = np.nan
                    v2coord = np.nan
                    u3coord = np.nan
                    v3coord = np.nan
                    u_coords.append([u1coord,u2coord,u3coord])
                    v_coords.append([v1coord,v2coord,v3coord])
                    continue
                t3 = table.data[i]['T3PHI']
                t3err = table.data[i]['T3PHIERR']
                t3flag = np.where(table.data[i].field('FLAG')==True)
                t3[t3flag] = np.nan
                t3err[t3flag] = np.nan
                t3phi.append(t3)
                t3phierr.append(t3err)
                tels.append([beam_map[a] for a in table.data[i]['STA_INDEX']])
                u1coord = table.data[i]['U1COORD']
                v1coord = table.data[i]['V1COORD']
                u2coord = table.data[i]['U2COORD']
                v2coord = table.data[i]['V2COORD']
                u3coord = -u1coord - u2coord
                v3coord = -v1coord - v2coord
                u_coords.append([u1coord,u2coord,u3coord])
                v_coords.append([v1coord,v2coord,v3coord])
                time_obs.append(table.data[i]['MJD'])
                eff_wave.append(hdu['OI_WAVELENGTH',wl_i].data['EFF_WAVE'])
            wl_i+=1

        if table.name=='OI_VIS2':
            for i in eachindex(table.data):
                ## use this to exclude a telescope
                stations = [beam_map[a] for a in table.data[i]['STA_INDEX']]
                if exclude in stations:
                    phases_empty = np.empty(table.data[i]['VIS2DATA'].shape)
                    phases_empty[:] = np.nan
                    vis2.append(phases_empty)
                    vis2err.append(phases_empty)
                    vistels.append([beam_map[a] for a in table.data[i]['STA_INDEX']])
                    ucoords.append(np.nan)
                    vcoords.append(np.nan)
                    continue
                v2 = table.data[i]['VIS2DATA']
                v2err = table.data[i]['VIS2ERR']
                v2flag = np.where(table.data[i].field('FLAG')==True)
                v2[v2flag] = np.nan
                v2err[v2flag] = np.nan
                vis2.append(v2)
                vis2err.append(v2err)
                vistels.append([beam_map[a] for a in table.data[i]['STA_INDEX']])
                ucoords.append(table.data[i]['UCOORD'])
                vcoords.append(table.data[i]['VCOORD'])

        if table.name=='OI_VIS':
            for i in eachindex(table.data):
                ## use this to exclude a telescope
                stations = [beam_map[a] for a in table.data[i]['STA_INDEX']]
                if exclude in stations:
                    phases_empty = np.empty(table.data[i]['VISPHI'].shape)
                    phases_empty[:] = np.nan
                    visphi.append(phases_empty)
                    visphierr.append(phases_empty)
                    visamp.append(phases_empty)
                    visamperr.append(phases_empty)
                    continue

                vis = table.data[i]['VISPHI']
                viserr = table.data[i]['VISPHIERR']
                vamp = table.data[i]['VISAMP']
                vamperr = table.data[i]['VISAMPERR']
                visflag = np.where(table.data[i].field('FLAG')==True)
                vis[visflag] = np.nan
                viserr[visflag] = np.nan
                vamp[visflag] = np.nan
                vamperr[visflag] = np.nan
                visphi.append(vis)
                visphierr.append(viserr)
                visamp.append(vamp)
                visamperr.append(vamperr)

    hdu.close()

    t3phi = np.array(t3phi)
    t3phierr = np.array(t3phierr)
    u_coords = np.array(u_coords)
    v_coords = np.array(v_coords)
    eff_wave = np.array(eff_wave)
    time_obs = np.array(time_obs)
    tels = np.array(tels)

    vis2=np.array(vis2)
    vis2err=np.array(vis2err)
    visphi = np.array(visphi)
    visphierr = np.array(visphierr)
    visamp = np.array(visamp)
    visamperr = np.array(visamperr)
    ucoords = np.array(ucoords)
    vcoords = np.array(vcoords)
    vistels = np.array(vistels)

    if interact=='y':

        ################################################
        ## Interactive Session to filter data
        ################################################

        ## regroup data by measurements
        n_cp = int(len(t3phi)/20)
        n_vp = int(len(visphi)/15)
        n_v2 = int(len(vis2)/15)
        n_wl = t3phi.shape[-1]

        t3phi = t3phi.reshape((20,n_cp,n_wl))
        t3phierr = t3phierr.reshape((20,n_cp,n_wl))
        visphi = visphi.reshape((15,n_vp,n_wl))
        visphierr = visphierr.reshape((15,n_vp,n_wl))
        vis2 = vis2.reshape((15,n_v2,n_wl))
        vis2err = vis2err.reshape((15,n_v2,n_wl))

        # Functions for interactive session
        def on_click(event):
            bad_x = event.xdata
            bad_y = event.ydata
            diff = np.sqrt((x-bad_x)**2+(y-bad_y)**2)
            idx = np.nanargmin(diff)
            idx2 = np.where(m1==y[idx])
            m1[idx2] = np.nan
            y[idx] = np.nan

            ax.cla()
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            plt.draw()

        ## t3phi
        t3phi_new = []
        index = np.linspace(0,19,20)
        for m1,m2,ind in zip(t3phi,t3phierr,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            t3phi_new.append(m1)
        t3phi_new = np.array(t3phi_new)

        ## vis2
        vis2_new = []
        index = np.linspace(0,14,15)
        for m1,m2,ind in zip(vis2,vis2err,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            vis2_new.append(m1)
        vis2_new = np.array(vis2_new)

        ## visphi
        visphi_new = []
        index = np.linspace(0,14,15)
        for m1,m2,ind in zip(visphi,visphierr,index):
            fig,ax = plt.subplots()
            y = np.ndarray.flatten(m1)
            yerr = np.ndarray.flatten(m2)
            x = np.linspace(1,len(np.ndarray.flatten(m1)),len(np.ndarray.flatten(m1)))
            ax.set_title('Triangle %s: click on points to remove'%int(ind))
            ax.errorbar(x,y,yerr=yerr,fmt='o',picker=True)
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            fig.canvas.mpl_disconnect(cid)
            visphi_new.append(m1)
        visphi_new = np.array(visphi_new)

        ## group data back to normal for fitting
        t3phi = t3phi_new.reshape((20*n_cp,n_wl))
        t3phierr = t3phierr.reshape((20*n_cp,n_wl))
        visphi = visphi_new.reshape((15*n_vp,n_wl))
        visphierr = visphierr.reshape((15*n_vp,n_wl))
        vis2 = vis2_new.reshape((15*n_v2,n_wl))
        vis2err = vis2err.reshape((15*n_v2,n_wl))

    ########################################################

    return t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs
