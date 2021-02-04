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
import os.path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
import matplotlib.cm as cm
from read_data import read_data,read_wds,read_orb6,get_types,read_idealWDS
from astrometry_model import astrometry_model,triple_model,lnlike,lnprior,lnpost,create_init
from orbit_plotting import orbit_model,triple_orbit_model
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord

def cart2pol(x,y):
    x=-x
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x) * 180 / np.pi
    if theta>0 and theta<90:
        theta_new = theta+270
    if theta>90 and theta<360:
        theta_new = theta-90
    if theta<0:
        theta_new = 270+theta
    if np.isnan(theta):
        theta_new=theta
    return(r,theta_new)
    
def plot_data(xpos_wds,ypos_wds,types,xpos,ypos,idx,err_maj,err_min,err_pa,t_wds):
    x=[]
    y=[]
    t=[]
    emx=[]
    emn=[]
    epa=[]
    twds=[]
    wdt = make_plotable(xpos_wds,ypos_wds,types,err_maj,err_min,err_pa,t_wds)
    
    for i in range(0,len(wdt)):
        plt.plot(wdt[i][0],wdt[i][1],'o',label=wdt[i][2])
            
    for i in wdt:
        for j in i[0]:
            x.append(j)
            t.append(i[2])
        for k in i[1]:
            y.append(k)
        for l in i[3]:
            emx.append(l)
        for m in i[4]:
            emn.append(m)
        for n in i[5]:
            epa.append(n)
        for o in i[6]:
            twds.append(o)

    plt.plot(xpos_wds[0],ypos_wds[0],'*')
    plt.plot(xpos[idx],ypos[idx],'*')
    plt.plot(xpos,ypos,'+',label='ARMADA')
    plt.plot(0,0,'*')
    plt.gca().invert_xaxis()
    plt.title('All Data')
    plt.xlabel('dra (mas)')
    plt.ylabel('ddec (mas)')
    plt.legend()
    plt.show()

    return x, y, t, emx, emn, epa, twds
    
def make_plotable(x, y, ty, e_maj, e_min, e_pa, t_w):
    tList = []
    data_tot = []
    repeats = 0

    for i in ty:
        if i in tList:
            ++repeats # 'if i not in tList' did not work so need soemthing here/
        else:
            tList.append(i)
    for i in tList:
        xType = []
        yType = []
        majType = []
        minType = []
        tType = []
        paType = []
        data = []
        for j in range(0,len(x)):
            if ty[j] == i:
                xType.append(x[j])
                yType.append(y[j])
                majType.append(e_maj[j])
                minType.append(e_min[j])
                paType.append(e_pa[j])
                tType.append(t_w[j])
        data.append(xType)
        data.append(yType)
        data.append(i)
        data.append(majType)
        data.append(minType)
        data.append(paType)
        data.append(tType)
        data_tot.append(data)
        
    # Returns an array of [[x, y, type, error_major, error_minor, error_pa, t],...]
    return data_tot

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
    path = '/Users/adam.scovera/Documents/Astro/BEPResearch_Data/ARMADA_orbits'
    path_etalon = '/Users/adam.scovera/Documents/Astro/BEPResearch_Data/etalon_factors_fit.txt'
    path_wds = '/Users/adam.scovera/Documents/AStro/BEPResearch_Data/wds_targets'
    path_orb6 = '/Users/adam.scovera/Documents/Astro/BEPResearch_Data/orb6orbits.sql.txt'

###########################################
## Specify Target
###########################################
target_hd = input('Target HD #: ')
#target = input('Target HIP #: ')
#target_wds = input('Target WDS #: ')

#emethod = input('bootstrap errors? (y/n) ')
emethod = 'n'
#mirc_scale = input('Scale OLD MIRCX data? (y/n) ')
mirc_scale = 'n'

query = Simbad.query_objectids('HD %s'%target_hd)
for item in query:
    if 'HIP' in item[0]:
        target = item[0].split()[1]
        hipNum = target;
        print('HIP %s'%target)
    if 'WDS' in item[0]:
        target_wds = item[0][5:15]
        print('WDS %s'%target_wds)


###########################################
## Read in ARMADA data
###########################################
if emethod == 'y':
    print('reading bootstrap errors')
    file=open('%s/HD_%s_bootstrap.txt'%(path,target_hd))
else:
    print('reading chi2 errors')
    file=open('%s/HD_%s_chi2err.txt'%(path,target_hd))
weight=1

t,p,theta,error_maj,error_min,error_pa,error_deg = read_data(file,weight)
file.close()

### correct PAs based on precession (only for WDS):
coord = SkyCoord.from_name("HD %s"%target_hd,parse=True)
ra = coord.ra.value*np.pi/180
dec = coord.dec.value*np.pi/180
#theta -= (0.00557*np.sin(ra)/np.cos(dec)*((t-51544.5)/365.25))/180*np.pi

## Include only NEW data after upgrade:
#idx = np.where(t>58362)
#t = t[idx]
#p = p[idx]
#theta = theta[idx]
#error_maj = error_maj[idx]
#error_min = error_min[idx]
#error_pa = error_pa[idx]
#error_deg = error_deg[idx]


###########################################
## Apply etalon correction
###########################################
file=open(path_etalon)
mjd_etalon=[]
f_etalon=[]
for line in file.readlines():
    if line.startswith('#'):
        continue
    mjd_etalon.append(float(line.split()[0]))
    f_etalon.append(float(line.split()[1]))
file.close()
mjd_etalon=np.array(mjd_etalon)
f_etalon=np.array(f_etalon)

etalon_factor=[]
for i in t:
    idx = np.where(abs(i-mjd_etalon)==min(abs(i-mjd_etalon)))
    if min(abs(i-mjd_etalon))>0.5:
        print('Closest factor for %s is %s days away'%(i,min(abs(i-mjd_etalon))))
    f = f_etalon[idx][0]
    etalon_factor.append(f)
etalon_factor=np.array(etalon_factor)

print('   date      etalon factor')
for i,j in zip(t,etalon_factor):
    print(i,j)

## apply etalon correction
etalon = input('Apply etalon correction? (y/n) ')

## FIXME: make it easier to choose vlti data
#vlti = input('Add indices for vlti (y/n)? ')
vlti = 'n'
if vlti=='y':
    vlti_idx = input('enter indices (e.g. 1 2 3): ').split(' ')
    vlti_idx = np.array([int(i) for i in vlti_idx])
else:
    vlti_idx = np.array([])

if etalon=='y':
    print('Applying etalon correction')
    if len(vlti_idx)>0:
        print('Setting VLTI correction factors to 1.0')
        etalon_factor[vlti_idx] = 1.0
    p = p/etalon_factor
else:
    print('No etalon correction applied')
xpos=p*np.sin(theta)
ypos=p*np.cos(theta)


###########################################
## Read in WDS data - and plot to check
###########################################
wds_data_tot = []
xWDS = []
yWDS = []
WDStype = []
errMajWDS = []
errMinWDS = []
errPa = []
tWDS = []
idealExists = False

try:
    idealFile = "%s/HD%s_chi2err/HD_%s_wds_ideal.txt"%(path,target_hd,target_hd)
    if os.path.isfile(idealFile):
        idealExists = True
        ans = input('Would you like to use the ideal data available? y/n? ')
        if (ans == 'y' or ans == 'Y'):
            file = open(os.path.expanduser("%s/HD%s_chi2err/HD_%s_wds_ideal.txt"%(path,target_hd,target_hd)))
            xpos_wds,ypos_wds,t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds,types = read_idealWDS(file,10)
        else:
            file1=open(os.path.expanduser("%s/wds%s.txt"%(path_wds,target_wds)))
            target_types = get_types(file1)
            file2=open(os.path.expanduser("%s/wds%s.txt"%(path_wds,target_wds)))
            weight = 10
            t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds,types = read_wds(file2,weight,target_types)
            ## correct WDS for PA
            theta_wds -= (0.00557*np.sin(ra)/np.cos(dec)*((t_wds-51544.5)/365.25))/180*np.pi

            xpos_wds=p_wds*np.sin(theta_wds)
            ypos_wds=p_wds*np.cos(theta_wds)
            
    else:
        file1=open(os.path.expanduser("%s/wds%s.txt"%(path_wds,target_wds)))
        target_types = get_types(file1)
        file2=open(os.path.expanduser("%s/wds%s.txt"%(path_wds,target_wds)))
        weight = 10
        t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds,types = read_wds(file2,weight,target_types)
        ## correct WDS for PA
        theta_wds -= (0.00557*np.sin(ra)/np.cos(dec)*((t_wds-51544.5)/365.25))/180*np.pi

        xpos_wds=p_wds*np.sin(theta_wds)
        ypos_wds=p_wds*np.cos(theta_wds)
    
    print('Number of WDS data points = %s'%len(p_wds))
    idx = np.argmin(t)
    
    xWDS, yWDS, WDStype, errMajWDS, errMinWDS, errPa, tWDS = plot_data(xpos_wds,ypos_wds,types,xpos,ypos,idx,error_maj_wds,error_min_wds,error_pa_wds,t_wds)
    
    if idealExists:
        flip = 'n'
    else:
        flip = input('Flip WDS data? (y/n): ')
    if flip=='y':
        xpos_wds=-p_wds*np.sin(theta_wds)
        ypos_wds=-p_wds*np.cos(theta_wds)
        xWDS, yWDS, WDStype, errMajWDS, errMinWDS, errPa, tWDS = plot_data(xpos_wds,ypos_wds,types,xpos,ypos,idx,error_maj_wds,error_min_wds,error_pa_wds,t_wds)
        
        better = input('Flip data back to original? (y/n): ')
        if better=='y':
            xpos_wds=p_wds*np.sin(theta_wds)
            ypos_wds=p_wds*np.cos(theta_wds)
            xWDS, yWDS, WDStype, errMajWDS, errMinWDS, errPa, tWDS = plot_data(xpos_wds,ypos_wds,types,xpos,ypos,idx,error_maj_wds,error_min_wds,error_pa_wds,t_wds)
   
    wds_data_tot = make_plotable(xWDS,yWDS,WDStype,errMajWDS,errMinWDS,errPa,tWDS)
    
except:
    t_wds = np.array([np.nan])
    p_wds = np.array([np.nan])
    theta_wds = np.array([np.nan])
    error_maj_wds = np.array([np.nan])
    error_min_wds = np.array([np.nan])
    error_pa_wds = np.array([np.nan])
    error_deg_wds = np.array([np.nan])
    xpos_wds=p_wds*np.sin(theta_wds)
    ypos_wds=p_wds*np.cos(theta_wds)
    print('NO WDS NUMBER')

###########################################
## Get an estimate of the orbital parameters
###########################################
try:
    a,P,e,inc,omega,bigomega,T = read_orb6(target,path_orb6)
except:
    print('No elements found in ORB6. Will need to enter your own.')
    
self_params = input('Input own params? (y/n)')
if self_params=='y':
    a = float(input('a (mas): '))
    P = float(input('P (year): '))*365.25
    e = float(input('ecc : '))
    inc = float(input('inc (deg): '))*np.pi/180
    omega = float(input('omega (deg): '))*np.pi/180
    bigomega = float(input('bigomega (deg): '))*np.pi/180
    T = float(input('T (mjd): '))

###########################################
## Combined WDS+ARMADA for fitting
###########################################
xpos_all = np.concatenate([xpos,xpos_wds])
ypos_all = np.concatenate([ypos,ypos_wds])
t_all = np.concatenate([t,t_wds])
error_maj_all = np.concatenate([error_maj,error_maj_wds])
error_min_all = np.concatenate([error_min,error_min_wds])
error_pa_all = np.concatenate([error_pa,error_pa_wds])
error_deg_all = np.concatenate([error_deg,error_deg_wds])

##########################################
## Function for fitting/plotting data
#########################################
def ls_fit(params,xp,yp,tp,emaj,emin,epa,wds):
    #do fit, minimizer uses LM for least square fitting of model to data
    minner = Minimizer(astrometry_model, params, fcn_args=(xp,yp,tp,
                                                       emaj,emin,epa),
                            nan_policy='omit')
    result = minner.minimize()
    # write error report
    print(report_fit(result))

    ## plot fit
    a_start = result.params['a']
    P_start = result.params['P']
    e_start = result.params['e']
    inc_start = result.params['inc']
    w_start = result.params['w']
    bigw_start = result.params['bigw']
    T_start = result.params['T']

    ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                            w_start,bigw_start,P_start,
                                            T_start,t_all)
    fig,ax=plt.subplots()
    
    for i in range(0,len(wds)):
        ax.plot(wds[i][0], wds[i][1], 'o', label=wds[i][2])
    # ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('HD%s Outer Orbit'%target_hd)
    plt.legend()
    plt.show()

    return result

###########################################
## Do a least-squares fit
###########################################
params = Parameters()
params.add('w',   value= omega, min=0, max=2*np.pi)
params.add('bigw', value= bigomega, min=0, max=2*np.pi)
params.add('inc', value= inc, min=0, max=2*np.pi)
params.add('e', value= e, min=0, max=0.99)
params.add('a', value= a, min=0)
params.add('P', value= P, min=0)
params.add('T', value= T, min=0)
if mirc_scale == 'y':
    params.add('mirc_scale', value= 1.0)
else:
    params.add('mirc_scale', value= 1.0, vary=False)

result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all,wds_data_tot)

#############################################
## Filter through bad WDS points
#############################################
def on_click_remove(event):
    bad_x = event.xdata
    bad_y = event.ydata
    diff = np.sqrt((xpos_all-bad_x)**2+(ypos_all-bad_y)**2)
    diffA = np.sqrt((xpos-bad_x)**2+(ypos-bad_y)**2)
    diffW = np.sqrt((xWDS-bad_x)**2+(yWDS-bad_y)**2)
    idx = np.nanargmin(diff)
    idxA = np.nanargmin(diffA)
    idxW = np.nanargmin(diffW)
    xpos_all[idx] = np.nan
    ypos_all[idx] = np.nan
    if diffA[idxA] < diffW[idxW]:
        xpos[idxA] = np.nan
        ypos[idxA] = np.nan
    else:
        xWDS[idxW] = np.nan
        yWDS[idxW] = np.nan
        WDStype[idxW] = np.nan
        errMajWDS[idxW] = np.nan
        errMinWDS[idxW] = np.nan
        errPa[idxW] = np.nan
        tWDS[idxW] = np.nan
        theta_wds[idxW] = np.nan
        p_wds[idxW]=np.nan
        #error_deg_wds.pop(idxW)
        wds_data_tot = make_plotable(xWDS, yWDS, WDStype,errMajWDS,errMinWDS,errPa,tWDS)

    ax.cla()
    for i in range(0,len(wds_data_tot)):
        ax.plot(wds_data_tot[i][0], wds_data_tot[i][1], 'o', label=wds_data_tot[i][2])
    # ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to REMOVE points')
    plt.legend()
    plt.draw()

def on_click_flip(event):
    bad_x = event.xdata
    bad_y = event.ydata
    diff = np.sqrt((xpos_all-bad_x)**2+(ypos_all-bad_y)**2)
    diffA = np.sqrt((xpos-bad_x)**2+(ypos-bad_y)**2)
    diffW = np.sqrt((xWDS-bad_x)**2+(yWDS-bad_y)**2)
    idx = np.nanargmin(diff)
    idxA = np.nanargmin(diffA)
    idxW = np.nanargmin(diffW)
    xpos_all[idx] = -xpos_all[idx]
    ypos_all[idx] = -ypos_all[idx]
    if diffA[idxA] < diffW[idxW]:
        xpos[idxA] = -xpos[idxA]
        ypos[idxA] = -ypos[idxA]
    else:
        xWDS[idxW] = -xWDS[idxW]
        yWDS[idxW] = -yWDS[idxW]
        wds_data_tot = make_plotable(xWDS, yWDS, WDStype,errMajWDS,errMinWDS,errPa,tWDS)

    ax.cla()
    for i in range(0,len(wds_data_tot)):
        ax.plot(wds_data_tot[i][0], wds_data_tot[i][1], 'o', label=wds_data_tot[i][2])
    # ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to FLIP points')
    plt.legend()
    plt.draw()

filter_wds = input('Remove/flip any WDS data? (y/n)')
while filter_wds == 'y':
    a_start = result.params['a']
    P_start = result.params['P']
    e_start = result.params['e']
    inc_start = result.params['inc']
    w_start = result.params['w']
    bigw_start = result.params['bigw']
    T_start = result.params['T']
    ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                        w_start,bigw_start,P_start,
                                        T_start,t_all)
    fig,ax=plt.subplots()
    for i in range(0,len(wds_data_tot)):
        ax.plot(wds_data_tot[i][0], wds_data_tot[i][1], 'o', label=wds_data_tot[i][2])
    #ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to FLIP points')
    plt.legend()
    cid = fig.canvas.mpl_connect('button_press_event', on_click_flip)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close()
    
    wds_data_tot = make_plotable(xWDS, yWDS, WDStype,errMajWDS,errMinWDS,errPa,tWDS)
    
    fig,ax=plt.subplots()
    for i in range(0,len(wds_data_tot)):
        ax.plot(wds_data_tot[i][0], wds_data_tot[i][1], 'o', label=wds_data_tot[i][2])
    #ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
    ax.plot(xpos,ypos,'o', label='ARMADA')
    ax.plot(0,0,'*')
    ax.plot(ra, dec, '--',color='g')
    #plot lines from data to best fit orbit
    i=0
    while i<len(decpoints):
        x=[xpos_all[i],rapoints[i]]
        y=[ypos_all[i],decpoints[i]]
        ax.plot(x,y,color="black")
        i+=1
    ax.set_xlabel('milli-arcsec')
    ax.set_ylabel('milli-arcsec')
    ax.invert_xaxis()
    ax.axis('equal')
    ax.set_title('Click to REMOVE points')
    plt.legend()
    cid = fig.canvas.mpl_connect('button_press_event', on_click_remove)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close()
    
    wds_data_tot = make_plotable(xWDS, yWDS, WDStype,errMajWDS,errMinWDS,errPa,tWDS)

    params = Parameters()
    params.add('w',   value= omega, min=0, max=2*np.pi)
    params.add('bigw', value= bigomega, min=0, max=2*np.pi)
    params.add('inc', value= inc, min=0, max=2*np.pi)
    params.add('e', value= e, min=0, max=0.99)
    params.add('a', value= a, min=0)
    params.add('P', value= P, min=0)
    params.add('T', value= T, min=0)
    if mirc_scale == 'y':
        params.add('mirc_scale', value= 1.0)
    else:
        params.add('mirc_scale', value= 1.0, vary=False)

    result = ls_fit(params,xpos_all,ypos_all,t_all,error_maj_all,error_min_all,error_pa_all,wds_data_tot)
    filter_wds = input('Remove more data? (y/n)')

##########################################
## Save Plots
##########################################
chi2Types = []

resids_armada = astrometry_model(result.params,xpos,ypos,t,error_maj,
                            error_min,error_pa)
ndata_armada = 2*sum(~np.isnan(xpos))
chi2_armada = np.nansum(resids_armada**2)/(ndata_armada-7) # 7 ~ len(result.params)
print('-'*10)
print('chi2 armada = %s'%chi2_armada)
print('-'*10)

for i in wds_data_tot:
    resid = astrometry_model(result.params,i[0],i[1],i[6],i[3],i[4],i[5])
    ndata = 2*sum(~np.isnan(i[0]))
    chi2 = np.nansum(resid**2)/(ndata-len(result.params))
    chi2Types.append([i[2], chi2])
    print('Collection Type: ', i[2], '-'*5)
    print('chi2 = %s'%chi2)
    print('-'*10)


if emethod == 'y':
    directory='%s/HD%s_bootstrap/'%(path,target_hd)
else:
    directory='%s/HD%s_chi2err/'%(path,target_hd)
if not os.path.exists(directory):
    os.makedirs(directory)

### plot fit
#if len(vlti_idx)>0:
#    xpos[vlti_idx]*=result.params['pscale']
#    ypos[vlti_idx]*=result.params['pscale']
#    xpos_all[vlti_idx]*=result.params['pscale']
#    ypos_all[vlti_idx]*=result.params['pscale']

scale=1
if chi2_armada<1.0:
    scale=1/np.sqrt(chi2_armada)
a_start = result.params['a']
P_start = result.params['P']
e_start = result.params['e']
inc_start = result.params['inc']
w_start = result.params['w']
bigw_start = result.params['bigw']
T_start = result.params['T']
ra,dec,rapoints,decpoints = orbit_model(a_start,e_start,inc_start,
                                        w_start,bigw_start,P_start,
                                        T_start,t_all)
fig,ax=plt.subplots()
for i in range(0,len(wds_data_tot)):
    ax.plot(wds_data_tot[i][0], wds_data_tot[i][1], 'o', label=wds_data_tot[i][2])
#ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS')
ax.plot(xpos,ypos,'o', label='ARMADA')
ax.plot(0,0,'*')
ax.plot(ra, dec, '--',color='g')
#plot lines from data to best fit orbit
i=0
while i<len(decpoints):
    x=[xpos_all[i],rapoints[i]]
    y=[ypos_all[i],decpoints[i]]
    ax.plot(x,y,color="black")
    i+=1
ax.set_xlabel('milli-arcsec')
ax.set_ylabel('milli-arcsec')
ax.invert_xaxis()
ax.axis('equal')
ax.set_title('HD%s Outer Orbit'%target_hd)
plt.legend()
plt.savefig('%s/HD%s_outer_leastsquares.pdf'%(directory,target_hd))
plt.close()

## plot resids for ARMADA
fig,ax=plt.subplots()
xresid = xpos - rapoints[:len(xpos)]
yresid = ypos - decpoints[:len(ypos)]

#need to measure error ellipse angle east of north
for ras, decs, w, h, angle in zip(xresid,yresid,error_maj/scale,error_min/scale,error_deg):
    ellipse = Ellipse(xy=(ras, decs), width=2*w, height=2*h, 
                      angle=90-angle, facecolor='none', edgecolor='black')
    ax.add_patch(ellipse)

ax.plot(xresid, yresid, 'o')
ax.plot(0,0,'*')
ax.set_xlabel('milli-arcsec')
ax.set_ylabel('milli-arcsec')
ax.invert_xaxis()
ax.axis('equal')
ax.set_title('HD%s Resids'%target_hd)
plt.savefig('%s/HD%s_resid_leastsquares.pdf'%(directory,target_hd))
plt.close()

## residuals
resids = np.sqrt(xresid**2 + yresid**2)
resids_median = np.around(np.median(resids)*1000,2)
print('-'*10)
print('Mean residual = %s micro-as'%resids_median)
print('-'*10)

## Save txt file with best orbit
f = open("%s/%s_orbit_ls.txt"%(directory,target_hd),"w+")
f.write("# P(d) a(mas) e i(deg) w(deg) W(deg) T(mjd) mean_resid(mu-as)\r\n")
f.write("%s %s %s %s %s %s %s %s"%(P_start.value,a_start.value,e_start.value,
                                   inc_start.value*180/np.pi,w_start.value*180/np.pi,
                                   bigw_start.value*180/np.pi,T_start.value,
                                  resids_median))
f.close()

## Print Values Needed for Spreadsheet
print('\nFor Spreadsheet')
print('HIP:', hipNum)
print('Resid (mas):', resids_median)
print('P(yr):', P_start.value/365.25)
print('a(mas):', a_start.value)
print('e:', e_start.value)
print('i(deg):', inc_start.value*180/np.pi)
print('w(deg):', w_start.value*180/np.pi)
print('bigw(deg):', bigw_start.value*180/np.pi)
print('T (mjd):', T_start.value, '\n')

## Save txt file with wds orbit
p_wds_new=[]
theta_wds_new = []
for i,j in zip(xpos_all[len(xpos):],ypos_all[len(xpos):]):
    pnew,tnew = cart2pol(i,j)
    p_wds_new.append(pnew)
    theta_wds_new.append(tnew)
p_wds_new = np.array(p_wds_new)
theta_wds_new = np.array(theta_wds_new)
f = open("%s/HD_%s_wds.txt"%(directory,target_hd),"w+")
f.write("# date mjd sep pa err_maj err_min err_pa\r\n")
for i,j,k,l,m,n in zip(t_wds,p_wds_new,theta_wds_new,error_maj_wds,error_min_wds,error_deg_wds):
    if (i != np.nan):
        f.write("-- %s %s %s %s %s %s\r\n"%(i,j,k,l,m,n))
for i in chi2Types:
    f.write("Data Collection Type: %s, Chi2: %s\r\n"%(i[0], i[1]))
f.write('#')
f.close()

overwrite = 's'

if idealExists:
    overwrite = input('Would you like to overwrite the existing ideal file? y/n? ')
else:
    overwrite = 'y'


## Save txt file with ideal wds elements
if (overwrite == 'y') or (overwrite == 'Y'):
    ## Save txt file with adjusted orbit
    f = open("%s/HD_%s_wds_ideal.txt"%(directory,target_hd),"w+")
    f.write("# x y Type MajorError MinorError PaError t p Theta\r\n")
    for i,j,k,l,m,n,o,p,q in zip(xWDS,yWDS,WDStype,errMajWDS,errMinWDS,errPa,tWDS,p_wds,theta_wds):
        f.write("-- %s %s %s %s %s %s %s %s %s\r\n"%(i,j,k,l,m,n,o,p,q))

search_triple = input('Search for triple? (y/n): ')
if search_triple=='n':
    exit()

##########################################
## Grid Search for Additional Companions
##########################################
ps = float(input('period search start (days): '))
pe = float(input('period search end (days): '))
ss = float(input('semi search start (mas): '))
se = float(input('semi search end (mas): '))
P2 = np.linspace(ps,pe,1000)
w2 = w_start
#bigw2 = bigw_start
#inc2 = inc_start
e2 = 0.01
a2 = resids_median/1000
#T2 = 55075

print('Grid Searching over period')
params_inner=[]
params_outer=[]
chi2 = []
for period in tqdm(P2):

    params_inner_n=[]
    params_outer_n=[]
    chi2_n = []

    for i in np.arange(10):
        ## randomize orbital elements
        bigw2 = np.random.uniform(0,2*np.pi)
        inc2 = np.random.uniform(0,np.pi)
        T2 = np.random.uniform(58000,58700)

        params = Parameters()
        params.add('w',   value= w_start, min=0, max=2*np.pi)
        params.add('bigw', value= bigw_start, min=0, max=2*np.pi)
        params.add('inc', value= inc_start, min=0, max=2*np.pi)
        params.add('e', value= e_start, min=0, max=0.99)
        params.add('a', value= a_start, min=0)
        params.add('P', value= P_start, min=0)
        params.add('T', value= T_start, min=0)
        params.add('w2',   value= 0, vary=False)
        params.add('bigw2', value= bigw2, min=0, max=2*np.pi)
        params.add('inc2', value= inc2, min=0, max=2*np.pi)
        params.add('e2', value= 0, vary=False)
        params.add('a2', value= a2, min=0)
        params.add('P2', value= period, vary=False)
        params.add('T2', value= T2, min=0)
        if mirc_scale == 'y':
            params.add('mirc_scale', value= 1.0)
        else:
            params.add('mirc_scale', value= 1.0, vary=False)

        #params.add('pscale', value=1)

        #do fit, minimizer uses LM for least square fitting of model to data
        minner = Minimizer(triple_model, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                           error_maj_all,error_min_all,
                                                           error_pa_all),
                          nan_policy='omit')
        result = minner.leastsq(xtol=1e-5,ftol=1e-5)
        params_inner_n.append([period,result.params['a2'],result.params['e2'],result.params['w2']
                            ,result.params['bigw2'],result.params['inc2'],result.params['T2']])
        params_outer_n.append([result.params['P'],result.params['a'],result.params['e'],result.params['w']
                            ,result.params['bigw'],result.params['inc'],result.params['T']])
        chi2_n.append(result.redchi)

    params_inner_n=np.array(params_inner_n)
    params_outer_n=np.array(params_outer_n)
    chi2_n = np.array(chi2_n)

    idx = np.argmin(chi2_n)
    chi2.append(chi2_n[idx])
    params_inner.append(params_inner_n[idx])
    params_outer.append(params_outer_n[idx])
    

params_inner=np.array(params_inner)
params_outer=np.array(params_outer)
chi2 = np.array(chi2)

idx = np.argmin(chi2)
period_best = params_inner[:,0][idx]

plt.plot(params_inner[:,0],1/chi2,'o-')
plt.xlabel('Period (d)')
plt.ylabel('1/chi2')
plt.title('Best Period = %s'%period_best)
plt.savefig('%s/HD%s_chi2_period.pdf'%(directory,target_hd))
plt.close()

print('Best inner period = %s'%period_best)

## Do a fit at best period
params = Parameters()
params.add('w',   value= w_start, min=0, max=2*np.pi)
params.add('bigw', value= bigw_start, min=0, max=2*np.pi)
params.add('inc', value= inc_start, min=0, max=np.pi)
params.add('e', value= e_start, min=0, max=0.99)
params.add('a', value= a_start, min=0)
params.add('P', value= P_start, min=0)
params.add('T', value= T_start, min=0)
params.add('w2',   value= 0, vary=False)#w2, min=0, max=2*np.pi)
params.add('bigw2', value= params_inner[:,4][idx], min=0, max=2*np.pi)
params.add('inc2', value= params_inner[:,5][idx], min=0, max=np.pi)
params.add('e2', value= 0, vary=False)#0.1, min=0,max=0.99)
params.add('a2', value= a2, min=0)
params.add('P2', value= period_best, min=0)
params.add('T2', value= params_inner[:,6][idx], min=0)
if mirc_scale == 'y':
    params.add('mirc_scale', value= 1.0)
else:
    params.add('mirc_scale', value= 1.0, vary=False)

#params.add('pscale', value=1)

#do fit, minimizer uses LM for least square fitting of model to data
minner = Minimizer(triple_model, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                   error_maj_all,error_min_all,
                                                   error_pa_all),
                  nan_policy='omit')
result = minner.minimize()
best_inner = [result.params['P2'],result.params['a2'],result.params['e2'],result.params['w2']
                    ,result.params['bigw2'],result.params['inc2'],result.params['T2']]
best_outer = [result.params['P'],result.params['a'],result.params['e'],result.params['w']
                    ,result.params['bigw'],result.params['inc'],result.params['T']]
try:
    report_fit(result)
except:
    print('-'*10)
    print('Triple fit FAILED!!!!')
    print('-'*10)

############################
## Grid Search on a/i 
############################
a_grid = np.linspace(ss,se,50)
i_grid = np.linspace(0,180,50)

params_inner=[]
params_outer=[]
chi2 = []
for semi in tqdm(a_grid):
    for angle in i_grid:
        params = Parameters()
        params.add('w',   value= best_outer[3], vary=False)#min=0, max=2*np.pi)
        params.add('bigw', value= best_outer[4], vary=False)#min=0, max=2*np.pi)
        params.add('inc', value= best_outer[5], vary=False)#min=0, max=2*np.pi)
        params.add('e', value= best_outer[2], vary=False)#min=0, max=0.99)
        params.add('a', value= best_outer[1], vary=False)#min=0)
        params.add('P', value= best_outer[0], vary=False)#min=0)
        params.add('T', value= best_outer[6], vary=False)#min=0)
        params.add('w2',   value= 0, vary=False)
        params.add('bigw2', value= best_inner[4], min=0, max=2*np.pi)
        params.add('inc2', value= angle*np.pi/180, vary=False)#min=0, max=2*np.pi)
        params.add('e2', value= 0, vary=False)
        params.add('a2', value= semi, vary=False)#a2, min=0)
        params.add('P2', value= best_inner[0], min=0)
        params.add('T2', value= best_inner[6], min=0)
        if mirc_scale == 'y':
            params.add('mirc_scale', value= 1.0)
        else:
            params.add('mirc_scale', value= 1.0, vary=False)

        #do fit, minimizer uses LM for least square fitting of model to data
        minner = Minimizer(triple_model, params, fcn_args=(xpos_all,ypos_all,t_all,
                                                           error_maj_all,error_min_all,
                                                           error_pa_all),
                                            nan_policy='omit')
        result = minner.leastsq(xtol=1e-5,ftol=1e-5)
        params_inner.append([result.params['P2'],semi,result.params['e2'],result.params['w2']
                            ,result.params['bigw2'],angle,result.params['T2']])
        params_outer.append([result.params['P'],result.params['a'],result.params['e'],result.params['w']
                            ,result.params['bigw'],result.params['inc'],result.params['T']])
        chi2.append(result.redchi)
params_inner = np.array(params_inner)
params_outer = np.array(params_outer)
chi2 = np.array(chi2)

a_inner = params_inner[:,1]
i_inner = params_inner[:,5]
plt.scatter(a_inner,i_inner,c=1/chi2,cmap=cm.inferno)
plt.colorbar(label='1 / $\chi^2$')
plt.xlabel('semi-major (mas)')
plt.ylabel('inclination (deg)')
plt.savefig('%s/HD%s_semi_inc_grid.pdf'%(directory,target_hd))
plt.close()
