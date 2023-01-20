######################################################################
## Tyler Gardner
##
## Take output of binary_fitting_armada.py and fit error ellipse
## 
## Outputs pdf and txt files with plots, results
##
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
eachindex = lambda lst: range(len(lst))
import os
import matplotlib.cm as cm
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from tqdm import tqdm
from matplotlib.patches import Ellipse
from ellipse_fitting import ellipse_hull_fit
from read_oifits import read_chara,read_vlti,read_chara_old

######################################################################
## DEFINE FITTING FUNCTIONS
######################################################################

chi_sq = np.load(input('chisq npy file: '))
ra_results = np.load(input('ra npy file: '))
dec_results = np.load(input('dec npy file: '))
file = input('OIFITS file: ')
date = input('date for saved file: ')
target_id = input('Target ID: ')

t3phi,t3phierr,vis2,vis2err,visphi,visphierr,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave,tels,vistels,time_obs,flux,fluxerr = read_vlti(file,'n')

## function which converts cartesian to polar coords
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
    return(r,theta_new)

## isolate region where delta_chisq < 2.296
params = [1,1,1,1,1,1]
#chi = combined_minimizer(params,t3phi,t3phierr,visphi_new,visphierr,vis2,vis2err,visamp,visamperr,u_coords,v_coords,ucoords,vcoords,eff_wave[0])
#chi2_best = np.nansum(chi**2)/(len(np.ndarray.flatten(t3phi))-len(params))
chi_sq = chi_sq/(len(np.ndarray.flatten(t3phi))-len(params))
chi2_best = np.nanmin(chi_sq)
idx = np.argmin(chi_sq)
best_fit = [ra_results[idx],dec_results[idx]]
print("Best chi2 reduced = ", chi2_best)
print("Best RA/DEC = ", best_fit)

## Isolate region around best fit
dr = 1
idx1 = np.where((ra_results >= best_fit[0]-dr) & (ra_results <= best_fit[0]+dr))
chi_sq = chi_sq[idx1]
ra_results = ra_results[idx1]
dec_results = dec_results[idx1]

idx2 = np.where((dec_results >= best_fit[1]-dr) & (dec_results <= best_fit[1]+dr))
chi_sq = chi_sq[idx2]
ra_results = ra_results[idx2]
dec_results = dec_results[idx2]

#index_err = np.where(chi_sq < (chi2_best+2.296) )
index_err = np.where(chi_sq < (chi2_best+1) )
chi_err = chi_sq[index_err]
ra_err = ra_results[index_err]
dec_err = dec_results[index_err]

## save arrays
#np.save('ra_err',ra_err)
#np.save('dec_err',dec_err)
## fit an ellipse to the data
ra_mean = np.mean(ra_err)
dec_mean = np.mean(dec_err)
a,b,theta = ellipse_hull_fit(ra_err,dec_err,ra_mean,dec_mean)
angle = theta*180/np.pi
## want to measure east of north (different than python)
angle_new = 90-angle
if angle_new<0:
    angle_new=360+angle_new
#angle_new = 360-angle_new
#angle = 360-angle
ellipse_params = np.around(np.array([a,b,angle_new]),decimals=4)
ell = Ellipse(xy=(ra_mean,dec_mean),width=2*a,height=2*b,angle=angle,facecolor='lightgrey')
plt.gca().add_patch(ell)
plt.scatter(ra_err, dec_err, c=chi_err, cmap=cm.inferno_r,zorder=2)
plt.colorbar()
plt.title('a,b,thet=%s'%ellipse_params)
plt.xlabel('d_RA (mas)')
plt.ylabel('d_DE (mas)')
plt.gca().invert_xaxis()
plt.axis('equal')
plt.savefig("/Users/tgardner/ARMADA_epochs/%s/%s_%s_ellipse.pdf"%(target_id,target_id,date))
plt.close()

## write results to a txt file
t = np.around(np.nanmedian(time_obs),4)
sep,pa = np.around(cart2pol(best_fit[0],best_fit[1]),decimals=4)
f = open("/Users/tgardner/ARMADA_epochs/%s/%s_%s_chi2err.txt"%(target_id,target_id,date),"w+")
f.write("# date mjd sep(mas) pa(Deg) err_maj(mas) err_min(mas) err_pa(deg)\r\n")
f.write("%s %s %s %s %s %s %s"%(date,t,sep,pa,ellipse_params[0],ellipse_params[1],ellipse_params[2]))
f.close()
