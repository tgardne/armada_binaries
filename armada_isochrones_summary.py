######################################################################
## Tyler Gardner
##
## Pipeline to fit binary orbits
## and search for additional companions
##
## For binary orbits from MIRCX/GRAVITY
##
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from orbit_plotting import orbit_model,triple_orbit_model
from astroquery.simbad import Simbad
from skimage import io
from read_data import read_data,read_wds,read_orb6
import os
import pandas as pd
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from matplotlib.backends.backend_pdf import PdfPages

###########################################
## SETUP PATHS
###########################################

path = '/Users/tgardner/ARMADA_final/ARMADA_isochrones/'

fixed_metallicity = input('Fixed metallicity results? (y, [n])')
if fixed_metallicity=='y':
    summary = '%s/summary/fixed/'%path
else:
    summary = '%s/summary/varied/'%path
if not os.path.exists(summary):
    print("Creating target directory")
    os.makedirs(summary)

armada_file = '/Users/tgardner/ARMADA_final/full_target_list.csv' ## path to csv target file
df_armada = pd.read_csv(armada_file, dtype=object)

###########################################
## Specify Targets
###########################################

targets = [1976,2772,5143,6456,10453,11031,16753,17094,27176,29316,29573,31093,31297,34319,36058,
               37269,37711,38545,38769,40932,41040,43358,43525,45542,46273,47105,48581,49643,60107,64235,
               75974,78316,82446,87652,87822,107259,112846,114993,118889,127726,128415,129246,133484,133955,
               137798,137909,140159,140436,144892,145589,148283,153370,154569,156190,158140,160935,163346,
               166045,173093,178475,179950,185404,185762,189037,189340,195206,196089,196867,198183,199766,
               201038,206901,217676,217782,220278,224512]

failed_targets = []
hr_plots = []
cdiff_plots = []
magnitude_plots = []
chi2_plots = []
msum_photometry = []
msum_photometry_err = []
msum_dynamical = []

full_summary = input('Do individual summaries for all targets? (y,[n]): ')

for target_hd in targets:
    target_hd = str(target_hd)
    try:
        if fixed_metallicity=='y':
            directory='%s/HD_%s/fixed/aavso'%(path,target_hd)
            if not os.path.exists(directory):
                directory='%s/HD_%s/fixed/simbad'%(path,target_hd)
        else:
            directory='%s/HD_%s/varied/aavso'%(path,target_hd)
            if not os.path.exists(directory):
                directory='%s/HD_%s/varied/simbad'%(path,target_hd)
        hr_plot = ''
        for file in os.listdir(directory):
            if file.endswith("_hrdiagram.png"):
                hr_plot = os.path.join(directory, file)
                hr_plots.append(hr_plot)
            if file.endswith("_cdiff.png"):
                cdiff_plot = os.path.join(directory, file)
                cdiff_plots.append(cdiff_plot)
            if file.endswith("_magnitudes.png"):
                magnitude_plot = os.path.join(directory, file)
                magnitude_plots.append(magnitude_plot)
            if file.endswith("_chi2grid.png"):
                chi2_plot = os.path.join(directory, file)
                chi2_plots.append(chi2_plot)
            if file.endswith("_photometry_masses.txt"):
                file = open(os.path.join(directory, file))
                lines = file.readlines()
                if fixed_metallicity=='y':
                    msum_p = float(lines[1].split()[0])
                    msum_photometry.append(msum_p)
                    msum_photometry_err.append(np.nan)
                else:
                    msum_p = float(lines[2].split()[1])
                    msum_e = (float(lines[3].split()[1]) - float(lines[1].split()[1])) / 2
                    msum_photometry.append(msum_p)
                    msum_photometry_err.append(msum_e)

        if hr_plot=='':
            failed_targets.append(target_hd)
            continue
        
        idx_armada = np.where(df_armada['HD'] == target_hd)[0][0]
        
        ## choose which distance to use -- Gaia, HIP, or something else
        distance_gaia = ufloat(float(df_armada['Gaia_distance (pc)'][idx_armada]), float(df_armada['Gaia_distance_err (pc)'][idx_armada]))
        distance_hip = ufloat(float(df_armada['HIP_distance (pc)'][idx_armada]), float(df_armada['HIP_distance_err (pc)'][idx_armada]))
        distance_kervella = ufloat(float(df_armada['kerv_dist'][idx_armada]), float(df_armada['e_kerv'][idx_armada]))

        distances = np.array([distance_gaia,distance_hip,distance_kervella])
        idx_lowest = np.nanargmin(unumpy.std_devs(distances))
        distance_best = distances[idx_lowest]

        #Finding Dynamical Mass
        Mdyn_over_d3 = float(df_armada['Mdyn_over_d3 (x10e-6)'][idx_armada])
        Mdyn_over_d3_err = float(df_armada['Mdyn_over_d3_err (x10e-6)'][idx_armada])
        mdyn_over_d3_float = ufloat(Mdyn_over_d3, Mdyn_over_d3_err) * 10 ** (-6)
        mdyn_mcmc = mdyn_over_d3_float * (distance_best ** 3)
        msum_dynamical.append(mdyn_mcmc)

        if full_summary=='y':

            f = plt.figure(figsize=(10,5))
            ax1 = f.add_subplot(231)
            ax2 = f.add_subplot(232)
            ax3 = f.add_subplot(233)
            ax4 = f.add_subplot(234)
            ax5 = f.add_subplot(236)
    
            img = plt.imread(chi2_plot)
            ax1.imshow(img)
            ax1.axis('off')
    
            img = plt.imread(magnitude_plot)
            ax2.imshow(img)
            ax2.axis('off')
    
            img = plt.imread(cdiff_plot)
            ax3.imshow(img)
            ax3.axis('off')
    
            img = plt.imread(hr_plot)
            ax4.imshow(img)
            ax4.axis('off')
    
            ax5.errorbar(mdyn_mcmc.nominal_value,msum_p,xerr=mdyn_mcmc.std_dev,fmt='o',color='black')
            ax5.plot([0,10],[0,10],'--')
            asp = np.diff(ax5.get_xlim())[0] / np.diff(ax5.get_ylim())[0]
            ax5.set_aspect(asp)
            ax5.set_xlabel('Mdyn (solar)')
            ax5.set_ylabel('Mphot (solar)')
    
            plt.subplots_adjust(wspace=0,hspace=0)
            plt.tight_layout()
            plt.savefig('%s/HD_%s_summary_plot.pdf'%(summary,target_hd),bbox_inches='tight')
    
            plt.close()

    except:
        failed_targets.append(target_hd)  
print(failed_targets)

## Chi2 plots
with PdfPages('%s/chi2_plots.pdf'%(summary)) as pdf:
    
    plot_number = len(chi2_plots)
    print('Number of plots = ', plot_number)
    number_pages = int(plot_number/16)
    number_leftover = plot_number%16

    for page_idx in np.arange(number_pages):
        img_idx = np.arange(page_idx*16,page_idx*16+16,1)
        fig,ax = plt.subplots(4,4,figsize=(20,20))
        ax = ax.flatten()
        for idx,i in enumerate(img_idx):
            img = plt.imread(chi2_plots[i])
            ax[idx].imshow(img)
            ax[idx].axis('off')
            plt.subplots_adjust(wspace=0,hspace=0)
        #fig.text(0.5, 0.04, '$\Delta$ RA (mas)', ha='center')
        #fig.text(0, 0.5, '$\Delta$ DEC (mas)', va='center', rotation='vertical')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    ## Last page -- the leftovers
    page_idx = number_pages
    img_idx = np.arange(page_idx*16,page_idx*16+number_leftover,1)  
    fig,ax = plt.subplots(4,4,figsize=(20,20))
    ax = ax.flatten()
    for idx,i in enumerate(img_idx):
        img = plt.imread(chi2_plots[i])
        ax[idx].imshow(img)
        ax[idx].axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()

## magnitude plots
with PdfPages('%s/magnitude_plots.pdf'%(summary)) as pdf:
    
    plot_number = len(magnitude_plots)
    print('Number of plots = ', plot_number)
    number_pages = int(plot_number/16)
    number_leftover = plot_number%16

    for page_idx in np.arange(number_pages):
        img_idx = np.arange(page_idx*16,page_idx*16+16,1)
        fig,ax = plt.subplots(4,4,figsize=(20,20))
        ax = ax.flatten()
        for idx,i in enumerate(img_idx):
            img = plt.imread(magnitude_plots[i])
            ax[idx].imshow(img)
            ax[idx].axis('off')
            plt.subplots_adjust(wspace=0,hspace=0)
        #fig.text(0.5, 0.04, '$\Delta$ RA (mas)', ha='center')
        #fig.text(0, 0.5, '$\Delta$ DEC (mas)', va='center', rotation='vertical')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    ## Last page -- the leftovers
    page_idx = number_pages
    img_idx = np.arange(page_idx*16,page_idx*16+number_leftover,1)  
    fig,ax = plt.subplots(4,4,figsize=(20,20))
    ax = ax.flatten()
    for idx,i in enumerate(img_idx):
        img = plt.imread(magnitude_plots[i])
        ax[idx].imshow(img)
        ax[idx].axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()

## magnitude plots
with PdfPages('%s/cdiff_plots.pdf'%(summary)) as pdf:
    
    plot_number = len(cdiff_plots)
    print('Number of plots = ', plot_number)
    number_pages = int(plot_number/16)
    number_leftover = plot_number%16

    for page_idx in np.arange(number_pages):
        img_idx = np.arange(page_idx*16,page_idx*16+16,1)
        fig,ax = plt.subplots(4,4,figsize=(20,20))
        ax = ax.flatten()
        for idx,i in enumerate(img_idx):
            img = plt.imread(cdiff_plots[i])
            ax[idx].imshow(img)
            ax[idx].axis('off')
            plt.subplots_adjust(wspace=0,hspace=0)
        #fig.text(0.5, 0.04, '$\Delta$ RA (mas)', ha='center')
        #fig.text(0, 0.5, '$\Delta$ DEC (mas)', va='center', rotation='vertical')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    ## Last page -- the leftovers
    page_idx = number_pages
    img_idx = np.arange(page_idx*16,page_idx*16+number_leftover,1)  
    fig,ax = plt.subplots(4,4,figsize=(20,20))
    ax = ax.flatten()
    for idx,i in enumerate(img_idx):
        img = plt.imread(cdiff_plots[i])
        ax[idx].imshow(img)
        ax[idx].axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()

## magnitude plots
with PdfPages('%s/hr_plots.pdf'%(summary)) as pdf:
    
    plot_number = len(hr_plots)
    print('Number of plots = ', plot_number)
    number_pages = int(plot_number/16)
    number_leftover = plot_number%16

    for page_idx in np.arange(number_pages):
        img_idx = np.arange(page_idx*16,page_idx*16+16,1)
        fig,ax = plt.subplots(4,4,figsize=(20,20))
        ax = ax.flatten()
        for idx,i in enumerate(img_idx):
            img = plt.imread(hr_plots[i])
            ax[idx].imshow(img)
            ax[idx].axis('off')
            plt.subplots_adjust(wspace=0,hspace=0)
        #fig.text(0.5, 0.04, '$\Delta$ RA (mas)', ha='center')
        #fig.text(0, 0.5, '$\Delta$ DEC (mas)', va='center', rotation='vertical')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    ## Last page -- the leftovers
    page_idx = number_pages
    img_idx = np.arange(page_idx*16,page_idx*16+number_leftover,1)  
    fig,ax = plt.subplots(4,4,figsize=(20,20))
    ax = ax.flatten()
    for idx,i in enumerate(img_idx):
        img = plt.imread(hr_plots[i])
        ax[idx].imshow(img)
        ax[idx].axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()

    ## mass summary plots
    plt.errorbar(unumpy.nominal_values(msum_dynamical),msum_photometry,xerr=unumpy.std_devs(msum_dynamical),yerr=msum_photometry_err,fmt='.',color='black')
    min_val = np.nanmin(np.concatenate([unumpy.nominal_values(msum_dynamical),msum_photometry]))
    max_val = np.nanmax(np.concatenate([unumpy.nominal_values(msum_dynamical),msum_photometry]))
    plt.plot([0,max_val+1],[0,max_val+1],'--',color='lightgrey')
    #plt.xlim(min_val,max_val)
    #plt.ylim(min_val,max_val)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.savefig('%s/mass_summary.pdf'%(summary))