import pdb
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import random
from astroquery.simbad import Simbad
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
import pandas as pd
from tqdm import tqdm
import time
from isochrones.mist import MIST_EvolutionTrack, MIST_Isochrone
from isochrones import get_ichrone
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import corner
from isochrones.mist.bc import MISTBolometricCorrectionGrid
import fitz
import matplotlib.image as mpimg
from skimage import io
from PIL import Image

Mist_iso = MIST_Isochrone()
Mist_evoTrack = MIST_EvolutionTrack()

matplotlib.rcParams['figure.figsize'] = (8, 5)

#save_directory = '/Users/tgardner/ARMADA_isochrones/' ## path for saved files
#summary_directory = '/Users/tgardner/ARMADA_isochrones/summary/' ## path for saved files
#armada_file = '/Users/tgardner/armada_binaries/full_target_list.csv' ## path to csv target file
#photometry_file = '/Users/tgardner/armada_binaries/Photometry.csv'
#csv = '/Users/tgardner/ARMADA_isochrones/target_info_hip_all_sigma.csv'

summary_directory = '/home/colton/ARMADA_binaries/summary/' ## path for saved files
save_directory = '/home/colton/ARMADA_binaries/' ## path for saved files
armada_file = '/home/colton/armada_binaries/full_target_list_newest_version3.csv' ## path to csv target file
photometry_file = '/home/colton/armada_binaries/Photometry.csv'
csv = '/home/colton/armada_binaries/target_info_all_sigma.csv'
orbit_directory = '/home/colton/ARMADA_binaries/ARMADA_orbits/'

Header =["HD", "M_Dyn", "M_Dyn_err",
                        "M_Tot", "M_Tot_err", "M1",
                        "M1_err", "M2", "M2_err",
                        "log_age", "log_age_err", "FeH", "Av",
                        "Redchi2"]

df_armada = pd.read_csv(armada_file, dtype=object)
df_photometry = pd.read_csv(photometry_file, dtype=object)
## note for saved files (e.g. 'hip' for hipparcos distance, or 'gaia')

note = input("Choose note for saved files in this run: ") 

Target_List = ['6456','1976', '2772','5143', '6456','10453', '11031', '16753', '17094', '27176', '29316', '29573', '31093', '31297', '34319', '36058',  '37711', '38545'

    , '38769', '40932', '41040', '43358', '43525', '45542', '46273', '47105', '48581', '49643', '60107', '64235',
               '75974', '78316', '82446', '87652', '87822', '107259', '112846'
    , '114993', '118889', '127726', '128415', '129246', '133484', '133955', '137798', '140159', '140436', '144892',
               '145589', '148283', '153370', '154569', '156190', '158140'
    , '160935', '166045', '173093', '178475', '179950', '185404', '185762', '189037', '189340', '195206',
               '196089', '196867', '198183', '199766', '201038', '206901'
    , '217676', '217782', '220278', '224512']

Target_List_Fail = []


## Fuction to fit a single star model
def single_star_fit(params,split_mag_star,d_mod,Av):
    split_mag_absolute = split_mag_star - d_mod
    split_mag_val = unumpy.nominal_values(split_mag_absolute)
    split_mag_err = unumpy.std_devs(split_mag_absolute)

    age = params['age']
    mass = params['mass']
    feh = params['feh']

    a1 = tracks.generate(mass, age, feh, return_dict=True)

    split_mag_model = np.array([(a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           a1["H_mag"],
                           a1["K_mag"]])
    diff = (split_mag_val - split_mag_model) / split_mag_err
    return(diff)

## Objective function for binaries to be minimized for lmfit
def isochrone_model(params, TOT_mag_star, D_mag_star, d_mod, Av):
    TOT_mag_absolute = TOT_mag_star - d_mod
    TOT_mag_val = unumpy.nominal_values(TOT_mag_absolute)
    TOT_mag_err = unumpy.std_devs(TOT_mag_absolute)

    D_mag_val = unumpy.nominal_values(D_mag_star)
    D_mag_err = unumpy.std_devs(D_mag_star)

    age = params['age']
    m1 = params['mass1']
    m2 = params['mass2']
    feh = params['feh']

    a1 = tracks.generate(m1, age, feh, return_dict=True)
    a2 = tracks.generate(m2, age, feh, return_dict=True)


    mag1_model = np.array([(a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           a1["J_mag"],
                           a1["H_mag"],
                           a1["K_mag"]])

    mag2_model = np.array([(a2['Mbol'] - bc_grid_B.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_V.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_I.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           a1["J_mag"],
                           a2["H_mag"],
                           a2["K_mag"]])

    D_mag_model = mag2_model - mag1_model
    TOT_mag_model = -2.5 * np.log10(10 ** (-0.4 * mag1_model) + 10 ** (-0.4 * mag2_model))

    diff1 = (D_mag_val - D_mag_model) / D_mag_err
    diff2 = (TOT_mag_val - TOT_mag_model) / TOT_mag_err

    # print(age.value, m1.value, m2.value,diff1,diff2)
    if np.isnan(a1['Mbol']) or np.isnan(a2['Mbol']):
        diff1[:] = np.inf
        diff2[:] = np.inf

    return np.concatenate([diff1, diff2])

## Objective function to find the best distance
def best_distance(gaia, hip, kervella):
    distances = np.array([gaia,hip,kervella])
    idx_lowest = np.nanargmin(unumpy.std_devs(distances))
    #pdb.set_trace()
    return(distances[idx_lowest])

## Objective function for binaries to be minimized for lmfit
def isochrone_model_v2(params, TOT_mag_star, D_mag_star, d_mod, Av):
    TOT_mag_absolute = TOT_mag_star - d_mod
    TOT_mag_val = unumpy.nominal_values(TOT_mag_absolute)
    TOT_mag_err = unumpy.std_devs(TOT_mag_absolute)

    D_mag_val = unumpy.nominal_values(D_mag_star)
    D_mag_err = unumpy.std_devs(D_mag_star)

    age = params['age']
    m1 = params['mass1']
    m2 = params['mass2']
    feh = params['feh']

    a1 = tracks.generate(m1, age, feh, return_dict=True)
    a2 = tracks.generate(m2, age, feh, return_dict=True)

    # if np.isnan(a1['Mbol']) or np.isnan(a2['Mbol']):
    #    return np.inf

    mag1_model = np.array([(a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           a1["H_mag"],
                           a1["K_mag"]])

    mag2_model = np.array([(a2['Mbol'] - bc_grid_V.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_V.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_I.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           a2["H_mag"],
                           a2["K_mag"]])

    D_mag_model = mag2_model - mag1_model

    diff1 = (D_mag_val - D_mag_model) / D_mag_err

    ## Bessel_U, Bessel_B, Bessel_V, Bessel_R, Gaia_G, Bessel_I, SDSS_z, 2MASS_J, 2MASS_H, 2MASS_K
    # Wavelengths = np.array([365, 445, 551, 658, 673, 806, 905, 1250, 1650, 2150])

    mag1_model = np.array([(a1['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           a1["H_mag"],
                           a1["K_mag"]])

    mag2_model = np.array([(a2['Mbol'] - bc_grid_U.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_B.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_V.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_R.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_I.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_J.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           a2["H_mag"],
                           a2["K_mag"]])

    TOT_mag_model = -2.5 * np.log10(10 ** (-0.4 * mag1_model) + 10 ** (-0.4 * mag2_model))

    diff2 = (TOT_mag_val - TOT_mag_model) / TOT_mag_err
    # print(np.concatenate([diff1,diff2]).size)

    return np.concatenate([diff1, diff2])




feh_set = [-0.1,0,0.1]

for target_hd in Target_List:
    print('--' * 10)
    print('--' * 10)
    print("Doing Target HD %s" % target_hd)
    # For Combined 3x3 plots
    all_mass1_result = []
    all_chi2_grid = []
    all_mass2_result = []
    all_chi2_grid2 = []
    all_ages = []
    all_chi2_grid3 = []
    all_xwave = []
    all_TOTmag_model = []
    all_data_wave = []
    all_split_mag1 = []
    all_split_mag2 = []
    all_model1 = []
    all_model2 = []
    all_yplot = []
    all_yplot1 = []
    all_Dmag_model = []
    all_modelx_best = []
    all_modely_best = []
    all_age_best = []

    for feh in feh_set:
        #try:
        print('--' * 10)
        print('--' * 10)
        print("Doing Metallicity [Fe/H] = %s" % feh)


        ## Create directory for saved files, if it doesn't already exist
        directory = "%s/HD_%s/" % (save_directory, target_hd)
        if not os.path.exists(directory):
            print("Creating directory")
            os.makedirs(directory)
        ## Create directory for saved files, if it doesn't already exist
        directory2 = summary_directory
        if not os.path.exists(directory2):
            print("Creating directory")
            os.makedirs(directory2)

        idx = np.where(df_armada['HD'] == target_hd)[0][0]
        Av = float(df_armada['Av'][idx])

        cdiff_h = ufloat(float(df_armada['dmag_h'][idx]), float(df_armada['dmag_h_err'][idx]))
        cdiff_k = ufloat(float(df_armada['dmag_k'][idx]), float(df_armada['dmag_k_err'][idx]))
        cdiff_i = ufloat(float(df_armada['dmag_speckle_i'][idx]), float(df_armada['dmag_speckle_i_err'][idx]))
        cdiff_b = ufloat(float(df_armada['dmag_speckle_b'][idx]), float(df_armada['dmag_speckle_b_err'][idx]))
        cdiff_wds = ufloat(float(df_armada['dmag_wds_v'][idx]), float(df_armada['dmag_wds_v_err'][idx]))

        fratio_h = 10 ** (cdiff_h / 2.5)
        fratio_k = 10 ** (cdiff_k / 2.5)
        fratio_i = 10 ** (cdiff_i / 2.5)
        fratio_b = 10 ** (cdiff_b / 2.5)
        fratio_wds = 10 ** (cdiff_wds / 2.5)

        ## get total magnitudes and errors from photometry file
        idx1 = np.where(df_photometry['HD'] == target_hd)[0][0]
        utot = ufloat(np.nan, np.nan)
        btot = ufloat(float(df_photometry['B_ I/259/tyc2'][idx1]), float(df_photometry['B_error_I/259/tyc2'][idx1]))
        vtot = ufloat(float(df_photometry['V_ I/259/tyc2'][idx1]), float(df_photometry['V_error_I/259/tyc2'][idx1]))
        rtot = ufloat(np.nan, np.nan)
        gtot = ufloat(float(df_photometry['G _I/350/gaiaedr3'][idx1]), float(df_photometry['G_err'][idx1]))
        itot = ufloat(np.nan, np.nan)
        jtot = ufloat(float(df_photometry['J_ II/246/out'][idx1]), float(df_photometry['J_err'][idx1]))
        htot = ufloat(float(df_photometry['H_ II/246/out'][idx1]), float(df_photometry['H_err'][idx1]))
        ktot = ufloat(float(df_photometry['K_ II/246/out'][idx1]), float(df_photometry['K_err'][idx1]))

        ## Compute individual magnitudes from flux ratios and total magnitudes
        ## Mostly for plotting. Though we will use these to estimate M1 and M2 roughly
        data_wave = np.array([551, 562, 832, 1630, 2190])
        k1 = -2.5 * log10(10 ** (-ktot / 2.5) / (1 + 10 ** (-cdiff_k / 2.5)))
        k2 = cdiff_k + k1
        h1 = -2.5 * log10(10 ** (-htot / 2.5) / (1 + 10 ** (-cdiff_h / 2.5)))
        h2 = cdiff_h + h1
        i1 = -2.5 * log10(10 ** (-gtot / 2.5) / (1 + 10 ** (-cdiff_i / 2.5)))  ## NOTE: G is not the best here probably
        i2 = cdiff_i + i1
        v1 = -2.5 * log10(10 ** (-vtot / 2.5) / (1 + 10 ** (-cdiff_wds / 2.5)))
        v2 = cdiff_wds + v1
        b1 = -2.5 * log10(10 ** (-vtot / 2.5) / (1 + 10 ** (-cdiff_b / 2.5)))
        b2 = cdiff_b + b1

        split_mag1 = np.array([b1, v1, i1, h1, k1])
        split_mag2 = np.array([b2, v2, i2, h2, k2])

        ## A new isochrone model that interpolate the model photometry to add more constraint
        tracks = get_ichrone('mist', tracks=True, accurate=True)
        bc_grid_U = MISTBolometricCorrectionGrid(['U'])
        bc_grid_B = MISTBolometricCorrectionGrid(['B'])
        bc_grid_V = MISTBolometricCorrectionGrid(['V'])
        bc_grid_R = MISTBolometricCorrectionGrid(['R'])
        bc_grid_I = MISTBolometricCorrectionGrid(['I'])
        bc_grid_Z = MISTBolometricCorrectionGrid(['z'])
        bc_grid_J = MISTBolometricCorrectionGrid(['J'])

        ## choose which distance to use -- Gaia, HIP, or something else
        distance_gaia = ufloat(float(df_armada['Gaia_distance (pc)'][idx]), float(df_armada['Gaia_distance_err (pc)'][idx]))
        distance_hip = ufloat(float(df_armada['HIP_distance (pc)'][idx]), float(df_armada['HIP_distance_err (pc)'][idx]))
        distance_kervella = ufloat(float(df_armada['kerv_dist'][idx]), float(df_armada['e_kerv'][idx]))

        distance_best = best_distance(distance_gaia, distance_hip, distance_kervella)
        #pdb.set_trace()
        distance_low = distance_best - distance_best.std_dev
        distance_high = distance_best + distance_best.std_dev


        distance_set =[distance_low,distance_best,distance_high]

        ## We only want to use this one below if we only want to look at one best distance!
        #distance_set = [distance_best]
        #distance_set = [distance_hip]

        #pdb.set_trace()
        for distance in distance_set:

            print("Distance = ", distance, 'pc')

            d_modulus = 5 * log10(distance) - 5


            name = f"{note}_{feh}_{distance.nominal_value}"

            ## central wavelengths of magnitudes chosen above, smallest to largest - ubvrgjhk
            ## Bessel_U, Bessel_B, Bessel_V, Bessel_R, Gaia_G, Bessel_I, SDSS_z, 2MASS_J, 2MASS_H, 2MASS_K
            x = np.array([365, 445, 551, 658, 673, 806, 1250, 1650, 2150])
            y = np.array([utot.nominal_value, btot.nominal_value, vtot.nominal_value,
                          rtot.nominal_value, gtot.nominal_value, itot.nominal_value, jtot.nominal_value,
                          htot.nominal_value, ktot.nominal_value])
            yerr = np.array([utot.std_dev, btot.std_dev, vtot.std_dev,
                             rtot.std_dev, gtot.std_dev, itot.std_dev, jtot.std_dev,
                             htot.std_dev, ktot.std_dev])

            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))
            fig.tight_layout(pad=5)

            #ax1.set_title("Total Apparent Magnitudes, HD %s" % target_hd)
            #ax1.set_xlabel("Wavelength (nm)")
            #ax1.set_ylabel("Apparent Mag")
            #ax1.errorbar(x, y, yerr, fmt='o--')

            #ax2.set_title("SED from Speckle and Interferometry, HD %s" % target_hd)
            #ax2.errorbar(data_wave, unumpy.nominal_values(split_mag1), unumpy.std_devs(split_mag1), fmt='o--')
            #ax2.errorbar(data_wave, unumpy.nominal_values(split_mag2), unumpy.std_devs(split_mag2), fmt='o--')
            #ax2.set_xlabel('Wavelength (nm)')
            #ax2.set_ylabel('Apparent Mag')
            #ax2.invert_yaxis()

            ## Choose observables for fitting
            TOT_Mag = np.array([utot, btot, vtot, rtot, itot, jtot, htot, ktot])
            DiffM = np.array([cdiff_wds, cdiff_b, cdiff_i, cdiff_h, cdiff_k])

            ##################
            ## Now let's do a rough estimation of M1 and M2 for age grid
            ##################
            print("Estimating M1 and M2 for age grid")
            log_age_guess = 8 ## rough age of main sequence Astar... might want to check
            mass1_grid = np.linspace(0.5,5,50)
            mass2_grid = np.linspace(0.5,5,50)


            ## Explore a grid of chi2 over M1 at fixed age
            chi2_grid = []
            mass1_result = []
            for mm in mass1_grid:
                try:
                    params = Parameters()
                    params.add('age', value=log_age_guess, vary=False) ## NOTE: I could vary this!
                    params.add('mass', value=mm, vary=False)
                    params.add('feh', value=feh, vary=False)
                    minner = Minimizer(single_star_fit, params, fcn_args=(split_mag1, d_modulus.nominal_value, Av),
                                    nan_policy='omit')
                    result = minner.minimize()
                    chi2_grid.append(result.redchi)
                    mass1_result.append(mm)
                except:
                    # print("Fails at log age = %s"%aa)
                    pass

            idx_mass1 = np.argmin(chi2_grid)
            mass1_guess = mass1_result[idx_mass1]
            all_mass1_result.append(mass1_result)
            all_chi2_grid.append(chi2_grid)

            print("Mass 1 Guess = %s" % mass1_guess)

            ax2.scatter(mass1_result, chi2_grid, alpha=0.6, marker="+", color="blue", label = 'Mass 1')
            ax2.plot(mass1_result, chi2_grid, alpha=0.6, ls="--", color="black")
            ax2.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
            #ax3.legend()
            ax2.set_yscale("log")
            ax2.set_xlabel('Mass (solar)', fontsize=15)
            ax2.set_ylabel(r'$\chi^2$', fontsize=15)
            #ax3.set_title('Mass 1 Guess = %s Msun'%np.around(mass1_guess,2))
            #plt.savefig("%s/HD_%s_%s_chi2_mass1.pdf" % (directory, target_hd, name))
            #plt.close()

            ## Explore a grid of chi2 over M1 at fixed age
            chi2_grid2 = []
            mass2_result = []
            for mm in mass2_grid:
                try:
                    params = Parameters()
                    params.add('age', value=log_age_guess, vary=False) ## NOTE: I could vary this!
                    params.add('mass', value=mm, vary=False)
                    params.add('feh', value=feh, vary=False)
                    minner = Minimizer(single_star_fit, params, fcn_args=(split_mag2, d_modulus.nominal_value, Av),
                                    nan_policy='omit')
                    result = minner.minimize()
                    chi2_grid2.append(result.redchi)
                    mass2_result.append(mm)
                except:
                    # print("Fails at log age = %s"%aa)
                    pass

            idx_mass2 = np.argmin(chi2_grid2)
            all_mass2_result.append(mass2_result)
            all_chi2_grid2.append(chi2_grid2)
            mass2_guess = mass2_result[idx_mass2]
            print("Mass 2 Guess = %s" % mass2_guess)

            ax2.scatter(mass2_result, chi2_grid, alpha=0.6, marker="+", color="Red", label ='Mass 2')
            ax2.plot(mass2_result, chi2_grid, alpha=0.6, ls="--", color="black")
            #ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
            ax2.legend()
            ax2.set_yscale("log")
            ax2.set_xlabel('Mass 2 (solar)', fontsize=15)
            ax2.set_ylabel(r'$\chi^2$', fontsize=15)
            ax2.set_title('Mass 1 & 2 Guess = %s Msun'%np.around(mass2_guess,2))
            #plt.savefig("%s/HD_%s_%s_chi2_mass2.pdf" % (directory, target_hd, name))
            #plt.close()

            ##################
            ## Now let's find the best age with our M1 and M2 starting points
            ##################
            print('Grid Searching over AGE to find best fit')
            #pdb.set_trace()
            ## Explore a grid of chi2 over age -- this paramter does not fit properly in least squares
            chi2_grid3 = []
            ages = []
            age_grid = np.linspace(6, 10, 100)  ## do fewer steps to go faster

            for aa in tqdm(age_grid):
                try:
                    params = Parameters()
                    params.add('age', value=aa, vary=False)
                    params.add('mass1', value=mass1_guess, min=0)
                    params.add('mass2', value=mass2_guess, min=0)
                    params.add('feh', value=feh, vary=False)  # min=-0.5, max=0.5)
                    minner = Minimizer(isochrone_model_v2, params, fcn_args=(TOT_Mag, DiffM, d_modulus.nominal_value, Av),
                                    nan_policy='omit')
                    result = minner.minimize()
                    chi2_grid3.append(result.redchi)
                    ages.append(aa)
                except:
                    # print("Fails at log age = %s"%aa)
                    pass

            idx2 = np.argmin(chi2_grid3)

            all_chi2_grid3.append(chi2_grid3)
            all_ages.append(ages)
            age_best = ages[idx2]
            age_max = ages[-1]

            ax3.scatter(ages, chi2_grid3, alpha=0.6, marker="+", color="blue")
            ax3.plot(ages, chi2_grid3, alpha=0.6, ls="--", color="black")
            ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
            ax3.legend()
            ax3.set_yscale("log")
            ax3.set_xlabel('Age', fontsize=15)
            ax3.set_ylabel(r'$\chi^2$', fontsize=15)
            #fig.show()
            #fig.savefig("%s/HD_%s_%s_original_SEDs_hr_chi2_age_mass.pdf" % (directory, target_hd, note))
            #fig.savefig("%s/HD_%s_%s_original_SEDs_hr_chi2_age_mass.pdf" % (directory2, target_hd, note))



            print("Best log Age = %s" % age_best)
            print('Fit fails at log Age = %s' % age_max)
            ## start with a chi2 fit (this does not always work)
            ## NOTE --> Since isochrones has a fixed age grid, the fitting currently fails to optimize this parameter
            ## We can search age on a grid, or change the default step size for the parameter
            params = Parameters()
            params.add('age', value=age_best, min=6, max=age_max)
            params.add('mass1', value=mass1_guess, min=0)  # , max=max_mass)
            params.add('mass2', value=mass2_guess, min=0)  # , max=max_mass)
            params.add('feh', value=feh, vary=False)  # min=-0.5, max=0.5)
            minner = Minimizer(isochrone_model_v2, params, fcn_args=(TOT_Mag, DiffM, d_modulus.nominal_value, Av),
                               nan_policy='omit')
            result = minner.minimize()
            #report_fit(result)

            ## We probably want least squares result as "best" parameters. TBD
            age_best = result.params['age'].value
            mass1_best = result.params['mass1'].value
            mass2_best = result.params['mass2'].value
            feh_best = result.params['feh'].value
            redchi2_best = result.redchi

            ########################
            ########################
            ## Setup MCMC fit
            emcee_params = result.params.copy()
            nwalkers = 2 * len(emcee_params)
            steps = 300
            burn = 100
            thin = 1

            print("Running MCMC chains: ")
            ## Do MCMC fit (this cell could take some time, depending on steps)
            minner = Minimizer(isochrone_model_v2, emcee_params, fcn_args=(TOT_Mag, DiffM, d_modulus.nominal_value, Av),
                               nan_policy='omit')
            result = minner.minimize(method='emcee', steps=steps, burn=burn, thin=thin, nwalkers=nwalkers)
            #print(report_fit(result))
            chains = result.flatchain

            ## save chains (so we don't need to run large ones again)
            #print(chains.shape)
            np.save("%s/HD_%s_%s_chains.npy" % (directory, target_hd, name), chains)

            ## load chains -- NOTE: Could start from here if a run has already been completed
            chains = np.load("%s/HD_%s_%s_chains.npy" % (directory, target_hd, name))
            try:
                emcee_plot = corner.corner(chains, labels=result.var_names)
                plt.savefig('%s/HD_%s_%s_corner.pdf' % (directory, target_hd, name))
                plt.savefig('%s/HD_%s_%s_corner.pdf' % (directory2, target_hd, name))
                plt.close()
            except:
                #print(result.var_names)
                emcee_plot = corner.corner(chains)
                plt.savefig('%s/HD_%s_%s_corner.pdf' % (directory, target_hd, name))
                plt.savefig('%s/HD_%s_%s_corner.pdf' % (directory2, target_hd, name))
                plt.close()

            age_chain = chains[:, 0]
            mass1_chain = chains[:, 1]
            mass2_chain = chains[:, 2]

            ## I am using least-sqaures fit for best values, though we could also get some from chains:
            ## Using errors from chains
            # age_best = np.median(age_chain)
            # mass1_best = np.median(mass1_chain)
            # mass2_best = np.median(mass2_chain)
            age_err = np.std(age_chain)
            mass1_err = np.std(mass1_chain)
            mass2_err = np.std(mass2_chain)

            age = ufloat(age_best, age_err)
            mass1 = ufloat(mass1_best, mass1_err)
            mass2 = ufloat(mass2_best, mass2_err)

            print("Final Fitted Results:")
            print('Log Age = ', age)
            print('M1 (solar) = ', mass1)
            print('M2 (solar) = ', mass2)
            print('Msum (solar) = ', mass1 + mass2)

            ## Now make plots from best fit results
            a1_best = tracks.generate(mass1_best, age_best, feh, return_dict=True)
            a2_best = tracks.generate(mass2_best, age_best, feh, return_dict=True)
            # if np.isnan(a1['Mbol']) or np.isnan(a2['Mbol']):
            #    return np.inf

            model1 = np.array(
                [(a1_best['Mbol'] - bc_grid_U.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a1_best['Mbol'] - bc_grid_B.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a1_best['Mbol'] - bc_grid_V.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a1_best['Mbol'] - bc_grid_R.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a1_best['Mbol'] - bc_grid_I.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a1_best['Mbol'] - bc_grid_J.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 a1_best["H_mag"] + d_modulus.nominal_value,
                 a1_best["K_mag"] + d_modulus.nominal_value])

            model2 = np.array(
                [(a2_best['Mbol'] - bc_grid_U.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a2_best['Mbol'] - bc_grid_B.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a2_best['Mbol'] - bc_grid_V.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a2_best['Mbol'] - bc_grid_R.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a2_best['Mbol'] - bc_grid_I.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 (a2_best['Mbol'] - bc_grid_J.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                 a2_best["H_mag"] + d_modulus.nominal_value,
                 a2_best["K_mag"] + d_modulus.nominal_value])

            Dmag_model = model2 - model1
            TOTmag_model = -2.5 * np.log10(10 ** (-0.4 * model1) + 10 ** (-0.4 * model2))

            ## central wavelengths of Simbad magnitudes chosen above (UBVRIJHK)
            x_wave = np.array([365.6, 435.3, 547.7, 658, 832, 1220, 1630, 2190])
            #fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
            #fig.tight_layout()
            yplot = TOT_Mag
            all_yplot.append(yplot)
            all_xwave.append(x_wave)
            all_TOTmag_model.append(TOTmag_model)
            #pdb.set_trace()
            ax4.set_title("Total Mag Model Fit, HD %s" % target_hd)
            ax4.errorbar(x_wave, unumpy.nominal_values(yplot), unumpy.std_devs(yplot), fmt='o', color='black')
            ax4.plot(x_wave, TOTmag_model, '--', color='red')
            # ax1.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('Total Mag')
            ax4.invert_yaxis()

            # ax1.gca().invert_yaxis()
            # plt.savefig("%s/HD_%s_%s_TOTmag_fit.pdf"%(directory,target_hd,note))

            yplot1 = DiffM
            all_data_wave.append(data_wave)
            all_yplot1.append(yplot1)
            all_Dmag_model.append(Dmag_model)

            ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd)
            ax5.errorbar(data_wave, unumpy.nominal_values(yplot1), unumpy.std_devs(yplot1), fmt='o', color='black')
            ax5.plot(x_wave, Dmag_model, '--', color='red')
            # ax2.set_xlabel('Wavelength (nm)')
            ax5.invert_yaxis()
            ax5.set_ylabel('Diff Mag')
            # plt.gca().invert_yaxis()
            # plt.savefig("%s/HD_%s_%s_Dmag_fit.pdf"%(directory,target_hd,note))
            # plt.show()

            all_split_mag1.append(split_mag1)
            all_split_mag2.append(split_mag2)
            all_model1.append(model1)
            all_model2.append(model2)

            ax6.set_title("Split SED Model Fit, HD %s" % target_hd)
            ax6.errorbar(data_wave, unumpy.nominal_values(split_mag1), unumpy.std_devs(split_mag1), fmt='o', color='black')
            ax6.errorbar(data_wave, unumpy.nominal_values(split_mag2), unumpy.std_devs(split_mag2), fmt='o', color='black')
            ax6.plot(x_wave, model1, '--', color='red')
            ax6.plot(x_wave, model2, '--', color='red')
            ax6.invert_yaxis()
            ax6.set_xlabel('Wavelength (nm)')
            ax6.set_ylabel('Apparent Mag')
            # plt.gca().invert_yaxis()
            # plt.savefig("%s/HD_%s_%s_split_fit.pdf"%(directory,target_hd,note))

            ###############
            ## Make an HR diagram plot
            ###############
            log_age_start = age_best - 1.0  ## starting age
            log_age_size = 0.5  ## step size
            log_age_steps = 5  ## number of steps

            paramList = [np.array([log_age_start, feh]) + np.array([log_age_size, 0]) * i for i in
                         range(0, log_age_steps)]
            isoList = [Mist_iso.isochrone(param[0], param[1]) for param in paramList]
            isoList_best = [Mist_iso.isochrone(age_best, feh)]

            #pdb.set_trace()
            bc_grid_B = MISTBolometricCorrectionGrid(['B'])
            bc_grid_V = MISTBolometricCorrectionGrid(['V'])
            bc_grid_I = MISTBolometricCorrectionGrid(['I'])

            Mbol_V = []
            for i, iso in enumerate(isoList):
                Mbol_V.append(bc_grid_V.interp([iso['Teff'], iso['logg'], feh, Av]).ravel())
            V = []
            for i, iso in enumerate(isoList):
                V.append(iso['Mbol'] - Mbol_V[i])

            H = []
            for iso in isoList:
                H.append(iso['H_mag'])

            K = []
            for iso in isoList:
                K.append(iso['K_mag'])

            Mbol_V_best = []
            for i, iso in enumerate(isoList_best):
                Mbol_V_best.append(bc_grid_V.interp([iso['Teff'], iso['logg'], feh, Av]).ravel())
            V_best = []
            for i, iso in enumerate(isoList_best):
                V_best.append(iso['Mbol'] - Mbol_V_best[i])

            H_best = []
            for iso in isoList_best:
                H_best.append(iso['H_mag'])

            K_best = []
            for iso in isoList_best:
                K_best.append(iso['K_mag'])

            ## Choose x/y axis. For example, V-H vs V
            xval1 = v1-h1 ## component 1
            yval1 = v1 - d_modulus
            xval2 = v2-h2 ## component 2
            yval2 = v2 - d_modulus

            xlabel = "V - H"
            ylabel = "V"

            iso_start = 100
            iso_end = 500
            for i, iso in enumerate(isoList):
                ## make sure model matches data magnitudes
                modelx = V[i][iso_start:iso_end] - H[i][iso_start:iso_end]
                modely = V[i][iso_start:iso_end]
                ax1.plot(modelx, modely, color='lightgrey')#,label=f"log age = {log_age_start + log_age_size * i} ")

            ## make plot
            ax1.errorbar(xval1.nominal_value, yval1.nominal_value,
                         xerr=xval1.std_dev, yerr=yval1.std_dev,
                         color="red")
            ax1.errorbar(xval2.nominal_value, yval2.nominal_value,
                         xerr=xval2.std_dev, yerr=yval2.std_dev,
                         color="red")
            for ii, iso in enumerate(isoList_best):
                ## make sure model matches data magnitudes
                modelx_best = V_best[ii][iso_start:iso_end] - H_best[ii][iso_start:iso_end]
                modely_best = V_best[ii][iso_start:iso_end]
                ax1.plot(modelx_best, modely_best, label=f"Best log age = {np.around(age_best,2)} ", color = 'black')

            all_modelx_best.append(modelx_best)
            all_modely_best.append(modely_best)
            all_age_best.append(age_best)

            ax1.set_xlabel(xlabel, fontsize=15)
            ax1.set_ylabel(ylabel, fontsize=15)
            ax1.invert_yaxis()
            ax1.set_title("HD %s" % target_hd, fontsize=15)
            ax1.legend()


            fig.savefig("%s/HD_%s_%s_all_SED_fit.pdf" % (directory, target_hd, note))


            #Finding Dynamical Mass
            Mdyn_over_d3 = float(df_armada['Mdyn_over_d3 (x10e-6)'][idx])
            Mdyn_over_d3_err = float(df_armada['Mdyn_over_d3_err (x10e-6)'][idx])
            mdyn_over_d3_float = ufloat(Mdyn_over_d3, Mdyn_over_d3_err) * 10 ** (-6)
            mdyn = mdyn_over_d3_float * (distance.nominal_value ** 3)

            #Save all of the desired Data (M_phot, M_dyn, Age, Feh, Av)
            m_tot= mass1+mass2
            df_new = pd.DataFrame(dict(HD=[target_hd], M_Dyn=[mdyn.nominal_value], M_Dyn_err=[mdyn.std_dev],
                                  M_Tot=[m_tot.nominal_value], M_Tot_err=[m_tot.std_dev],M1=[mass1.nominal_value],
                                 M1_err=[mass1.std_dev],M2=[mass2.nominal_value],M2_err=[mass2.std_dev],
                                log_age=[age.nominal_value], log_age_err=[age.std_dev],FeH=[feh_best], Av=[Av],
                               Redchi2=[redchi2_best]))
            file_name = f"{note}_{feh}_{distance.nominal_value}"
            print(df_new)
            df_new.to_csv('%s/HD_%s/target_info_%s.csv' % (save_directory, target_hd,file_name), mode='a', index=False, header=True)

            print('Going to New Target')
    fig = plt.figure(figsize=(11.69,8.27))

    gs = mpl.gridspec.GridSpec(6, 9, wspace=0.01, hspace=0.01)  # 2x2 grid
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # first row, second col
    ax2 = fig.add_subplot(gs[0:2,3:5])  # full second row
    ax3 = fig.add_subplot(gs[0:2, 6:8])  # full second row
    ax4 = fig.add_subplot(gs[3, 0:2])  # first row, second col
    ax5 = fig.add_subplot(gs[4,0:2])  # full second row
    ax6 = fig.add_subplot(gs[5, 0:2])  # full second row
    ax7 = fig.add_subplot(gs[3:5, 3:8])  # full second row


    for i in [1,7]:
        ax2.scatter(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, marker="+", color="blue", s=1)
        ax2.plot(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, ls="--", color="blue", linewidth =1)
        #ax2.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=5)
        ax2.set_ylabel(r'$\chi^2$', fontsize=5)
        ax2.scatter(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, marker="+", color="Red", s=1)
        ax2.plot(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, ls="--", color="blue", linewidth =1)
        # ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        #ax2.legend()
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=5)
        ax2.set_ylabel(r'$\chi^2$', fontsize=5)
        ax2.set_title('Mass 1 & 2 Guess', fontsize=5)
        #pdb.set_trace()

    for i in [3,5]:
        ax2.scatter(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, marker="+", color="blue",s = 1)
        ax2.plot(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, ls="--", color="red", linewidth =1)
        #ax2.axhline(y=1, color="red", alpha=0.6)
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=5)
        ax2.set_ylabel(r'$\chi^2$', fontsize=5)
        ax2.scatter(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, marker="+", color="Red",s = 1)
        ax2.plot(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, ls="--", color="red", linewidth =1)
        # ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        #ax2.legend()
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=5)
        ax2.set_ylabel(r'$\chi^2$', fontsize=5)
        ax2.set_title('Mass 1 & 2 Guess', fontsize=5)
        #pdb.set_trace()


    for i in [4]:
        ax2.scatter(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, marker="+", color="blue", label='Mass 1', s = 1)
        ax2.plot(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, ls="--", color="black", linewidth=1)
        ax2.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=5)
        ax2.set_ylabel(r'$\chi^2$', fontsize=5)
        ax2.scatter(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, marker="+", color="Red", label='Mass 2',  s = 1)
        ax2.plot(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, ls="--", color="black", linewidth = 1)
        # ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax2.legend(fontsize= 5)
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=5)
        ax2.set_ylabel(r'$\chi^2$', fontsize=5)
        ax2.set_title('Mass 1 & 2 Guess', fontsize=5)
        ax2.tick_params(axis='both', labelsize=5)
        #pdb.set_trace()

    for i in [1,7]:
        ax3.scatter(all_ages[i], all_chi2_grid3[i], alpha=0.6, marker="+", color="blue",  s = 1)
        ax3.plot(all_ages[i], all_chi2_grid3[i], alpha=0.6, ls="--", color="blue", linewidth=1)
        ax3.axhline(y=1, color="red", alpha=0.6)
        #ax3.legend()
        ax3.set_yscale("log")
        ax3.set_xlabel('Age', fontsize=5)
        ax3.set_ylabel(r'$\chi^2$', fontsize=5)
    #pdb.set_trace()

    for i in [3,5]:
        ax3.scatter(all_ages[i], all_chi2_grid3[i], alpha=0.6, marker="+", color="red",  s = 1)
        ax3.plot(all_ages[i], all_chi2_grid3[i], alpha=0.6, ls="--", color="red", linewidth=1)
        ax3.axhline(y=1, color="red", alpha=0.6)
        #ax3.legend()
        ax3.set_yscale("log")
        ax3.set_xlabel('Age', fontsize=5)
        ax3.set_ylabel(r'$\chi^2$', fontsize=5)
    #pdb.set_trace()

    for i in [4]:
        ax3.scatter(all_ages[i], all_chi2_grid3[i], alpha=0.6, marker="+", color="black",  s = 1)
        ax3.plot(all_ages[i], all_chi2_grid3[i], alpha=0.6, ls="--", color="black", linewidth=1)
        ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax3.legend(fontsize=5)
        ax3.set_yscale("log")
        ax3.set_xlabel('Age', fontsize=5)
        ax3.set_ylabel(r'$\chi^2$', fontsize=5)
        ax3.tick_params(axis='both', labelsize=5)
    #pdb.set_trace()
    for i in [1,7]:
        #ax4.set_title("Total Mag Model Fit, HD %s" % target_hd)
        #ax4.errorbar(all_xwave, unumpy.nominal_values(all_yplot[i]), unumpy.std_devs(all_yplot[i]), fmt='o', color='black')
        ax4.plot(all_xwave[i], all_TOTmag_model[i], '--', color='blue', linewidth=1)
        # ax1.set_xlabel('Wavelength (nm)')
        #ax4.set_ylabel('Total Mag', fontsize=5)
        #ax4.invert_yaxis()

    for i in [3,5]:
        #ax4.set_title("Total Mag Model Fit, HD %s" % target_hd)
        #ax4.errorbar(all_xwave, unumpy.nominal_values(all_yplot[i]), unumpy.std_devs(all_yplot[i]), fmt='o', color='black')
        ax4.plot(all_xwave[i], all_TOTmag_model[i], '--', color='red', linewidth=1)
        # ax1.set_xlabel('Wavelength (nm)')
        #ax4.set_ylabel('Total Mag', fontsize=5)
        #ax4.invert_yaxis()

    for i in [4]:
        ax4.set_title("Total Mag Model Fit, HD %s" % target_hd, fontsize=5)
        ax4.errorbar(all_xwave[i], unumpy.nominal_values(all_yplot[i]), unumpy.std_devs(all_yplot[i]), fmt='o', color='black', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax4.plot(all_xwave[i], all_TOTmag_model[i], '--', color='black', linewidth=1)
        # ax1.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Total Mag', fontsize=5)
        ax4.tick_params(axis='both', labelsize=5)
        ax4.invert_yaxis()


    for i in [1,7]:
        #ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd)
        ax5.plot(all_xwave[i], all_Dmag_model[i], '--', color='blue', linewidth=1)
        # ax2.set_xlabel('Wavelength (nm)')
        #ax5.invert_yaxis()
        #ax5.set_ylabel('Diff Mag')

    for i in [3,5]:
        #ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd)
        #ax5.errorbar(all_data_wave[i], unumpy.nominal_values(all_yplot1[i]), unumpy.std_devs(all_yplot1[i]), fmt='o', color='blue', elinewidth=1,
             #ecolor='black', capsize=1, capthick=1)
        ax5.plot(all_xwave[i], all_Dmag_model[i], '--', color='red', linewidth=1)
        # ax2.set_xlabel('Wavelength (nm)')
        #ax5.invert_yaxis()
        #ax5.set_ylabel('Diff Mag')

    for i in [4]:
        ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd, fontsize=5)
        ax5.errorbar(all_data_wave[i], unumpy.nominal_values(all_yplot1[i]), unumpy.std_devs(all_yplot1[i]), fmt='o', color='black', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax5.plot(all_xwave[i], all_Dmag_model[i], '--', color='black', linewidth=1)
        ax5.set_xlabel('Wavelength (nm)', fontsize=5)
        ax5.invert_yaxis()
        ax5.set_ylabel('Diff Mag', fontsize=5)
        ax5.tick_params(axis='both', labelsize=5)

    for i in [1,7]:
        ax6.set_title("Split SED Model Fit, HD %s" % target_hd, fontsize=5)
        ax6.errorbar(all_data_wave[i], unumpy.nominal_values(all_split_mag1[i]), unumpy.std_devs(all_split_mag1[i]), fmt='o', color='blue', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax6.errorbar(all_data_wave[i], unumpy.nominal_values(all_split_mag2[i]), unumpy.std_devs(all_split_mag2[i]), fmt='o', color='blue', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax6.plot(all_xwave[i], all_model1[i], '--', color='blue', linewidth=1)
        ax6.plot(all_xwave[i], all_model2[i], '--', color='blue', linewidth=1)
        #ax6.invert_yaxis()
        ax6.set_xlabel('Wavelength (nm)', fontsize=5)
        ax6.set_ylabel('Apparent Mag', fontsize=5)
    for i in [3,5]:
        ax6.set_title("Split SED Model Fit, HD %s" % target_hd, fontsize=5)
        ax6.errorbar(all_data_wave[i], unumpy.nominal_values(all_split_mag1[i]), unumpy.std_devs(all_split_mag1[i]), fmt='o', color='red', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax6.errorbar(all_data_wave[i], unumpy.nominal_values(all_split_mag2[i]), unumpy.std_devs(all_split_mag2[i]), fmt='o', color='red', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax6.plot(all_xwave[i], all_model1[i], '--', color='red', linewidth=1)
        ax6.plot(all_xwave[i], all_model2[i], '--', color='red', linewidth=1)
        #ax6.invert_yaxis()
        ax6.set_xlabel('Wavelength (nm)', fontsize=5)
        ax6.set_ylabel('Apparent Mag', fontsize=5)
    for i in [4]:
        ax6.set_title("Split SED Model Fit, HD %s" % target_hd, fontsize=5)
        ax6.errorbar(all_data_wave[i], unumpy.nominal_values(all_split_mag1[i]), unumpy.std_devs(all_split_mag1[i]), fmt='o', color='black', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax6.errorbar(all_data_wave[i], unumpy.nominal_values(all_split_mag2[i]), unumpy.std_devs(all_split_mag2[i]), fmt='o', color='black', ms=1, elinewidth=1,
             ecolor='black', capsize=1, capthick=1)
        ax6.plot(all_xwave[i], all_model1[i], '--', color='black', linewidth=1)
        ax6.plot(all_xwave[i], all_model2[i], '--', color='black', linewidth=1)
        ax6.invert_yaxis()
        ax6.set_xlabel('Wavelength (nm)', fontsize=5)
        ax6.set_ylabel('Apparent Mag', fontsize=5)
        ax6.tick_params(axis='both', labelsize=5)


    for i in [1,7]:
            ax1.plot(all_modelx_best[i], all_modely_best[i], label=f"Best log age = {np.around(all_age_best[i], 2)} ", color='blue', linewidth=1)
            #ax1.invert_yaxis()
            ax1.set_title("HD %s" % target_hd, fontsize=15)
            ax1.legend(fontsize=5)


    for i in [3,5]:
            ax1.plot(all_modelx_best[i], all_modely_best[i], label=f"Best log age = {np.around(all_age_best[i], 2)} ", color='red', linewidth=1)
            #ax1.invert_yaxis()
            ax1.set_title("HD %s" % target_hd, fontsize=5)
            ax1.legend(fontsize=5)
    for i in [4]:
            ax1.plot(all_modelx_best[i], all_modely_best[i], label=f"Best log age = {np.around(all_age_best[i], 2)} ", color='black', linewidth=1)
            ax1.invert_yaxis()
            ax1.set_title("HD %s" % target_hd, fontsize=5)
            ax1.legend(fontsize=5)
            ax1.tick_params(axis='both', labelsize=5)

    #pdb.set_trace()
    file = f'{orbit_directory}HD{target_hd}__outer_mcmc.pdf'
    pdf_file = fitz.open(file)
    for page in pdf_file:  # iterate through the pages
        pix = page.get_pixmap()  # render page to an image
        pix.save(f'{orbit_directory}HD{target_hd}.png')  # store image as a PNG

    pic = f'{orbit_directory}HD{target_hd}.png'

    img = io.imread(pic)
    ax7.imshow(img)
    ax7.tick_params(axis='both',top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
    fig.tight_layout(pad=0.5)






    fig.savefig("%s/HD_%s_%s_all_SED_fit.pdf" % (directory2, target_hd, note))
    #except:
            #Target_List_Fail.append(target_hd)
            #print('--'*10)
            #print('--'*10)
            #print("Target HD %s FAILED !!!!!!! Check this one. Continuing for now..."%target_hd)

print("Failed Targets = ")
print(Target_List_Fail)

#df = pd.read_csv(csv, header= None)
#df.to_csv('file.csv', header = Header)
