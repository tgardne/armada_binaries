import pdb
import shutil
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
import pandas as pd
from tqdm import tqdm
from isochrones.mist import MIST_EvolutionTrack, MIST_Isochrone
from isochrones import get_ichrone
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import corner
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from skimage import io
from pdf2image import convert_from_path, convert_from_bytes
import matplotlib
from pathlib import Path
#matplotlib.use('Agg')

Mist_iso = MIST_Isochrone()
Mist_evoTrack = MIST_EvolutionTrack()

matplotlib.rcParams['figure.figsize'] = (8, 5)

save_directory = '/Users/tgardner/ARMADA_isochrones/' ## path for saved files
summary_directory = '/Users/tgardner/ARMADA_isochrones/summary/' ## path for saved files
armada_file = '/Users/tgardner/armada_binaries/full_target_list.csv' ## path to csv target file
photometry_file = '/Users/tgardner/armada_binaries/Photometry.csv'
csv = '/Users/tgardner/ARMADA_isochrones/target_info_hip_all_sigma.csv'
orbit_directory = '/Users/tgardner/ARMADA_isochrones/ARMADA_orbits/'
corner_directory = '/Users/tgardner/ARMADA_isochrones/summary/corner_plots/'
wds_file = '/Users/tgardner/ARMADA_isochrones/WDS_Data/'

#summary_directory = '/home/colton/ARMADA_binaries/summary/' ## path for saved file
#save_directory = '/home/colton/ARMADA_binaries/' ## path for saved files
#corner_directory = '/home/colton/ARMADA_binaries/summary/corner_plots/' ## path for saved files
#photometry_file = '/home/colton/armada_binaries/Photometry.csv'
#armada_file = '/home/colton/armada_binaries/full_target_list_10_3.csv' ## path to csv target file
#orbit_directory = '/home/colton/ARMADA_binaries/ARMADA_orbits/'
#csv = '/home/colton/armada_binaries/target_info_all_sigma.csv'
#wds_file = '/home/colton/ARMADA_binaries/WDS_Data/'

Header =["HD", "M_Dyn_mcmc", "M_Dyn_mcmc_err", "M_Dyn_orbital", "M_Dyn_orbital_err",
                        "M_Tot", "M_Tot_err", "M1",
                        "M1_err", "M2", "M2_err",
                        "log_age", "log_age_err", "FeH", "Av",
                        "Redchi2", "Main Seq Only"]

df_armada = pd.read_csv(armada_file, dtype=object)
df_photometry = pd.read_csv(photometry_file, dtype=object)
## note for saved files (e.g. 'hip' for hipparcos distance, or 'gaia')

switch = input("Would you like to only use the main sequence ages?: ")
note_first = input("Choose note for saved files in this run: ")
if (switch == 'no') == True or (switch == 'No')== True:
    note = note_first+'_full_seq'
if (switch == 'yes') == True or (switch == 'Yes')== True:
    note = note_first+'_main_seq'
#pdb.set_trace()

Target_List = ['1976','2772', '5143', '6456','10453', '11031', '16753', '17094', '27176', '29316', '29573', '31093', '31297', '34319', '36058',  '37711', '38545'
    , '38769', '40932', '41040', '43358', '43525', '45542', '46273', '47105', '48581', '49643', '60107', '64235',
               '75974', '78316', '82446', '87652', '87822', '107259', '112846'
    , '114993', '118889', '127726', '128415', '129246', '133484', '133955', '137798', '140159', '140436', '144892',
              '145589', '148283', '153370', '154569', '156190', '158140'
    , '160935', '166045', '173093', '178475', '179950', '185404', '185762', '189037', '189340', '195206',
               '196089', '196867', '198183', '199766', '201038', '206901'
    , '217676', '217782', '220278', '224512']

Target_List = ['1976']

Target_List_Fail = []


## Fuction to fit a single star model
def single_star_fit(params,split_mag_star,wave_star,d_mod,Av):
    split_mag_absolute = split_mag_star - d_mod
    split_mag_val = unumpy.nominal_values(split_mag_absolute)
    split_mag_err = unumpy.std_devs(split_mag_absolute)

    age = params['age']
    mass = params['mass']
    feh = params['feh']

    a1 = tracks.generate(mass, age, feh, return_dict=True)

    split_mag_model_all = np.array([(a1['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                                (a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                                (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                                (a1['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                                (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                                (a1['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                                (a1['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                                (a1['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0]])
    ## UBVRIJHK
    wave_model = np.array([365, 445, 551, 675, 806, 1250, 1650, 2150])

    split_mag_model = np.interp(wave_star,wave_model,split_mag_model_all)
    diff = (split_mag_val - split_mag_model) / split_mag_err

    if np.isnan(a1['Mbol']):
        diff[:] = FAIL ## can do this more elegantly. This will just kill the minimization :)

    return(diff)

## Objective function to find the best distance
def best_distance(gaia, hip, kervella):
    distances = np.array([gaia,hip,kervella])
    idx_lowest = np.nanargmin(unumpy.std_devs(distances))
    #pdb.set_trace()
    return(distances[idx_lowest])

## Objective function for binaries to be minimized for lmfit
def isochrone_model(params, TOT_mag_star, D_mag_star, mag_wave_star, dmag_wave_star, d_mod, Av):
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

    if np.isnan(a1['Mbol']) or np.isnan(a2['Mbol']):
       #print('ALL NAN ENCOUNTERED. NEED TO FIX!!')
       FAIL ## kill the program

    mag1_model_all = np.array([(a1['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0]])

    mag2_model_all = np.array([(a2['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0]])
    ## UBVRIJHK
    wave_model = np.array([365, 445, 551, 675, 806, 1250, 1650, 2150])

    mag1_model = np.interp(dmag_wave_star,wave_model,mag1_model_all)
    mag2_model = np.interp(dmag_wave_star,wave_model,mag2_model_all)
    D_mag_model = mag2_model - mag1_model
    diff1 = (D_mag_val - D_mag_model) / D_mag_err

    mag1_model = np.interp(mag_wave_star,wave_model,mag1_model_all)
    mag2_model = np.interp(mag_wave_star,wave_model,mag2_model_all)
    TOT_mag_model = -2.5 * np.log10(10 ** (-0.4 * mag1_model) + 10 ** (-0.4 * mag2_model))
    diff2 = (TOT_mag_val - TOT_mag_model) / TOT_mag_err
    # print(np.concatenate([diff1,diff2]).size)

    return np.concatenate([diff1, diff2])

## Objective function for binaries to be minimized for lmfit
def isochrone_model_mcmc(params, TOT_mag_star, D_mag_star, mag_wave_star, dmag_wave_star, d_mod, Av):
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

    mag1_model_all = np.array([(a1['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0]])

    mag2_model_all = np.array([(a2['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0]])

    ## UBVRIJHK
    wave_model = np.array([365, 445, 551, 675, 806, 1250, 1650, 2150])

    mag1_model = np.interp(dmag_wave_star,wave_model,mag1_model_all)
    mag2_model = np.interp(dmag_wave_star,wave_model,mag2_model_all)
    D_mag_model = mag2_model - mag1_model
    diff1 = (D_mag_val - D_mag_model) / D_mag_err

    mag1_model = np.interp(mag_wave_star,wave_model,mag1_model_all)
    mag2_model = np.interp(mag_wave_star,wave_model,mag2_model_all)
    TOT_mag_model = -2.5 * np.log10(10 ** (-0.4 * mag1_model) + 10 ** (-0.4 * mag2_model))
    diff2 = (TOT_mag_val - TOT_mag_model) / TOT_mag_err
    # print(np.concatenate([diff1,diff2]).size)

    if np.isnan(a1['Mbol']) or np.isnan(a2['Mbol']):
       return np.inf
    else:
        return np.concatenate([diff1, diff2])

def mass_search(mass1_grid,mass2_grid,log_age_guess,split_mag1,split_mag2,dmag_wave_star,d_modulus,Av,feh):
        
    ## search for mass 1
    chi2_grid1 = []
    mass1_result = []
    for mm in mass1_grid:
        try:
            params = Parameters()
            params.add('age', value=log_age_guess, vary=False) ## NOTE: I could vary this!
            params.add('mass', value=mm, vary=False)
            params.add('feh', value=feh, vary=False)
            minner = Minimizer(single_star_fit, params, fcn_args=(split_mag1,dmag_wave_star,d_modulus.nominal_value,Av),
                            nan_policy='omit')
            result = minner.minimize()
            chi2_grid1.append(result.redchi)
            mass1_result.append(mm)
        except:
            # print("Fails at log age = %s"%aa)
            pass

    ## search for mass 2
    chi2_grid2 = []
    mass2_result = []
    for mm in mass2_grid:
        try:
            params = Parameters()
            params.add('age', value=log_age_guess, vary=False) ## NOTE: I could vary this!
            params.add('mass', value=mm, vary=False)
            params.add('feh', value=feh, vary=False)
            minner = Minimizer(single_star_fit, params, fcn_args=(split_mag2,dmag_wave_star,d_modulus.nominal_value,Av),
                            nan_policy='omit')
            result = minner.minimize()
            chi2_grid2.append(result.redchi)
            mass2_result.append(mm)
        except:
            # print("Fails at log age = %s"%aa)
            pass

    return chi2_grid1,mass1_result,chi2_grid2,mass2_result

def find_max_mass(age,m1_grid,feh,TOT_Mag,DiffM,mag_wave_star,dmag_wave_star,d_modulus,Av):
    m1_max = np.nanmax(m1_grid)
    for m1 in m1_grid:
        try:
            params = Parameters()
            params.add('age', value=age, vary=False)
            params.add('mass1', value=m1, vary=False)
            params.add('mass2', value=0.5, vary=False)
            params.add('feh', value=feh, vary=False)  # min=-0.5, max=0.5)
            minner = Minimizer(isochrone_model, params, fcn_args=(TOT_Mag, DiffM, mag_wave_star, dmag_wave_star, d_modulus.nominal_value, Av),
                            nan_policy='omit')
            result = minner.minimize()
        except:
            m1_max = m1
            break
    return (m1_max)

feh_set = [-0.1,0,0.1]

for target_hd in Target_List:
    print('--' * 10)
    print('--' * 10)
    print("Doing Target HD %s" % target_hd)

    df_wds = pd.read_csv("%s/HD_%s_wds.csv"%(wds_file,target_hd))

    # For Combined 3x3 plots
    all_mass1_result = []
    all_chi2_grid = []
    all_mass2_result = []
    all_chi2_grid2 = []
    all_ages = []
    all_chi2_grid3 = []
    all_mass = []
    all_TOTmag_model_wave = []
    all_TOTmag_wave = []
    all_TOTmag_model = []
    all_TOTmag = []
    all_split_mag1 = []
    all_split_mag2 = []
    all_model1 = []
    all_model2 = []
    all_Dmag_model = []
    all_Dmag_model_wave = []
    all_Dmag = []
    all_Dmag_wave = []
    all_modelx_best = []
    all_modely_best = []
    all_age_best = []
    all_m_dyn = []
    all_m_phot = []
    all_xval1 = []
    all_xval2 = []
    all_yval1 = []
    all_yval2 = []
    all_m_dyn_orbit_nom =[]
    all_m_dyn_orbit_err =[]
    all_m_phot_err = []
    all_m_phot_nom = []
    all_m_dyn_mcmc_nom =[]
    all_m_dyn_mcmc_err =[]
    all_m_dyn_mcmc = []
    all_m_dyn_orbit = []

    ## Create directory for saved files, if it doesn't already exist
    directory_corner = "%s/HD_%s/" % (corner_directory, target_hd)
    if os.path.exists(directory_corner):
         shutil.rmtree(directory_corner)
         print("Removing corner directory")
    if not os.path.exists(directory_corner):
        print("Creating corner directory")
        os.makedirs(directory_corner)

    for feh in feh_set:
        #try:
            print('--' * 10)
            print('--' * 10)
            print("Doing Metallicity [Fe/H] = %s" % feh)

            ## Create directory for saved files, if it doesn't already exist
            directory = "%s/HD_%s" % (save_directory, target_hd)
            if not os.path.exists(directory):
                print("Creating directory")
                os.makedirs(directory)
            ## Create directory for saved files, if it doesn't already exist
            directory2 = summary_directory
            if not os.path.exists(directory2):
                print("Creating directory")
                os.makedirs(directory2)

            ## Get magnitude differences for target
            ## from WDS measurements:
            cdiff_wds = unumpy.uarray(df_wds['f_ratio'].values,df_wds['f_ratio_err'].values)
            fratio_wds = 10 ** (cdiff_wds / 2.5)
            fratio_wds_wl = df_wds['Wavelength'].values

            ## Get target from spreadsheet
            idx = np.where(df_armada['HD'] == target_hd)[0][0]
            Av = float(df_armada['Av'][idx])

            cdiff_h = ufloat(float(df_armada['dmag_h'][idx]), float(df_armada['dmag_h_err'][idx]) )
            cdiff_k = ufloat(float(df_armada['dmag_k'][idx]),float(df_armada['dmag_k_err'][idx]))
            #cdiff_h = ufloat(float(np.nan), float(np.nan))
            #cdiff_k = ufloat(float(np.nan), float(np.nan))
            cdiff_i = ufloat(float(df_armada['dmag_speckle_i'][idx]), float(df_armada['dmag_speckle_i_err'][idx]))
            cdiff_b = ufloat(float(df_armada['dmag_speckle_v'][idx]),float(df_armada['dmag_speckle_v_err'][idx]))
            #cdiff_wds = ufloat(float(df_armada['dmag_wds_v'][idx]), float(df_armada['dmag_wds_v_err'][idx]))
            ##pdb.set_trace()
            fratio_h = 10 ** (cdiff_h / 2.5)
            fratio_k = 10 ** (cdiff_k / 2.5)
            fratio_i = 10 ** (cdiff_i / 2.5)
            fratio_b = 10 ** (cdiff_b / 2.5)
            
            fratio_data = np.array([fratio_b,fratio_i,fratio_h,fratio_k])
            cdiff_data = np.array([cdiff_b,cdiff_i,cdiff_h,cdiff_k])
            cdiff_data_fit = np.array([cdiff_b,cdiff_i,np.nan,np.nan]) ## NOTE: I am setting H and K bands to nan for fitting!!!!
            fratio_data_wl = np.array([562,832,1650,2150])
            
            fratio_all = np.concatenate([fratio_wds,fratio_data_wl])
            cdiff_all = np.concatenate([cdiff_wds,cdiff_data])
            fratio_all_wl = np.concatenate([fratio_wds_wl,fratio_data_wl])

            ## get total magnitudes and errors from photometry file. Set minimum error 
            err_min = 0.02        
            utot = ufloat(np.nan, np.nan)
            if np.isnan(float(df_photometry['B_err_complete'][idx])) or float(df_photometry['B_err_complete'][idx]) < err_min:
                btot = ufloat(float(df_photometry['B_complete'][idx]), err_min)
            else:
                btot = ufloat(float(df_photometry['B_complete'][idx]), float(df_photometry['B_err_complete'][idx]))
            if np.isnan(float(df_photometry['V_err_complete'][idx])) or float(df_photometry['V_err_complete'][idx]) < err_min:
                vtot = ufloat(float(df_photometry['V_complete'][idx]), err_min)
            else:
                vtot = ufloat(float(df_photometry['V_complete'][idx]), float(df_photometry['V_err_complete'][idx]))
            rtot = ufloat(float(df_photometry['R2_I/284'][idx]), 0.15)
            gtot = ufloat(np.nan, np.nan)
            itot = ufloat(np.nan, np.nan)
            if np.isnan(float(df_photometry['J_err'][idx])) or float(df_photometry['J_err'][idx]) < err_min:
                jtot = ufloat(float(df_photometry['J_ II/246/out'][idx]), err_min)
            else:
                jtot = ufloat(float(df_photometry['J_ II/246/out'][idx]), float(df_photometry['J_err'][idx]))
            if np.isnan(float(df_photometry['H_err'][idx])) or float(df_photometry['H_err'][idx]) < err_min:
                htot = ufloat(float(df_photometry['H_ II/246/out'][idx]), err_min)
            else:
                htot = ufloat(float(df_photometry['H_ II/246/out'][idx]), float(df_photometry['H_err'][idx]))
            if np.isnan(float(df_photometry['K_err'][idx])) or float(df_photometry['K_err'][idx]) < err_min:
                ktot = ufloat(float(df_photometry['K_ II/246/out'][idx]), err_min)
            else:
                ktot = ufloat(float(df_photometry['K_ II/246/out'][idx]), float(df_photometry['K_err'][idx]))

            ## Bessel_U, Bessel_B, Bessel_V, Johnson_R, Bessel_I, 2MASS_J, 2MASS_H, 2MASS_K
            ## Choose observables for fitting
            TOT_Mag = np.array([utot, btot, vtot, rtot, itot, jtot, htot, ktot])
            TOT_Mag_wl = np.array([365, 445, 551, 675, 806, 1250, 1650, 2150])

            ## Compute individual magnitudes at wavelengths with measure fratios
            tot_mag_ratios_val = np.interp(fratio_all_wl,TOT_Mag_wl,unumpy.nominal_values(TOT_Mag))
            tot_mag_ratios_err = np.interp(fratio_all_wl,TOT_Mag_wl,unumpy.std_devs(TOT_Mag))
            tot_mag_ratios = unumpy.uarray(tot_mag_ratios_val,tot_mag_ratios_err)
            split_mag1 = -2.5 * unumpy.log10(10 ** (-tot_mag_ratios / 2.5) / (1 + 10 ** (-cdiff_all / 2.5)))
            split_mag2 = cdiff_all + split_mag1

            ## A new isochrone model that interpolate the model photometry to add more constraint
            tracks = get_ichrone('mist', tracks=True, accurate=True)
            bc_grid_U = MISTBolometricCorrectionGrid(['U'])
            bc_grid_B = MISTBolometricCorrectionGrid(['B'])
            bc_grid_V = MISTBolometricCorrectionGrid(['V'])
            bc_grid_R = MISTBolometricCorrectionGrid(['R'])
            bc_grid_I = MISTBolometricCorrectionGrid(['I'])
            bc_grid_J = MISTBolometricCorrectionGrid(['J'])
            bc_grid_H = MISTBolometricCorrectionGrid(['H'])
            bc_grid_K = MISTBolometricCorrectionGrid(['K'])
            TOT_Mag_model_wl = np.array([365, 445, 551, 675, 806, 1250, 1650, 2150])

            ## choose which distance to use -- Gaia, HIP, or something else
            distance_gaia = ufloat(float(df_armada['Gaia_distance (pc)'][idx]), float(df_armada['Gaia_distance_err (pc)'][idx]))
            distance_hip = ufloat(float(df_armada['HIP_distance (pc)'][idx]), float(df_armada['HIP_distance_err (pc)'][idx]))
            distance_kervella = ufloat(float(df_armada['kerv_dist'][idx]), float(df_armada['e_kerv'][idx]))
            distance_best = best_distance(distance_gaia, distance_hip, distance_kervella)
            distance_low = distance_best - distance_best.std_dev
            distance_high = distance_best + distance_best.std_dev

            distance_set =[distance_low,distance_best,distance_high]

            ## We only want to use this one below if we only want to look at one best distance!
            #distance_set = [distance_best]
            #distance_set = [distance_hip]

            for distance in distance_set:

                fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))
                fig.tight_layout(pad=5)

                print("Distance = ", distance, 'pc')
                d_modulus = 5 * log10(distance) - 5
                name = f"{note}_{feh}_{distance.nominal_value}"

                ##################
                ## Now let's find best masses and age
                ##################
                mass1_grid = np.linspace(0.5,5,50)
                mass2_grid = np.linspace(0.5,5,50)
                age_grid = np.linspace(6, 10, 100)  ## do fewer steps to go faster

                print('Grid Searching over AGE to find best fit')
                #pdb.set_trace()
                ## Explore a grid of chi2 over age -- this paramter does not fit properly in least squares
                chi2_grid = []
                ages = []
                for aa in tqdm(age_grid):
                    try:
                        chi2_grid1,mass1_result,chi2_grid2,mass2_result = mass_search(mass1_grid,mass2_grid,aa,split_mag1[:-2],split_mag2[:-2],fratio_all_wl[:-2],d_modulus,Av,feh)
                        idx_mass1 = np.argmin(chi2_grid1)
                        mass1_guess = mass1_result[idx_mass1]
                        idx_mass2 = np.argmin(chi2_grid2)
                        mass2_guess = mass2_result[idx_mass2]

                        max_mass = find_max_mass(aa,mass1_grid,feh,TOT_Mag,cdiff_all[:-2],TOT_Mag_model_wl,fratio_all_wl[:-2],d_modulus,Av)

                        params = Parameters()
                        params.add('age', value=aa, vary=False)
                        params.add('mass1', value=mass1_guess, min=0, max=max_mass)
                        params.add('mass2', value=mass2_guess, min=0, max=max_mass)
                        params.add('feh', value=feh, vary=False)  # min=-0.5, max=0.5)
                        minner = Minimizer(isochrone_model, params, fcn_args=(TOT_Mag, cdiff_all[:-2], TOT_Mag_model_wl, fratio_all_wl[:-2], d_modulus.nominal_value, Av),
                                        nan_policy='omit')
                        result = minner.minimize()

                        chi2_grid.append(result.redchi)
                        ages.append(result.params['age'].value)
                    except:
                        # print("Fails at log age = %s"%aa)
                        pass

                ## Get best age
                ## Note: We only want to fit for the Main Sequence (Ages larger than 10 mill years or log 8)
                ages_array = np.array(ages)
                chi2_grid_array = np.array(chi2_grid)
                ages_10mil = ages_array[ ages_array > 8]
                idx_age_10mil = np.argmin(chi2_grid_array[ages_array > 8])
                best_age_ms = ages_10mil[idx_age_10mil]
                #pdb.set_trace()
                ##This is for minimum for entire age grid (Not being used in the fitting or plots)
                idx_age = np.argmin(chi2_grid)
                all_chi2_grid3.append(chi2_grid)
                all_ages.append(ages)
                best_age_total = ages[idx_age]
                age_max = ages[-1]
                #print('Fit fails at log Age = %s' % age_max)

                ## choose which age to use as best
                #age_best = best_age_ms
                #age_best = best_age_total
                #pdb.set_trace()
                if (switch == 'yes') == True or switch == 'Yes' == True:
                    age_best = best_age_ms
                elif (switch == 'no') == True or (switch == 'No') == True:
                    age_best = best_age_total

                ## Make chi2 plot of masses at best age
                chi2_grid1,mass1_result,chi2_grid2,mass2_result = mass_search(mass1_grid,mass2_grid,age_best,split_mag1[:-2],split_mag2[:-2],fratio_all_wl[:-2],d_modulus,Av,feh)

                idx_mass1 = np.argmin(chi2_grid1)
                mass1_best = mass1_result[idx_mass1]
                all_mass1_result.append(mass1_result)
                all_chi2_grid.append(chi2_grid1)

                idx_mass2 = np.argmin(chi2_grid2)
                all_mass2_result.append(mass2_result)
                all_chi2_grid2.append(chi2_grid2)
                mass2_best = mass2_result[idx_mass2]

                ## Plot Mass 1 and Mass 2 chi2 grids
                ax2.scatter(mass1_result, chi2_grid1, alpha=0.6, marker=".", color="Blue", label ='Mass 1')
                ax2.scatter(mass2_result, chi2_grid2, alpha=0.6, marker="+", color="Red", label ='Mass 2')
                ax2.plot(mass1_result, chi2_grid1, alpha=0.6, ls="--", color="black")
                ax2.plot(mass2_result, chi2_grid2, alpha=0.6, ls="--", color="black")
                ax2.legend()
                ax2.set_yscale("log")
                ax2.set_xlabel('Mass (solar)', fontsize=15)
                ax2.set_ylabel(r'$\chi^2$', fontsize=15)
                ax2.set_title('M1 = %s, M2 = %s, log age = %s'%(np.around(mass1_best,2),np.around(mass2_best,2),np.around(age_best,2)))

                ## Plot age chi2 grid
                ax3.scatter(ages, chi2_grid, alpha=0.6, marker="+", color="blue")
                ax3.plot(ages, chi2_grid, alpha=0.6, ls="--", color="black")
                ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
                ax3.legend()
                ax3.set_yscale("log")
                ax3.set_xlabel('Age', fontsize=15)
                ax3.set_ylabel(r'$\chi^2$', fontsize=15)

                ## Do one more least squares fit to minimize all parameters
                max_mass = find_max_mass(age_best,mass1_grid,feh,TOT_Mag,cdiff_all[:-2],TOT_Mag_model_wl,fratio_all_wl[:-2],d_modulus,Av)

                params = Parameters()
                params.add('age', value=age_best, min=6, max=age_max)
                params.add('mass1', value=mass1_best, min=0, max=max_mass)
                params.add('mass2', value=mass2_best, min=0, max=max_mass)
                params.add('feh', value=feh, vary=False)  # min=-0.5, max=0.5)
                minner = Minimizer(isochrone_model, params, fcn_args=(TOT_Mag, cdiff_all[:-2], TOT_Mag_model_wl, fratio_all_wl[:-2], d_modulus.nominal_value, Av),
                                   nan_policy='omit')
                result = minner.minimize()
                #report_fit(result)

                ## We probably want least squares result as "best" parameters. TBD
                age_best = result.params['age'].value
                mass1_best = result.params['mass1'].value
                mass2_best = result.params['mass2'].value
                feh_best = result.params['feh'].value
                redchi2_best = result.redchi

                print("Best log Age = %s" % age_best)
                print("Best M1 = %s" % mass1_best)
                print("Best M2 = %s" % mass2_best)

                ########################
                ########################
                ## Setup MCMC fit
                emcee_params = result.params.copy()
                nwalkers = 2 * len(emcee_params)
                steps = 300
                burn = 10
                thin = 1

                print("Running MCMC chains: ")
                ## Do MCMC fit (this cell could take some time, depending on steps)
                minner = Minimizer(isochrone_model_mcmc, emcee_params, fcn_args=(TOT_Mag, cdiff_all[:-2], TOT_Mag_model_wl, fratio_all_wl[:-2], d_modulus.nominal_value, Av),
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
                    #plt.savefig('%s/HD_%s_%s_corner.pdf' % (directory2, target_hd, name))
                    plt.savefig('%s/HD_%s/_%s_corner.png' % (corner_directory, target_hd, name))
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
                     (a1_best['Mbol'] - bc_grid_H.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a1_best['Mbol'] - bc_grid_K.interp([a1_best['Teff'], a1_best['logg'], feh, Av]) + d_modulus.nominal_value)[0]])

                model2 = np.array(
                    [(a2_best['Mbol'] - bc_grid_U.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a2_best['Mbol'] - bc_grid_B.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a2_best['Mbol'] - bc_grid_V.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a2_best['Mbol'] - bc_grid_R.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a2_best['Mbol'] - bc_grid_I.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a2_best['Mbol'] - bc_grid_J.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a2_best['Mbol'] - bc_grid_H.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0],
                     (a2_best['Mbol'] - bc_grid_K.interp([a2_best['Teff'], a2_best['logg'], feh, Av]) + d_modulus.nominal_value)[0]])

                D_mag_model = model2 - model1
                TOT_mag_model = -2.5 * np.log10(10 ** (-0.4 * model1) + 10 ** (-0.4 * model2))

                #fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
                #fig.tight_layout()
                all_TOTmag_wave.append(TOT_Mag_wl)
                all_TOTmag.append(TOT_Mag)
                all_TOTmag_model_wave.append(TOT_Mag_model_wl)
                all_TOTmag_model.append(TOT_mag_model)
                #pdb.set_trace()
                ax4.set_title("Total Mag Model Fit, HD %s" % target_hd)
                ax4.errorbar(TOT_Mag_wl, unumpy.nominal_values(TOT_Mag), unumpy.std_devs(TOT_Mag), fmt='o', color='black')
                ax4.plot(TOT_Mag_model_wl, TOT_mag_model, '--', color='red')
                # ax1.set_xlabel('Wavelength (nm)')
                ax4.set_ylabel('Total Mag')
                ax4.invert_yaxis()

                # ax1.gca().invert_yaxis()
                # plt.savefig("%s/HD_%s_%s_TOTmag_fit.pdf"%(directory,target_hd,note))

                all_Dmag_wave.append(fratio_all_wl)
                all_Dmag.append(cdiff_all)
                all_Dmag_model.append(D_mag_model)
                all_Dmag_model_wave.append(TOT_Mag_model_wl)

                ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd)
                ax5.errorbar(fratio_all_wl[:-2], unumpy.nominal_values(cdiff_all[:-2]), unumpy.std_devs(cdiff_all[:-2]), fmt='o', color='black')
                ax5.errorbar(fratio_all_wl[-2:], unumpy.nominal_values(cdiff_all[-2:]), unumpy.std_devs(cdiff_all[-2:]), fmt='o', color='lightgrey')
                ax5.plot(TOT_Mag_model_wl, D_mag_model, '--', color='red')
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
                ax6.errorbar(fratio_all_wl[:-2], unumpy.nominal_values(split_mag1[:-2]), unumpy.std_devs(split_mag1[:-2]), fmt='o', color='black')
                ax6.errorbar(fratio_all_wl[-2:], unumpy.nominal_values(split_mag1[-2:]), unumpy.std_devs(split_mag1[-2:]), fmt='o', color='lightgrey')
                ax6.errorbar(fratio_all_wl[:-2], unumpy.nominal_values(split_mag2[:-2]), unumpy.std_devs(split_mag2[:-2]), fmt='o', color='grey')
                ax6.errorbar(fratio_all_wl[-2:], unumpy.nominal_values(split_mag2[-2:]), unumpy.std_devs(split_mag2[-2:]), fmt='o', color='lightgrey')
                ax6.plot(TOT_Mag_model_wl, model1, '--', color='red')
                ax6.plot(TOT_Mag_model_wl, model2, '--', color='red')
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

                mass_start = mass2_best - 0.5
                mass_size = 0.5  ## step size
                mass_steps = 5  ## number of steps

                paramList = [np.array([log_age_start, feh]) + np.array([log_age_size, 0]) * i for i in
                             range(0, log_age_steps)]
                isoList = [Mist_iso.isochrone(param[0], param[1]) for param in paramList]
                isoList_best = [Mist_iso.isochrone(age_best, feh)]

                paramList_mass = [np.array([mass_start, feh]) + np.array([mass_size, 0]) * i for i in
                             range(0, mass_steps)]

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

                ## Choose x/y axis for isochrone plot. For example, V-H vs V-K
                if np.isnan(split_mag1[-2].nominal_value) == True:
                    xval1 = split_mag1[-4] - split_mag1[-1]  ## component 1
                    yval1 = split_mag1[-4] - d_modulus
                    xval2 = split_mag2[-4] - split_mag2[-1]  ## component 2
                    yval2 = split_mag2[-4] - d_modulus
                    xlabel = "V - K"
                    ylabel = "V"
                elif np.isnan(split_mag1[-2].nominal_value) == False:
                    xval1 = split_mag1[-4] - split_mag1[-2]  ## component 1
                    yval1 = split_mag1[-4] - d_modulus
                    xval2 = split_mag2[-4] - split_mag2[-2]  ## component 2
                    yval2 = split_mag2[-4] - d_modulus
                    xlabel = "V - H"
                    ylabel = "V"

                all_xval1.append(xval1)
                all_xval2.append(xval2)
                all_yval1.append(yval1)
                all_yval2.append(yval2)

                iso_start = 50
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
                #pdb.set_trace()
                fig.savefig("%s/HD_%s_%s_all_SED_fit.pdf" % (directory, target_hd, name))

                #Finding Dynamical Mass
                Mdyn_over_d3 = float(df_armada['Mdyn_over_d3 (x10e-6)'][idx])
                Mdyn_over_d3_err = float(df_armada['Mdyn_over_d3_err (x10e-6)'][idx])
                mdyn_over_d3_float = ufloat(Mdyn_over_d3, Mdyn_over_d3_err) * 10 ** (-6)
                mdyn_mcmc = mdyn_over_d3_float * (distance.nominal_value ** 3)

                #pdb.set_trace()
                a_mas = ufloat(float(df_armada['a (mas)'][idx]),float(df_armada['a_err (mas)'][idx]))
                a_mas = float(df_armada['a (mas)'][idx])
                a_au = (a_mas/1000)*distance.nominal_value
                p_yr = ufloat(float(df_armada['P (yr)'][idx]),float(df_armada['P_err (yr)'][idx]))
                mdyn_orbit = a_au**3/p_yr**2

                #pdb.set_trace()

                #Save all of the desired Data (M_phot, M_dyn, Age, Feh, Av)
                m_tot= mass1+mass2
                all_m_phot.append(m_tot)
                all_m_dyn_mcmc.append(mdyn_mcmc)
                all_m_dyn_orbit.append(mdyn_orbit)
                df_new = pd.DataFrame(dict(HD=[target_hd], M_Dyn_mcmc=[mdyn_mcmc.nominal_value], M_Dyn_mcmc_err=[mdyn_mcmc.std_dev], M_Dyn_orbital=[mdyn_orbit.nominal_value], M_Dyn_orbital_err=[mdyn_orbit.std_dev],
                                      M_Tot=[m_tot.nominal_value], M_Tot_err=[m_tot.std_dev],M1=[mass1.nominal_value],
                                     M1_err=[mass1.std_dev],M2=[mass2.nominal_value],M2_err=[mass2.std_dev],
                                    log_age=[age.nominal_value], log_age_err=[age.std_dev],FeH=[feh_best], Av=[Av],
                                   Redchi2=[redchi2_best], main_seq_only = [switch]))
                file_name = f"{note}"
                #print(df_new)
                df_new.to_csv('%s/HD_%s/target_info_%s.csv' % (save_directory, target_hd,file_name), mode='a', index=False, header=False)

            print('Going to New Target')
        #except:
            #print('Target HD%s Failed'%target_hd)
            #Target_List_Fail.append(target_hd)

    df = pd.read_csv('%s/HD_%s/target_info_%s.csv'%(save_directory,target_hd,file_name), header=None, index_col= None)
    df.to_csv('%s/HD_%s/target_info_%s.csv'%(save_directory,target_hd,file_name), header = Header, index= False)
    df2 = pd.read_csv("%s/HD_%s/target_info_%s.csv"%(save_directory,target_hd,file_name))
    #print(df2)

    fig = plt.figure(figsize=(55.0,42.5), constrained_layout=False)

    gs1 = GridSpec(1, 1)
    gs1.update(left=0.05, right=0.35, bottom=0.5,top=0.95, hspace=0)
    ax1 = fig.add_subplot(gs1[0])  # first row, first col
    gs2 = GridSpec(1, 2)
    gs2.update(left=0.38, right=0.97, bottom=0.55, wspace=0,top=0.95, hspace=0)
    ax2 = fig.add_subplot(gs2[0])  # first row, sec col
    ax3 = fig.add_subplot(gs2[1], sharey = ax2)  # first row, third col
    gs3 = GridSpec(3, 1)
    gs3.update(bottom=0.05, left=0.05, right=0.35, top=0.45, wspace=0, hspace=0)
    ax4 = fig.add_subplot(gs3[0], sharex = ax6)  # sec.1 row, first col
    ax5 = fig.add_subplot(gs3[1], sharex = ax6)  # sec.2 row, first col
    ax6 = fig.add_subplot(gs3[2])  # sec.3  row, first col
    gs4 = GridSpec(1, 1)
    gs4.update(left=0.38, right=0.66, bottom=0.075, top=0.45)  # 0.53 aligns with sides
    ax7 = fig.add_subplot(gs4[0])  # sec row, sec col
    gs5 = GridSpec(1, 1)
    gs5.update(bottom=0.05, left=0.71, right=0.97, top=0.45, wspace=0, hspace=0)
    ax8 = fig.add_subplot(gs5[0])  # sec row. third col

    #pdb.set_trace()
    for i in [1,7]:
        ax2.scatter(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, marker="+", color="blue", s=5)
        ax2.plot(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, ls="--", color="blue", linewidth =5)
        #ax2.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax2.set_yscale("log")
        ax2.scatter(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, marker="+", color="Red", s=5)
        ax2.plot(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, ls="--", color="blue", linewidth =5)
        # ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        #ax2.legend()
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=30)
        ax2.set_ylabel(r'$\chi^2$', fontsize=30)
        ax2.set_title('Mass 1 & 2 Guess', fontsize=35)
        #pdb.set_trace()

    for i in [3,5]:
        ax2.scatter(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, marker="+", color="blue",s = 5)
        ax2.plot(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, ls="--", color="red", linewidth =5)
        #ax2.axhline(y=1, color="red", alpha=0.6)
        ax2.set_yscale("log")
        ax2.scatter(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, marker="+", color="Red",s = 5)
        ax2.plot(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, ls="--", color="red", linewidth =5)
        #ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        #ax2.legend()
        ax2.set_yscale("log")
        ax2.set_xlabel('Mass (solar)', fontsize=30)
        ax2.set_ylabel('$\chi^2$', fontsize=30)
        ax2.set_title('Mass 1 & 2 Guess', fontsize=35)
        #pdb.set_trace()

    for i in [4]:
        idx_mass1 = np.array(all_chi2_grid[i]).argmin()
        mass1_best = all_mass1_result[i][idx_mass1]
        idx_mass2 = np.array(all_chi2_grid2[i]).argmin()
        mass2_best = all_mass2_result[i][idx_mass2]
        idx_age1 = np.array(all_chi2_grid3[i]).argmin()
        age1_best = all_ages[i][idx_age1]

        ax2.scatter(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, marker="+", color="blue", label=f'Mass 1 ={mass1_best:.2f}', s = 5)
        ax2.plot(all_mass1_result[i], all_chi2_grid[i], alpha=0.6, ls="--", color="black", linewidth=5)
        ax2.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax2.set_yscale("log")
        ax2.scatter(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, marker="+", color="Red", label=f'Mass 2 ={mass2_best:.2f}',  s = 5)
        ax2.plot(all_mass2_result[i], all_chi2_grid2[i], alpha=0.6, ls="--", color="black", linewidth = 5)
        # ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax2.legend(fontsize= 25)
        ax2.set_yscale("log")
        ax2.set_yscale("log")
        ax2.grid()
        ax2.set_xlabel('Mass (solar)', fontsize=35)
        ax2.set_ylabel('$\chi^2$', fontsize=35)
        ax2.set_title('Mass 1 & 2 Guess, log age = %s'%np.around(age1_best,2), fontsize=40)
        ax2.tick_params(axis='both', labelsize=30)
        ax2.set_aspect('auto')

        #pdb.set_trace()

    for i in [1,7]:
        ax3.scatter(all_ages[i], all_chi2_grid3[i], alpha=0.6, marker="+", color="blue",  s = 5)
        ax3.plot(all_ages[i], all_chi2_grid3[i], alpha=0.6, ls="--", color="blue", linewidth=5)
        ax3.axhline(y=1, color="red", alpha=0.6)
        #ax3.legend()
        ax3.set_yscale("log")
        ax3.set_xlabel('Age', fontsize=30)
        #ax3.set_ylabel(r'$\chi^2$', fontsize=6)
        ax3.set_aspect('equal')

    for i in [7]:
        ax3.scatter(all_ages[i], all_chi2_grid3[i], alpha=0.6, marker="+", color="green",  s = 5)
        ax3.plot(all_ages[i], all_chi2_grid3[i], alpha=0.6, ls="--", color="green", linewidth=5)
        ax3.axhline(y=1, color="red", alpha=0.6)
        #ax3.legend()
        ax3.set_yscale("log")
        ax3.set_xlabel('Age', fontsize=30)
        #ax3.set_ylabel(r'$\chi^2$', fontsize=6)
        ax3.set_aspect('equal')

    #pdb.set_trace()

    for i in [3,5]:

        ax3.scatter(all_ages[i], all_chi2_grid3[i], alpha=0.6, marker="+", color="red",  s = 5)
        ax3.plot(all_ages[i], all_chi2_grid3[i], alpha=0.6, ls="--", color="red", linewidth=5)
        ax3.axhline(y=1, color="red", alpha=0.6)
        #ax3.legend()
        ax3.set_yscale("log")
        ax3.set_xlabel('Age', fontsize=35)
        #ax3.set_ylabel(r'$\chi^2$', fontsize=6)
        ax3.set_aspect('equal')

    #pdb.set_trace()

    for i in [4]:
        idx_age1 = np.array(all_chi2_grid3[i]).argmin()
        age1_best = all_ages[i][idx_age1]
        ax3.scatter(all_ages[i], all_chi2_grid3[i], alpha=0.6, marker="+", color="black", label=f'Best Age ={age1_best:.2f}' ,  s = 5)
        ax3.plot(all_ages[i], all_chi2_grid3[i], alpha=0.6, ls="--", color="black", linewidth=5)
        ax3.axhline(y=1, color="red", alpha=0.6, label=r"$\chi^2=1$")
        ax3.legend(fontsize=35)
        ax3.set_yscale("log")
        ax3.set_xlabel('Age', fontsize=35)
       # ax3.set_ylabel(r'$\chi^2$', fontsize=7)
        ax3.tick_params(axis='both', labelsize=30)
        ax3.set_title("Chi2 vs Age, HD %s" % target_hd, fontsize=40)
        ax3.set_aspect('auto')
        ax3.tick_params('y', labelleft=False)

    #pdb.set_trace()
    for i in [1]:
        #ax4.set_title("Total Mag Model Fit, HD %s" % target_hd)
        #ax4.errorbar(all_xwave, unumpy.nominal_values(all_yplot[i]), unumpy.std_devs(all_yplot[i]), fmt='o', color='black')
        ax4.plot(all_TOTmag_model_wave[i], all_TOTmag_model[i], '--', color='blue', linewidth=5)
        # ax1.set_xlabel('Wavelength (nm)')
        #ax4.set_ylabel('Total Mag', fontsize=5)
        #ax4.invert_yaxis()

    for i in [7]:
        #ax4.set_title("Total Mag Model Fit, HD %s" % target_hd)
        #ax4.errorbar(all_xwave, unumpy.nominal_values(all_yplot[i]), unumpy.std_devs(all_yplot[i]), fmt='o', color='black')
        ax4.plot(all_TOTmag_model_wave[i], all_TOTmag_model[i], '--', color='green', linewidth=5)
        # ax1.set_xlabel('Wavelength (nm)')
        #ax4.set_ylabel('Total Mag', fontsize=5)
        #ax4.invert_yaxis()

    for i in [3,5]:
        #ax4.set_title("Total Mag Model Fit, HD %s" % target_hd)
        #ax4.errorbar(all_xwave, unumpy.nominal_values(all_yplot[i]), unumpy.std_devs(all_yplot[i]), fmt='o', color='black')
        ax4.plot(all_TOTmag_model_wave[i], all_TOTmag_model[i], '--', color='red', linewidth=5)
        # ax1.set_xlabel('Wavelength (nm)')
        #ax4.set_ylabel('Total Mag', fontsize=5)
        #ax4.invert_yaxis()

    for i in [4]:
        #ax4.set_title("Total Mag Model Fit, HD %s" % target_hd, fontsize=8)
        ax4.errorbar(all_TOTmag_wave[i], unumpy.nominal_values(all_TOTmag[i]), unumpy.std_devs(all_TOTmag[i]), fmt='o', color='black', ms=1, elinewidth=5,
             ecolor='black', capsize=5, capthick=5)
        ax4.plot(all_TOTmag_model_wave[i], all_TOTmag_model[i], '--', color='black', linewidth=5)
        # ax1.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Total Mag', fontsize=35)
        ax4.tick_params(axis='both', labelsize=30)
        ax4.set_aspect('auto')
        ax4.invert_yaxis()
        ax4.tick_params('x', labelbottom=False)



    for i in [1]:
        #ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd)
        ax5.plot(all_Dmag_model_wave[i], all_Dmag_model[i], '--', color='blue', linewidth=5)
        # ax2.set_xlabel('Wavelength (nm)')
        #ax5.invert_yaxis()
        #ax5.set_ylabel('Diff Mag')

    for i in [7]:
        #ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd)
        ax5.plot(all_Dmag_model_wave[i], all_Dmag_model[i], '--', color='green', linewidth=5)
        # ax2.set_xlabel('Wavelength (nm)')
        #ax5.invert_yaxis()
        #ax5.set_ylabel('Diff Mag')

    for i in [3,5]:
        #ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd)
        #ax5.errorbar(all_data_wave[i], unumpy.nominal_values(all_yplot1[i]), unumpy.std_devs(all_yplot1[i]), fmt='o', color='blue', elinewidth=1,
             #ecolor='black', capsize=1, capthick=1)
        ax5.plot(all_Dmag_model_wave[i], all_Dmag_model[i], '--', color='red', linewidth=5)
        # ax2.set_xlabel('Wavelength (nm)')
        #ax5.invert_yaxis()
        #ax5.set_ylabel('Diff Mag')

    for i in [4]:
        #ax5.set_title("Diff Mag Model Fit, HD %s" % target_hd, fontsize=8)
        ax5.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_Dmag[i][:-2]), unumpy.std_devs(all_Dmag[i][:-2]), fmt='o', color='black', ms=5, elinewidth=5,
             ecolor='black', capsize=5, capthick=5)
        ax5.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_Dmag[i][-2:]), unumpy.std_devs(all_Dmag[i][-2:]), fmt='o', color='lightgrey', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax5.plot(all_Dmag_model_wave[i], all_Dmag_model[i], '--', color='black', linewidth=5)
        #ax5.set_xlabel('Wavelength (nm)', fontsize=6)
        ax5.invert_yaxis()
        ax5.set_ylabel('Diff Mag', fontsize=35)
        ax5.tick_params(axis='both', labelsize=30)
        ax5.set_aspect('auto')
        ax5.tick_params('x', labelbottom=False)


    for i in [1]:
        #ax6.set_title("Split SED Model Fit, HD %s" % target_hd, fontsize=5)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag1[i][:-2]), unumpy.std_devs(all_split_mag1[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='black', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag1[i][-2:]), unumpy.std_devs(all_split_mag1[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag2[i][:-2]), unumpy.std_devs(all_split_mag2[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='grey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag2[i][-2:]), unumpy.std_devs(all_split_mag2[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model1[i], '--', color='blue', linewidth=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model2[i], '--', color='blue', linewidth=5)
        #ax6.invert_yaxis()
        ax6.set_xlabel('Wavelength (nm)', fontsize=25)
        ax6.set_ylabel('Apparent Mag', fontsize=25)

    for i in [7]:
        #ax6.set_title("Split SED Model Fit, HD %s" % target_hd, fontsize=5)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag1[i][:-2]), unumpy.std_devs(all_split_mag1[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='black', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag1[i][-2:]), unumpy.std_devs(all_split_mag1[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag2[i][:-2]), unumpy.std_devs(all_split_mag2[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='grey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag2[i][-2:]), unumpy.std_devs(all_split_mag2[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model1[i], '--', color='green', linewidth=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model2[i], '--', color='green', linewidth=5)
        #ax6.invert_yaxis()
        ax6.set_xlabel('Wavelength (nm)', fontsize=25)
        ax6.set_ylabel('Apparent Mag', fontsize=25)

    for i in [3,5]:
        ax6.set_title("Split SED Model Fit, HD %s" % target_hd, fontsize=8)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag1[i][:-2]), unumpy.std_devs(all_split_mag1[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='black', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag1[i][-2:]), unumpy.std_devs(all_split_mag1[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag2[i][:-2]), unumpy.std_devs(all_split_mag2[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='grey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag2[i][-2:]), unumpy.std_devs(all_split_mag2[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model1[i], '--', color='red', linewidth=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model2[i], '--', color='red', linewidth=5)
        #ax6.invert_yaxis()
        ax6.set_xlabel('Wavelength (nm)', fontsize=25)
        ax6.set_ylabel('Apparent Mag', fontsize=25)
    for i in [4]:
        ax6.set_title("Split SED Model Fit, HD %s" % target_hd, fontsize=8)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag1[i][:-2]), unumpy.std_devs(all_split_mag1[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='black', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag1[i][-2:]), unumpy.std_devs(all_split_mag1[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][:-2], unumpy.nominal_values(all_split_mag2[i][:-2]), unumpy.std_devs(all_split_mag2[i][:-2]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='grey', capsize=5, capthick=5)
        ax6.errorbar(all_Dmag_wave[i][-2:], unumpy.nominal_values(all_split_mag2[i][-2:]), unumpy.std_devs(all_split_mag2[i][-2:]), fmt='o', color='blue', ms=5, elinewidth=5,
             ecolor='lightgrey', capsize=5, capthick=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model1[i], '--', color='black', linewidth=5)
        ax6.plot(all_TOTmag_model_wave[i], all_model2[i], '--', color='black', linewidth=5)
        ax6.invert_yaxis()
        ax6.set_xlabel('Wavelength (nm)', fontsize=35)
        ax6.set_ylabel('Apparent Mag', fontsize=35)
        ax6.tick_params(axis='both', labelsize=30)
        ax6.set_aspect('auto')

    for i in [1]:
            ax1.plot(all_modelx_best[i], all_modely_best[i], color='blue', linewidth=5,label=f"Best log age_feh_-0.1 = {np.around(all_age_best[i], 2)} ")
            #ax1.invert_yaxis()
            label=f"Best log age_feh_-0.1 = {np.around(all_age_best[i], 2)} "
            ax1.set_title("HD %s" % target_hd, fontsize=50)
            ax1.legend(fontsize=5)

    for i in [7]:
            ax1.plot(all_modelx_best[i], all_modely_best[i], color='green', linewidth=5, label=f"Best log age feh_0.1= {np.around(all_age_best[i], 2)} ")
            #ax1.invert_yaxis()
            ax1.set_title("HD %s" % target_hd, fontsize=50)
            ax1.legend(fontsize=5)


    for i in [3,5]:
            ax1.plot(all_modelx_best[i], all_modely_best[i], label=f"Best log age = {np.around(all_age_best[i], 2)} ", color='red', linewidth=5)
            #ax1.invert_yaxis()
            ax1.set_title("HD %s" % target_hd, fontsize=35)
            ax1.legend(fontsize=5)

    for i in [4]:
            ax1.plot(all_modelx_best[i], all_modely_best[i], label=f"Best log age = {np.around(all_age_best[i], 2)} ", color='black', linewidth=5)
            ax1.errorbar(all_xval1[i].nominal_value, all_yval1[i].nominal_value,
                         xerr=all_xval1[i].std_dev, yerr=all_yval1[i].std_dev,
                         color="red")
            ax1.errorbar(all_xval2[i].nominal_value, all_yval2[i].nominal_value,
                         xerr=all_xval2[i].std_dev, yerr=all_yval2[i].std_dev,
                         color="red")
            ax1.annotate(f'Feh =-0.1: Age = {np.around(all_age_best[1], 2)}', xy=(all_xval1[i].nominal_value+0.05, all_yval1[i].nominal_value), xytext=(all_xval1[i].nominal_value+0.2, all_yval1[i].nominal_value+0.5), color = 'blue',size=25)
            ax1.annotate(f'Feh =+0.1: Age = {np.around(all_age_best[7], 2)}', xy=(all_xval1[i].nominal_value+0.05, all_yval1[i].nominal_value), xytext=(all_xval1[i].nominal_value+0.2, all_yval1[i].nominal_value-0.1), color = 'green',size=25)
            ax1.set_title("HD %s" % target_hd, fontsize=40)
            ax1.set_xlabel(xlabel, fontsize=35)
            ax1.set_ylabel(ylabel, fontsize=35)
            ax1.legend(fontsize=30)
            comb_xvals = [all_xval2[i].nominal_value, all_xval1[i].nominal_value]
            comb_err_xvals = [all_xval2[i].std_dev, all_xval1[i].std_dev]
            #pdb.set_trace()
            main_xvals = [all_modely_best[i]]
            #pdb.set_trace()


            if (np.min(comb_xvals) < np.median(all_modelx_best[i])) == True:
                x_axis_max = np.median(all_modelx_best[i]) + 0.15
                idx_xmax = comb_xvals.index(np.max(comb_xvals))
                idx_xmin = comb_xvals.index(np.min(comb_xvals))
                x_axis_min =np.min(comb_xvals) - comb_err_xvals[idx_xmin] - 0.1- comb_err_xvals[idx_xmin] - 0.1
            elif (np.min(comb_xvals) < np.median(all_modelx_best[i])) == False and np.isnan(comb_xvals[1]) == False:
                idx_xmax = comb_xvals.index(np.max(comb_xvals))
                x_axis_max = np.max(comb_xvals) + comb_err_xvals[idx_xmax] + 0.1
                x_axis_min = np.median(all_modelx_best[i]) - 0.15
            elif np.isnan(comb_xvals[1]) == True:
                x_axis_min = np.min(all_modelx_best[i])-0.15
                x_axis_max = np.max(all_modelx_best[i]) - 0.15
            ax1.set_xlim(x_axis_min,x_axis_max)

            comb_yvals = [all_yval2[i].nominal_value, all_yval1[i].nominal_value]
            comb_err_yvals = [all_yval2[i].std_dev, all_yval1[i].std_dev]
            if (np.min(comb_yvals) < np.median(all_modely_best[i])) == True:
                idx_ymin = comb_yvals.index(np.min(comb_yvals))
                idx_ymax = comb_yvals.index(np.max(comb_yvals))
                y_axis_max = np.median(all_modely_best[i]) + 0.15
                y_axis_min = np.min(comb_yvals) - comb_err_yvals[idx_ymin] - 0.1
            elif (np.min(comb_yvals) > np.median(all_modely_best[i])) == True  and np.isnan(comb_yvals[1]) == False:
                idx_ymin = comb_yvals.index(np.min(comb_yvals))
                idx_ymax = comb_yvals.index(np.max(comb_yvals))
                y_axis_max = np.max(comb_yvals) + comb_err_yvals[idx_ymax] + 0.1
                y_axis_min = np.median(all_modely_best[i]) - 0.15
            elif np.isnan(comb_yvals[1]) == True:
                y_axis_min = np.min(all_modely_best[i])-0.15
                y_axis_max = np.max(all_modely_best[i]) - 0.15
            #ax1.set_ylim(y_axis_min,y_axis_max)
            ax1.set_xlim(-1,2)
            ax1.invert_yaxis()
            ax1.tick_params(axis='both', labelsize=30)
            ax1.set_aspect('auto')
    #pdb.set_trace()
    for i in range(len(all_m_dyn_mcmc)):

        ax8.scatter(all_m_dyn_orbit[i].nominal_value, all_m_phot[i].nominal_value, s=25, color ='black')
        ax8.set_title('M_dyn vs M_phot', fontsize=40)
        ax8.set_xlim(all_m_dyn_orbit[0].nominal_value - 0.2, all_m_dyn_orbit[8].nominal_value + 0.2)
        ax8.set_ylim(all_m_phot[0].nominal_value - 0.3, all_m_phot[8].nominal_value + 0.3)
        #ax8.set_xlim(3.3,7.8)
        #ax8.set_ylim(3.3,7.8)
        #ax8.axvline(x= all_m_dyn[i].nominal_value, color='grey')
        ax8.tick_params(axis='both', labelsize=30)

    #pdb.set_trace()
    for i in range(len(all_m_dyn_mcmc)):
        all_m_dyn_mcmc_nom.append(all_m_dyn_mcmc[i].nominal_value)
        all_m_dyn_mcmc_err.append(all_m_dyn_mcmc[i].std_dev)




    for i in range(len(all_m_dyn_orbit)):
        all_m_dyn_orbit_nom.append(all_m_dyn_orbit[i].nominal_value)
        all_m_dyn_orbit_err.append(all_m_dyn_orbit[i].std_dev)
        all_m_phot_nom.append(all_m_phot[i].nominal_value)
        all_m_phot_err.append(all_m_phot[i].std_dev)


    all_mass = np.append(all_mass, all_m_dyn_mcmc_nom)
    all_mass = np.append(all_mass, all_m_phot_nom)
    all_mass = np.append(all_mass, all_m_dyn_orbit_nom)
    for i in range(len(all_m_phot)):
        plus_sig_phot =  all_m_phot_nom[i]+all_m_phot_err[i]
        minus_sig_phot =  all_m_phot_nom[i]-all_m_phot_err[i]
        plus_sig_mcmc =  all_m_dyn_mcmc_nom[i]+all_m_dyn_mcmc_err[i]
        minus_sig_mcmc =  all_m_dyn_mcmc_nom[i]-all_m_dyn_mcmc_err[i]
        plus_sig_orbit =  all_m_dyn_orbit_nom[i]+all_m_dyn_orbit_err[i]
        minus_sig_orbit =  all_m_dyn_orbit_nom[i]-all_m_dyn_orbit_err[i]

        all_mass= np.append(all_mass, plus_sig_phot)
        all_mass= np.append(all_mass, minus_sig_phot)
        all_mass= np.append(all_mass, plus_sig_mcmc)
        all_mass= np.append(all_mass, minus_sig_mcmc)
        all_mass= np.append(all_mass, plus_sig_orbit)
        all_mass= np.append(all_mass, minus_sig_orbit)


    all_mass = np.append(all_mass, all_m_dyn_mcmc_nom)
    all_mass = np.append(all_mass, all_m_dyn_orbit_nom)
    max = np.max(all_mass)+0.1
    min = np.min(all_mass)-0.1
    #pdb.set_trace()
    ax8.plot(all_m_dyn_mcmc_nom[0:3], all_m_phot_nom[0:3], color = 'darkred', linewidth =5)
    ax8.plot(all_m_dyn_mcmc_nom[3:6], all_m_phot_nom[3:6], color = 'navy', linewidth =5)
    ax8.plot(all_m_dyn_mcmc_nom[6:9], all_m_phot_nom[6:9], color = 'darkgreen',linewidth =5)
    ax8.plot(all_m_dyn_orbit_nom[0:3], all_m_phot_nom[0:3], color = 'red', linewidth =5)
    ax8.plot(all_m_dyn_orbit_nom[3:6], all_m_phot_nom[3:6], color = 'blue', linewidth =5)
    ax8.plot(all_m_dyn_orbit_nom[6:9], all_m_phot_nom[6:9], color = 'green',linewidth =5)
    ax8.errorbar(all_m_dyn_mcmc_nom[0:3], all_m_phot_nom[0:3], all_m_phot_err[0:3], all_m_dyn_mcmc_err[0:3],  color = 'darkred', linewidth =3)
    ax8.errorbar(all_m_dyn_mcmc_nom[3:6], all_m_phot_nom[3:6], all_m_phot_err[3:6], all_m_dyn_mcmc_err[3:6], color = 'navy', linewidth =3)
    ax8.errorbar(all_m_dyn_mcmc_nom[6:9], all_m_phot_nom[6:9], all_m_phot_err[6:9], all_m_dyn_mcmc_err[6:9], color = 'darkgreen',linewidth =3)
    ax8.errorbar(all_m_dyn_orbit_nom[0:3], all_m_phot_nom[0:3], all_m_phot_err[0:3], all_m_dyn_orbit_err[0:3],color = 'red', linewidth =3)
    ax8.errorbar(all_m_dyn_orbit_nom[3:6], all_m_phot_nom[3:6], all_m_phot_err[3:6], all_m_dyn_orbit_err[3:6], color = 'blue', linewidth =3)
    ax8.errorbar(all_m_dyn_orbit_nom[6:9], all_m_phot_nom[6:9], all_m_phot_err[6:9], all_m_dyn_orbit_err[6:9], color = 'green',linewidth =3)
    ax8.set_xlim(min,max)
    ax8.set_ylim(min,max)
    #ax8.set_xlim(-1,2)
    #ax8.set_ylim(-1,2)
    #ax8.plot([all_m_dyn_nom[2],all_m_dyn_nom[1]], [all_m_phot_nom[2],all_m_phot_nom[1]], color = 'red',linewidth =5)
    #ax8.plot([all_m_dyn_nom[5],all_m_dyn_nom[4]], [all_m_phot_nom[5],all_m_phot_nom[4]], color = 'blue', linewidth =5)
    #ax8.plot([all_m_dyn_nom[8],all_m_dyn_nom[7]], [all_m_phot_nom[8],all_m_phot_nom[7]], color = 'green', linewidth =5)
    #ax8.annotate('Feh =-0.1', xy=(all_m_dyn_nom[2]+0.03, all_m_phot_nom[5]- (np.absolute(all_m_phot_nom[5]- all_m_phot_nom[2]))), xytext=(all_m_dyn_nom[5]+0.03, all_m_phot_nom[5]- (np.absolute(all_m_phot_nom[5]- all_m_phot_nom[2]))), color = 'red',size=25)
    #ax8.annotate('Feh =0.0', xy=(all_m_dyn_nom[5]+0.03, all_m_phot_nom[5]), xytext=(all_m_dyn_nom[5]+0.03, all_m_phot_nom[5]), color = 'blue',size=25)
    #ax8.annotate('Feh =0.1', xy=(all_m_dyn_nom[8]+0.03, all_m_phot_nom[5]+ (np.absolute(all_m_phot_nom[5]- all_m_phot_nom[8]))), xytext=(all_m_dyn_nom[8]+00.03, all_m_phot_nom[5]+ (np.absolute(all_m_phot_nom[5]- all_m_phot_nom[8]))), color = 'green',size=25)
    #ax8.annotate('d=-$\sigma$', xy=(all_m_dyn_nom[1]- (np.absolute(all_m_phot_nom[5]- all_m_phot_nom[2])), all_m_phot_nom[0]-0.03), xytext=(all_m_dyn_nom[0]-0.07, all_m_phot_nom[0]-0.03), color = 'red',size=25)
    #ax8.annotate('d=0$\sigma$', xy=(all_m_dyn_nom[1], all_m_phot_nom[1]-0.03), xytext=(all_m_dyn_nom[1]-0.01, all_m_phot_nom[1]-0.05),color = 'blue',size=25)
    #ax8.annotate('d=+$\sigma$', xy=(all_m_dyn_nom[1]+ (np.absolute(all_m_phot_nom[5]- all_m_phot_nom[2])), all_m_phot_nom[2]-0.03), xytext=(all_m_dyn_nom[2]+0.03, all_m_phot_nom[2]-0.03),size=25, color ='green')
    ax8.axline((0, 0), slope=1., color='green', label='M_dyn = M_phot')
    #ax8.axvline(x=all_m_dyn[1].nominal_value, color='salmon')
    ax8.legend(fontsize=35)
    #ax8.set_aspect('equal')
    ax8.set_xlabel('M_dyn', fontsize=35)
    ax8.set_ylabel('M_phot', fontsize=35)
    ax8.grid()



    #pdb.set_trace()
    #file = f'{orbit_directory}HD{target_hd}__outer_mcmc.pdf'
    #pdf_file = fitz.open(file)
    #for page in pdf_file:  # iterate through the pages
        #pix = page.get_pixmap()  # render page to an image
        #pix.save(f'{orbit_directory}HD{target_hd}.png')  # store image as a PNG

    # Store Pdf with convert_from_path function
    #images = convert_from_path(f'{orbit_directory}HD{target_hd}__outer_mcmc.pdf',700)
    #for i in range(len(images)):
        # Save pages as images in the pdf
        #images[i].save(f'{orbit_directory}HD{target_hd}.png', 'PNG')



    pic = f'{orbit_directory}HD{target_hd}.png'

    img = io.imread(pic)
    ax7.imshow(img)
    box = ax7.get_position()
    box.y0 = box.y0 - 4000
    box.y1 = box.y1 + 4000
    ax7.set_position(box)
    e = np.round(float(df_armada['e'][idx]),2)
    a = np.round(float(df_armada['a (mas)'][idx]),2)
    p = np.round(float(df_armada['P (yr)'][idx]),2)
    #pdb.set_trace()
    ax7.set_aspect('equal')
    e = ufloat(float(df_armada['e'][idx]),float(df_armada['e_err'][idx]))

    ax7.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    ax7.annotate(f"PA(deg)= {df_photometry['PA(deg)_tycoDouble'][idx]}", xy=(0, 0), xytext=(2000, -300), color = 'black',size=35)
    ax7.annotate(f"Sep(arcsec) ={df_photometry['Sep(arcsec)'][idx]}", xy=(0, 0), xytext=(2000, -100), color = 'black',size=35)
    ax7.annotate(f"Triple = {df_armada['triple'][idx]}", xy=(0, 0), xytext=(0.0, -300), color = 'black',size=35)
    ax7.annotate(f"Residual ={df_armada['residual (micro-as)'][idx]}", xy=(0, 0), xytext=(0.0, -100), color = 'black',size=35)
    ax7.annotate(f"e={e:4f}", xy=(0, 0), xytext=(0.0, 4300), color = 'black',size=35)
    ax7.annotate(f"a(mas)={a_mas:.3f}", xy=(0, 0), xytext=(2000, 4300), color = 'black',size=35)
    ax7.annotate(f"P(Yr)={p_yr:.3f}", xy=(0, 0), xytext=(0, 4100), color = 'black',size=35)
    ax7.annotate(f"d(pc)={distance_best:.2f}", xy=(0, 0), xytext=(2000, 4100), color = 'black',size=35)
    ax7.annotate(f"Spectral Type={df_armada['SpType (primary?)'][idx]}", xy=(0, 0), xytext=(0, 3900), color = 'black',size=35)
    ax7.annotate(f"M_Dyn Orbit={all_m_dyn_orbit[4]}", xy=(0, 0), xytext=(2000, 3900), color = 'black',size=35)
    ax7.annotate(f"M_Dyn McMc={all_m_dyn_mcmc[4]}", xy=(0, 0), xytext=(2000, 3700), color='black', size=35)
    ax7.annotate(f"M_Phot={all_m_phot[4]}", xy=(0, 0), xytext=(0, 3700), color='black', size=35)



    #ax7.annotate(f"a={df_armada['plx (mas)'][idx]}", xy=(0.6, 0.01), xytext=(0.6, 0.01), color='black', size=5)


    fig.savefig("%s/HD_%s_%s_all_SED_fit.pdf" % (directory2, target_hd, note))

    # create a list of directories
    dirs = [f'{corner_directory}HD_{target_hd}/']

    # extract the image paths into a list
    files = [f for dir_ in dirs for f in list(Path(dir_).glob('*.png'))]

    # create the figure
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(55, 44.2))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.0, wspace=0.0)

    # flatten the axis into a 1-d array to make it easier to access each axes
    axs = axs.flatten()

    # iterate through and enumerate the files, use i to index the axes
    for i, file in enumerate(files):
        # read the image in
        pic = plt.imread(file)

        # add the image to the axes
        axs[i].imshow(pic)
        axs[i].axis('off')

        # add an axes title; .stem is a pathlib method to get the filename
        axs[i].set(title=file.stem)
    fig.savefig(f'{corner_directory}_{target_hd}', dpi=fig.dpi)



print("Failed Targets = ")
print(Target_List_Fail)

#df = pd.read_csv(csv, header= None)
#df.to_csv('file.csv', header = Header)