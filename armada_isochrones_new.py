## Routine to plot Gaia binaries on an HR diagram and fit age

import numpy as np
from matplotlib import pyplot as plt
import os
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from isochrones.mist import MIST_EvolutionTrack, MIST_Isochrone
from isochrones import get_ichrone
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from tqdm import tqdm
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import pandas as pd
Mist_iso = MIST_Isochrone()
Mist_evoTrack = MIST_EvolutionTrack()

user_path = '/Users/tgardner/ARMADA_final/'

wds_file = '%s/WDS_Data/WDS_Data_converted/'%user_path
aavso_file = '%s/armada_photometry_aavso.xlsx'%user_path
armada_file = '%s/full_target_list.csv'%user_path
photometry_file = '%s/Photometry.csv'%user_path

aavso_method = input('Use new AAVSO magnitudes when available? (y/[n]):')
fit_contrasts = 'y'
feh_user = input('Fix feh? (y,[n]): ')
note = input('Note for saved file: ')
if feh_user == 'y':
    feh = float(input('Feh = '))
else:
    feh = [-0.1,0,0.1]
main_sequence = input('Use main sequence ages (y,[n])? ')

Target_List = [1976,2772,5143,6456,10453,11031,16753,17094,27176,29316,29573,31093,31297,34319,36058,
               37269,37711,38545,38769,40932,41040,43358,43525,45542,46273,47105,48581,49643,60107,64235,
               75974,78316,82446,87652,87822,107259,112846,114993,118889,127726,128415,129246,133484,133955,
               137798,137909,140159,140436,144892,145589,148283,153370,154569,156190,158140,160935,163346,
               166045,173093,178475,179950,185404,185762,189037,189340,195206,196089,196867,198183,199766,
               201038,206901,217676,217782,220278,224512]

## failed targets with 0 metallicity fixed --> [10453,17094,27176,137909,163346]
## failed targets with varying metallicity --> [17094, 27176, 137909]

Target_List = [10453]

## Fuction to fit a single star model
def single_star_fit(params,split_mag_star,wave_star):
    split_mag_val = unumpy.nominal_values(split_mag_star)
    split_mag_err = unumpy.std_devs(split_mag_star)

    age = params['age']
    mass = params['mass']
    feh = params['feh']
    Av = params['Av']

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

def mass_search(mass1_grid,mass2_grid,log_age_guess,split_mag1,split_mag2,split_mag_wl,Av,feh):
        
    ## search for mass 1
    chi2_grid1 = []
    mass1_result = []
    for mm in mass1_grid:
        try:
            params = Parameters()
            params.add('age', value=log_age_guess, vary=False) ## NOTE: I could vary this!
            params.add('mass', value=mm, vary=False)
            params.add('feh', value=feh, vary=False)
            params.add('Av', value=Av, vary=False)
            minner = Minimizer(single_star_fit, params, fcn_args=(split_mag1,split_mag_wl),
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
            params.add('Av', value=Av, vary=False)
            minner = Minimizer(single_star_fit, params, fcn_args=(split_mag2,split_mag_wl),
                            nan_policy='omit')
            result = minner.minimize()
            chi2_grid2.append(result.redchi)
            mass2_result.append(mm)
        except:
            # print("Fails at log age = %s"%aa)
            pass

    return chi2_grid1,mass1_result,chi2_grid2,mass2_result

def isochrone_fit(params,TOT_mag,mag_wave_star,D_mag,dmag_wave_star,Teff):

    TOT_mag_val = unumpy.nominal_values(TOT_mag)
    TOT_mag_err = unumpy.std_devs(TOT_mag)
    D_mag_val = unumpy.nominal_values(D_mag)
    D_mag_err = unumpy.std_devs(D_mag)

    Teff_val = Teff.nominal_value
    Teff_err = Teff.std_dev
    
    feh = params['feh']
    age = params['age']
    Av = params['Av']
    m1 = params['mass1']
    m2 = params['mass2']
    
    a1 = tracks.generate(m1, age, feh, return_dict=True)
    a2 = tracks.generate(m2, age, feh, return_dict=True)
    
    if np.isnan(a1['Mbol']) or np.isnan(a2['Mbol']):
       FAIL ## kill the program
    
    mag1_model_all = np.array([(a1['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0],
                           (a1['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh, Av]))[0][0]])

    mag2_model_all = np.array([(a2['Mbol'] - bc_grid_U.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_B.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_V.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_R.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_I.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_J.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_H.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0],
                           (a2['Mbol'] - bc_grid_K.interp([a2['Teff'], a2['logg'], feh, Av]))[0][0]])
    
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

    Teff_model = a1['Teff']
    diff3 = [(Teff_val - Teff_model) / Teff_err]
    
    return np.concatenate([diff1,diff2,diff3])

def find_max_mass(age,m1_grid,tot_mag,tot_mag_wavelengths,cdiff,diff_mag_wavelengths,feh,Av,Teff):
    m1_max = np.nanmax(m1_grid)
    for m1 in m1_grid:
        try:
            params = Parameters()
            params.add('age',value=age,vary=False)
            params.add('feh',value=feh,vary=False)
            params.add('Av',value=Av,vary=False)
            params.add('mass1',value=m1,vary=False)
            params.add('mass2',value=0.5,vary=False)
            minner = Minimizer(isochrone_fit, params, fcn_args=(tot_mag,tot_mag_wavelengths,cdiff,diff_mag_wavelengths,Teff),
                                        nan_policy='omit')
            result = minner.minimize()
        except:
            m1_max = m1
            break
    return (m1_max)

failed_targets = []
for target_hd in Target_List:
    try:
        ## Specify target of interest
        target_hd = str(target_hd)
        target = 'HD_%s'%target_hd
        target_query = target.replace("_", " ")
        print('Doing Target %s'%target_query)

        ## Get target from spreadsheet
        df_armada = pd.read_csv(armada_file, dtype=object)
        df_photometry = pd.read_csv(photometry_file, dtype=object)
        df_aavso = pd.read_excel(aavso_file)
        df_wds = pd.read_csv("%s/HD_%s.csv"%(wds_file,target_hd))

        ## Get WDS flux ratio measurements:
        filtered_df_wds = df_wds[~df_wds['Method'].isin(['Ma', 'Mb'])]
        df_wds = filtered_df_wds
        err_array1 = df_wds['F1_err'].values
        idx_err_nan = np.where(np.isnan(err_array1))
        err_array1[idx_err_nan] = 0.1
        idx_err_nan = np.where(err_array1<0.02)
        err_array1[idx_err_nan] = 0.02
        err_array2 = df_wds['F2_err'].values
        idx_err_nan = np.where(np.isnan(err_array2))
        err_array2[idx_err_nan] = 0.1
        idx_err_nan = np.where(err_array2<0.02)
        err_array2[idx_err_nan] = 0.02

        F1 = unumpy.uarray(df_wds['F1'].values, err_array1)
        F2 = unumpy.uarray(df_wds['F2'].values, err_array2)
        cdiff_wds = F2-F1

        idx_nan = np.where(np.isnan(unumpy.nominal_values(cdiff_wds)))
        cdiff_wds[idx_nan] = F2[idx_nan]
        fratio_wds_wl = df_wds['Wavelength'].values

        ## get photometry and other properties either from AAVSO or from big spreadsheet
        idx_armada = np.where(df_armada['HD'] == target_hd)[0][0]
        idx_phot = np.where(df_photometry['HD'] == target_hd)[0][0]
        idx_aavso = np.where(df_aavso['typed ident'] == 'HD %s'%target_hd)[0][0]

        Av = float(df_armada['Av'][idx_armada])
        Teff_primary = ufloat(float(df_armada['Teff'][idx_armada]),0.1*float(df_armada['Teff'][idx_armada]))
        cdiff_h = ufloat(float(df_armada['dmag_h'][idx_armada]), float(df_armada['dmag_h_err'][idx_armada]) )
        cdiff_k = ufloat(float(df_armada['dmag_k'][idx_armada]),float(df_armada['dmag_k_err'][idx_armada]))
        cdiff_i = ufloat(float(df_armada['dmag_speckle_i'][idx_armada]), float(df_armada['dmag_speckle_i_err'][idx_armada]))
        cdiff_b = ufloat(float(df_armada['dmag_speckle_v'][idx_armada]),float(df_armada['dmag_speckle_v_err'][idx_armada]))
        fratio_h = 10 ** (cdiff_h / 2.5)
        fratio_k = 10 ** (cdiff_k / 2.5)
        fratio_i = 10 ** (cdiff_i / 2.5)
        fratio_b = 10 ** (cdiff_b / 2.5)
        fratio_wds = 10 ** (cdiff_wds / 2.5)

        fratio_data = np.array([fratio_b,fratio_i,fratio_h,fratio_k])
        cdiff_data = np.array([cdiff_b,cdiff_i,cdiff_h,cdiff_k])
        cdiff_data_fit = np.array([cdiff_b,cdiff_i,np.nan,np.nan]) ## NOTE: I am setting H and K bands to nan for fitting!!!!
        fratio_data_wl = np.array([562,832,1650,2150]) # Speckle + CHARA/VLTI

        fratio_all = np.concatenate([fratio_wds,fratio_data_wl])
        cdiff_all = np.concatenate([cdiff_wds,cdiff_data])
        cdiff_all_fit = np.concatenate([cdiff_wds,cdiff_data_fit])
        fratio_all_wl = np.concatenate([fratio_wds_wl,fratio_data_wl])

        ## get total magnitudes and errors from photometry file. Set minimum error 
        err_min = 0.02 
        aavso_fail='n'
        if aavso_method=='y':
            utot = ufloat(np.nan, np.nan)
            btot = ufloat(df_aavso['B'][idx_aavso],df_aavso['sig_B'][idx_aavso])
            vtot = ufloat(df_aavso['V'][idx_aavso],df_aavso['sig_V'][idx_aavso])
            rtot = ufloat(df_aavso['R'][idx_aavso],df_aavso['sig_R'][idx_aavso])
            itot = ufloat(np.nan, np.nan)
            if np.isnan(btot.nominal_value):
                print('No AAVSO for HD %s'%target_hd)
                aavso_fail='y'

        if aavso_method!='y' or aavso_fail=='y':      
            print('Using old photometry file (not aavso)')
            utot = ufloat(np.nan, np.nan)
            if np.isnan(float(df_photometry['B_err_complete'][idx_phot])) or float(df_photometry['B_err_complete'][idx_phot]) < err_min:
                btot = ufloat(float(df_photometry['B_complete'][idx_phot]), err_min)
            else:
                btot = ufloat(float(df_photometry['B_complete'][idx_phot]), float(df_photometry['B_err_complete'][idx_phot]))
            if np.isnan(float(df_photometry['V_err_complete'][idx_phot])) or float(df_photometry['V_err_complete'][idx_phot]) < err_min:
                vtot = ufloat(float(df_photometry['V_complete'][idx_phot]), err_min)
            else:
                vtot = ufloat(float(df_photometry['V_complete'][idx_phot]), float(df_photometry['V_err_complete'][idx_phot]))
            #rtot = ufloat(float(df_photometry['R2_I/284'][idx_phot]), 0.15)
            rtot = ufloat(np.nan, np.nan)
            itot = ufloat(np.nan, np.nan)
        
        ## JHK always comes from photometry spreadsheet (no AAVSO)
        if np.isnan(float(df_photometry['J_err'][idx_phot])) or float(df_photometry['J_err'][idx_phot]) < err_min:
            jtot = ufloat(float(df_photometry['J_ II/246/out'][idx_phot]), err_min)
        else:
            jtot = ufloat(float(df_photometry['J_ II/246/out'][idx_phot]), float(df_photometry['J_err'][idx_phot]))
        if np.isnan(float(df_photometry['H_err'][idx_phot])) or float(df_photometry['H_err'][idx_phot]) < err_min:
            htot = ufloat(float(df_photometry['H_ II/246/out'][idx_phot]), err_min)
        else:
            htot = ufloat(float(df_photometry['H_ II/246/out'][idx_phot]), float(df_photometry['H_err'][idx_phot]))
        if np.isnan(float(df_photometry['K_err'][idx_phot])) or float(df_photometry['K_err'][idx_phot]) < err_min:
            ktot = ufloat(float(df_photometry['K_ II/246/out'][idx_phot]), err_min)
        else:
            ktot = ufloat(float(df_photometry['K_ II/246/out'][idx_phot]), float(df_photometry['K_err'][idx_phot]))

        ## choose which distance to use -- Gaia, HIP, or Kervella catalog
        distance_gaia = ufloat(float(df_armada['Gaia_distance (pc)'][idx_armada]), float(df_armada['Gaia_distance_err (pc)'][idx_armada]))
        distance_hip = ufloat(float(df_armada['HIP_distance (pc)'][idx_armada]), float(df_armada['HIP_distance_err (pc)'][idx_armada]))
        distance_kervella = ufloat(float(df_armada['kerv_dist'][idx_armada]), float(df_armada['e_kerv'][idx_armada]))

        distances = np.array([distance_gaia,distance_hip,distance_kervella])
        idx_lowest = np.nanargmin(unumpy.std_devs(distances))
        distance_best = distances[idx_lowest]
        #distance_low = distance_best - distance_best.std_dev
        #distance_high = distance_best + distance_best.std_dev
        #distance_set =[distance_low,distance_best,distance_high]

        ## Bessel_U, Bessel_B, Bessel_V, Johnson_R, Bessel_I, 2MASS_J, 2MASS_H, 2MASS_K
        TOT_Mag = np.array([utot, btot, vtot, rtot, itot, jtot, htot, ktot])
        TOT_Mag_wl = np.array([365, 445, 551, 675, 806, 1250, 1650, 2150])

        ## absolute magnitudes
        TOT_Mag_absolute = TOT_Mag - 2.5 * log10( (distance_best/10)**2 )

        ## Compute individual split magnitudes at wavelengths with measured fratios
        tot_mag_ratios_val = np.interp(fratio_all_wl,TOT_Mag_wl,unumpy.nominal_values(TOT_Mag_absolute))
        tot_mag_ratios_err = np.interp(fratio_all_wl,TOT_Mag_wl,unumpy.std_devs(TOT_Mag_absolute))
        tot_mag_ratios = unumpy.uarray(tot_mag_ratios_val,tot_mag_ratios_err)
        split_mag1 = -2.5 * unumpy.log10(10 ** (-tot_mag_ratios / 2.5) / (1 + 10 ** (-cdiff_all_fit / 2.5)))
        split_mag2 = cdiff_all_fit + split_mag1

        ## An isochrone model that interpolate the model photometry
        tracks = get_ichrone('mist', tracks=True, accurate=True)
        bc_grid_U = MISTBolometricCorrectionGrid(['U'])
        bc_grid_B = MISTBolometricCorrectionGrid(['B'])
        bc_grid_V = MISTBolometricCorrectionGrid(['V'])
        bc_grid_R = MISTBolometricCorrectionGrid(['R'])
        bc_grid_I = MISTBolometricCorrectionGrid(['I'])
        bc_grid_J = MISTBolometricCorrectionGrid(['J'])
        bc_grid_H = MISTBolometricCorrectionGrid(['H'])
        bc_grid_K = MISTBolometricCorrectionGrid(['K'])

        ## age range to grid over
        ages = np.linspace(6,11,1000)

        if feh_user == 'y':
            ## FIXED metallicity method
            age_grid = []
            chi2_grid = []
            m1_grid = []
            m2_grid = []
            for aa in tqdm(ages):        
            
                mass1_grid = np.linspace(0.5,5,20)
                mass2_grid = np.linspace(0.5,5,20)
                try:
                    chi2_grid1,mass1_result,chi2_grid2,mass2_result = mass_search(mass1_grid,mass2_grid,aa,split_mag1,split_mag2,fratio_all_wl,Av,feh)
                    idx_mass1 = np.argmin(chi2_grid1)
                    mass1_guess = mass1_result[idx_mass1]
                    idx_mass2 = np.argmin(chi2_grid2)
                    mass2_guess = mass2_result[idx_mass2]
                    max_mass = find_max_mass(aa,mass1_grid,TOT_Mag_absolute,TOT_Mag_wl,cdiff_all_fit,fratio_all_wl,feh,Av,Teff_primary)

                    params = Parameters()
                    params.add('age',value=aa,vary=False)
                    params.add('feh',value=feh,vary=False)
                    params.add('Av',value=Av,vary=False)
                    params.add('mass1',value=mass1_guess, min=0, max=max_mass)
                    params.add('mass2',value=mass2_guess, min=0, max=max_mass)

                    minner = Minimizer(isochrone_fit, params, fcn_args=(TOT_Mag_absolute,TOT_Mag_wl,cdiff_all_fit,fratio_all_wl,Teff_primary),
                                                nan_policy='omit')
                    result = minner.minimize()
                    chi2_grid.append(result.redchi)
                    age_grid.append(result.params['age'].value)
                    m1_grid.append(result.params['mass1'].value)
                    m2_grid.append(result.params['mass2'].value)
                except:
                    chi2_grid.append(np.nan)
                    age_grid.append(aa)
                    m1_grid.append(np.nan)
                    m2_grid.append(np.nan)

            age_grid = np.array(age_grid)
            chi2_grid = np.array(chi2_grid)
            m1_grid = np.array(m1_grid)
            m2_grid = np.array(m2_grid)

            if main_sequence=='y':
                ## only pay attendtion to main sequence ages
                chi2_grid_ms = chi2_grid.copy()
                chi2_grid_ms[np.where(age_grid<7)[0]] = np.nan
                idx = np.nanargmin(chi2_grid_ms)
            else:
                idx = np.nanargmin(chi2_grid)
            age_best = age_grid[idx]
            feh_best = feh
            mass1_best = m1_grid[idx]
            mass2_best = m2_grid[idx]

            print("Best log age, best feh = ", age_best,feh_best)
            print("Best M1, M2 = ", mass1_best, mass2_best)

            if aavso_method=='y' and aavso_fail=='n':
                if feh_user=='y':
                    directory = '%s/ARMADA_isochrones/%s/fixed/aavso/'%(user_path,target)
                else:
                    directory = '%s/ARMADA_isochrones/%s/varied/aavso/'%(user_path,target)
            else:
                if feh_user=='y':
                    directory = '%s/ARMADA_isochrones/%s/fixed/simbad/'%(user_path,target)
                else:
                    directory = '%s/ARMADA_isochrones/%s/varied/simbad/'%(user_path,target)

            if not os.path.exists(directory):
                print("Creating target directory")
                os.makedirs(directory)

            ## Make plots
            chi2mask = np.isfinite(chi2_grid)
            plt.plot(age_grid[chi2mask],chi2_grid[chi2mask],'--')
            plt.title(target_query)
            plt.figtext(0.15, 0.83, 'Best log age = %s, Feh = %s'%(np.around(age_best,2), feh_best))
            plt.figtext(0.15, 0.77, 'M1, M2 (M$_{\odot}$) = %s, %s'%(np.around(mass1_best,2),np.around(mass2_best,2)))
            plt.xlabel('Log Age')
            plt.ylabel('Chi2')
            plt.yscale('log')
            plt.savefig('%s/%s_%s_%s_chi2grid.png'%(directory,target,feh_best,note),dpi=400,bbox_inches='tight')
            plt.close()

            ## Make HR diagram plot
            a1 = tracks.generate(mass1_best, age_best, feh_best, return_dict=True)
            a2 = tracks.generate(mass2_best, age_best, feh_best, return_dict=True)

            mag1_u = (a1['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
            mag1_b = (a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
            mag1_v = (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
            mag1_r = (a1['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
            mag1_i = (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
            mag1_j = (a1['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
            mag1_h = (a1['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
            mag1_k = (a1['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]

            mag2_u = (a2['Mbol'] - bc_grid_U.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
            mag2_b = (a2['Mbol'] - bc_grid_B.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
            mag2_v = (a2['Mbol'] - bc_grid_V.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
            mag2_r = (a2['Mbol'] - bc_grid_R.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
            mag2_i = (a2['Mbol'] - bc_grid_I.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
            mag2_j = (a2['Mbol'] - bc_grid_J.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
            mag2_h = (a2['Mbol'] - bc_grid_H.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
            mag2_k = (a2['Mbol'] - bc_grid_K.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]

            mag1 = np.array([mag1_u,mag1_b,mag1_v,mag1_r,mag1_i,mag1_j,mag1_h,mag1_k])
            mag2 = np.array([mag2_u,mag2_b,mag2_v,mag2_r,mag2_i,mag2_j,mag2_h,mag2_k])
            tot_mag = -2.5 * np.log10(10 ** (-0.4 * mag1) + 10 ** (-0.4 * mag2))
            cdiff = mag2-mag1
            xval = np.array([365,445,551,658,806,1250,1653,2200])

            plt.plot(np.sort(xval),tot_mag[np.argsort(xval)],'+--',color='red')
            for wl,magnitude in zip(xval,TOT_Mag_absolute):
                plt.errorbar(wl,magnitude.nominal_value,yerr=magnitude.std_dev,fmt='.',color='black')
            plt.xlabel('Wavelength')
            plt.ylabel('Magnitude')
            plt.title(target_query)
            plt.gca().invert_yaxis()
            plt.savefig('%s/%s_%s_%s_magnitudes.png'%(directory,target,feh_best,note),dpi=400,bbox_inches='tight')
            plt.close()

            for wl,contrast in zip(fratio_all_wl,cdiff_all[:-2]):
                plt.errorbar(wl,contrast.nominal_value,yerr=contrast.std_dev,fmt='.',color='black')
            plt.errorbar(fratio_all_wl[-2],cdiff_all[-2].nominal_value,yerr=cdiff_all[-2].std_dev,fmt='.',color='lightgrey')
            plt.errorbar(fratio_all_wl[-1],cdiff_all[-1].nominal_value,yerr=cdiff_all[-1].std_dev,fmt='.',color='lightgrey')
            plt.plot(np.sort(xval),cdiff[np.argsort(xval)],'+--',color='red')
            plt.xlabel('Wavelength')
            plt.ylabel('Flux difference')
            plt.title(target_query)
            plt.gca().invert_yaxis()
            plt.savefig('%s/%s_%s_%s_cdiff.png'%(directory,target,feh_best,note),dpi=400,bbox_inches='tight')
            plt.close()

            ## Here I am plotting V vs B-V for HR diagram
            ## Using V and B absolute magnitudes, and best-fit model contrasts to split to individual stars
            v1 = -2.5 * log10(10 ** (-TOT_Mag_absolute[2] / 2.5) / (1 + 10 ** (-cdiff[2] / 2.5)))
            v2 = cdiff[2] + v1
            b1 = -2.5 * log10(10 ** (-TOT_Mag_absolute[1] / 2.5) / (1 + 10 ** (-cdiff[1] / 2.5)))
            b2 = cdiff[1] + b1

            ###############
            ## Make an HR diagram plot
            ###############
            log_age_start = 7  ## starting age
            log_age_size = 1  ## step size
            log_age_steps = 4  ## number of steps

            ## isochrones
            paramList = [np.array([log_age_start, feh_best]) + np.array([log_age_size, 0]) * i for i in
                         range(0, log_age_steps)]
            paramBest = [np.array([age_best,feh_best])]
            isoList = [Mist_iso.isochrone(param[0], param[1]) for param in paramList]
            isoBest = [Mist_iso.isochrone(param[0], param[1]) for param in paramBest]

            V = []
            B = []
            for i, iso in enumerate(isoList):
                Mbol_V = bc_grid_V.interp([iso['Teff'],iso['logg'],feh,Av]).ravel()
                Mbol_B= bc_grid_B.interp([iso['Teff'],iso['logg'],feh,Av]).ravel()
                V.append(iso['Mbol'] - Mbol_V)
                B.append(iso['Mbol'] - Mbol_B)

            V_best = []
            B_best = []
            for i, iso in enumerate(isoBest):
                Mbol_V = bc_grid_V.interp([iso['Teff'],iso['logg'],feh,Av]).ravel()
                Mbol_B= bc_grid_B.interp([iso['Teff'],iso['logg'],feh,Av]).ravel()
                V_best.append(iso['Mbol'] - Mbol_V)
                B_best.append(iso['Mbol'] - Mbol_B)

            xval1 = ufloat((b1 - v1).nominal_value, (b1+v1).std_dev)  ## component 1
            yval1 = v1 
            xval2 = ufloat((b2 - v2).nominal_value, (b2+v2).std_dev)  ## component 2
            yval2 = v2

            iso_start = 50
            iso_end = 500
            for i, iso in enumerate(isoList):
                ## make sure model matches data magnitudes
                modelx = B[i][iso_start:iso_end] - V[i][iso_start:iso_end]
                modely = V[i][iso_start:iso_end]
                plt.plot(modelx, modely,color='lightgreen')#,label=f"log age = {log_age_start + log_age_size * i} "
                plt.annotate("%s"%(log_age_start + log_age_size * i),xy=(modelx.values[-1], modely.values[-1]-0.2), color='lightgreen')
            for i, iso in enumerate(isoBest):
                ## make sure model matches data magnitudes
                modelx = B_best[i][iso_start:iso_end] - V_best[i][iso_start:iso_end]
                modely = V_best[i][iso_start:iso_end]
                plt.plot(modelx, modely,color='darkgreen')#,label=f"log age = {log_age_start + log_age_size * i} "
                plt.annotate("%s"%np.around(age_best,2),xy=(modelx.values[-1], modely.values[-1]-0.2),color='darkgreen')

            ## make plot
            plt.errorbar(xval1.nominal_value, yval1.nominal_value,
                         xerr=xval1.std_dev, yerr=yval1.std_dev,
                         fmt='+',color="darkgreen",markersize=10)
            plt.errorbar(xval2.nominal_value, yval2.nominal_value,
                         xerr=xval2.std_dev, yerr=yval2.std_dev,
                         fmt='+',color="darkgreen",markersize=10)

            xmin = min(xval1,xval2).nominal_value
            xmax = max(xval1,xval2).nominal_value
            ymin = min(yval1,yval2).nominal_value
            ymax = max(yval1,yval2).nominal_value

            #plt.xlim(xmin-1,xmax+1)
            #plt.ylim(ymin-3,ymax+3)
            plt.annotate("Feh = %s"%np.around(feh_best,2), xy=(0.75, 0.95), xycoords='axes fraction')
            plt.xlabel('B-V')
            plt.ylabel('V')
            plt.title('%s'%target_query)
            plt.gca().invert_yaxis()
            plt.savefig('%s/%s_%s_%s_hrdiagram.png'%(directory,target,feh_best,note),dpi=400,bbox_inches='tight')
            plt.close()

            ## Save txt file with best orbit
            msum = mass1_best + mass2_best
            f = open("%s/%s_%s_%s_photometry_masses.txt"%(directory,target,feh_best,note),"w+")
            f.write("# Msum (solar) M1(solar) M2(solar)\r\n")
            f.write("%s %s %s\r\n"%(msum,mass1_best,mass2_best))
            f.close()
        
        else:
            ## Compute at different metallicities
            age_grid = []
            chi2_grid = []
            m1_grid = []
            m2_grid = []

            for ff in feh:

                print("Doing metallicity %s"%ff)
                
                age_step = []
                chi2_step = []
                m1_step = []
                m2_step = []
                
                for aa in tqdm(ages):        
                    mass1_grid = np.linspace(0.5,5,20)
                    mass2_grid = np.linspace(0.5,5,20)
                    try:
                        chi2_grid1,mass1_result,chi2_grid2,mass2_result = mass_search(mass1_grid,mass2_grid,aa,split_mag1,split_mag2,fratio_all_wl,Av,ff)
                        idx_mass1 = np.argmin(chi2_grid1)
                        mass1_guess = mass1_result[idx_mass1]
                        idx_mass2 = np.argmin(chi2_grid2)
                        mass2_guess = mass2_result[idx_mass2]
                        max_mass = find_max_mass(aa,mass1_grid,TOT_Mag_absolute,TOT_Mag_wl,cdiff_all_fit,fratio_all_wl,ff,Av,Teff_primary)

                        params = Parameters()
                        params.add('age',value=aa,vary=False)
                        params.add('feh',value=ff,vary=False)
                        params.add('Av',value=Av,vary=False)
                        params.add('mass1',value=mass1_guess, min=0, max=max_mass)
                        params.add('mass2',value=mass2_guess, min=0, max=max_mass)

                        minner = Minimizer(isochrone_fit, params, fcn_args=(TOT_Mag_absolute,TOT_Mag_wl,cdiff_all_fit,fratio_all_wl,Teff_primary),
                                                    nan_policy='omit')
                        result = minner.minimize()
                        chi2_step.append(result.redchi)
                        age_step.append(result.params['age'].value)
                        m1_step.append(result.params['mass1'].value)
                        m2_step.append(result.params['mass2'].value)
                    except:
                        chi2_step.append(np.nan)
                        age_step.append(aa)
                        m1_step.append(np.nan)
                        m2_step.append(np.nan)

                age_step = np.array(age_step)
                chi2_step = np.array(chi2_step)
                m1_step = np.array(m1_step)
                m2_step = np.array(m2_step)

                if main_sequence=='y':
                    ## only pay attendtion to main sequence ages
                    chi2_grid_ms = chi2_step.copy()
                    chi2_grid_ms[np.where(age_step<7)[0]] = np.nan
                
                age_grid.append(age_step)
                chi2_grid.append(chi2_step)
                m1_grid.append(m1_step)
                m2_grid.append(m2_step)

            idx1 = np.nanargmin(chi2_grid[0])
            idx2 = np.nanargmin(chi2_grid[1])
            idx3 = np.nanargmin(chi2_grid[2])

            age_best = np.array([age_grid[0][idx1],age_grid[1][idx2],age_grid[2][idx3]])
            feh_best = np.array([feh[0],feh[1],feh[2]])
            mass1_best = np.array([m1_grid[0][idx1],m1_grid[1][idx2],m1_grid[2][idx3]])
            mass2_best = np.array([m2_grid[0][idx1],m2_grid[1][idx2],m2_grid[2][idx3]])

            if aavso_method=='y' and aavso_fail=='n':
                if feh_user=='y':
                    directory = '%s/ARMADA_isochrones/%s/fixed/aavso/'%(user_path,target)
                else:
                    directory = '%s/ARMADA_isochrones/%s/varied/aavso/'%(user_path,target)
            else:
                if feh_user=='y':
                    directory = '%s/ARMADA_isochrones/%s/fixed/simbad/'%(user_path,target)
                else:
                    directory = '%s/ARMADA_isochrones/%s/varied/simbad/'%(user_path,target)

            if not os.path.exists(directory):
                print("Creating target directory")
                os.makedirs(directory)

            ## Make plots
            for chi2_step,age_step,ff in zip(chi2_grid,age_grid,feh_best):
                chi2mask = np.isfinite(chi2_step)
                plt.plot(age_step[chi2mask],chi2_step[chi2mask],'--',label=ff)
            plt.title(target_query)
            plt.figtext(0.15, 0.83, 'Mean log age = %s'%np.around(np.mean(age_best),2))
            plt.figtext(0.15, 0.77, 'M1, M2 (M$_{\odot}$) = %s, %s'%(np.around(np.mean(mass1_best),2),np.around(np.mean(mass2_best),2)))
            plt.xlabel('Log Age')
            plt.ylabel('Chi2')
            plt.yscale('log')
            plt.legend(loc=3)
            plt.savefig('%s/%s_%s_chi2grid.png'%(directory,target,note),dpi=400,bbox_inches='tight')
            plt.close()

            ## Make HR diagram plot
            tot_mag_models=[]
            cdiff_models=[]
            for m1,m2,age,ff in zip(mass1_best,mass2_best,age_best,feh_best):
                a1 = tracks.generate(m1, age, ff, return_dict=True)
                a2 = tracks.generate(m2, age, ff, return_dict=True)

                mag1_u = (a1['Mbol'] - bc_grid_U.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
                mag1_b = (a1['Mbol'] - bc_grid_B.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
                mag1_v = (a1['Mbol'] - bc_grid_V.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
                mag1_r = (a1['Mbol'] - bc_grid_R.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
                mag1_i = (a1['Mbol'] - bc_grid_I.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
                mag1_j = (a1['Mbol'] - bc_grid_J.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
                mag1_h = (a1['Mbol'] - bc_grid_H.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]
                mag1_k = (a1['Mbol'] - bc_grid_K.interp([a1['Teff'], a1['logg'], feh_best, Av]))[0]

                mag2_u = (a2['Mbol'] - bc_grid_U.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
                mag2_b = (a2['Mbol'] - bc_grid_B.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
                mag2_v = (a2['Mbol'] - bc_grid_V.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
                mag2_r = (a2['Mbol'] - bc_grid_R.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
                mag2_i = (a2['Mbol'] - bc_grid_I.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
                mag2_j = (a2['Mbol'] - bc_grid_J.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
                mag2_h = (a2['Mbol'] - bc_grid_H.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]
                mag2_k = (a2['Mbol'] - bc_grid_K.interp([a2['Teff'], a2['logg'], feh_best, Av]))[0]

                mag1 = np.array([mag1_u,mag1_b,mag1_v,mag1_r,mag1_i,mag1_j,mag1_h,mag1_k])
                mag2 = np.array([mag2_u,mag2_b,mag2_v,mag2_r,mag2_i,mag2_j,mag2_h,mag2_k])
                tot_mag = -2.5 * np.log10(10 ** (-0.4 * mag1) + 10 ** (-0.4 * mag2))
                cdiff = mag2-mag1
                tot_mag_models.append(tot_mag)
                cdiff_models.append(cdiff)

            xval = np.array([365,445,551,658,806,1250,1653,2200]) 
            for yy,ff in zip(tot_mag_models,feh_best):
                plt.plot(np.sort(xval),yy[np.argsort(xval)],'+--',label=ff)
            for wl,magnitude in zip(xval,TOT_Mag_absolute):
                plt.errorbar(wl,magnitude.nominal_value,yerr=magnitude.std_dev,fmt='.',color='black')
            plt.xlabel('Wavelength')
            plt.ylabel('Magnitude')
            plt.title(target_query)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.savefig('%s/%s_%s_magnitudes.png'%(directory,target,note),dpi=400,bbox_inches='tight')
            plt.close()

            for yy,ff in zip(cdiff_models,feh_best):
                plt.plot(np.sort(xval),yy[np.argsort(xval)],'+--',label=ff)
            for wl,contrast in zip(fratio_all_wl,cdiff_all[:-2]):
                plt.errorbar(wl,contrast.nominal_value,yerr=contrast.std_dev,fmt='.',color='black')
            plt.errorbar(fratio_all_wl[-2],cdiff_all[-2].nominal_value,yerr=cdiff_all[-2].std_dev,fmt='.',color='lightgrey')
            plt.errorbar(fratio_all_wl[-1],cdiff_all[-1].nominal_value,yerr=cdiff_all[-1].std_dev,fmt='.',color='lightgrey')
            plt.xlabel('Wavelength')
            plt.ylabel('Flux difference')
            plt.title(target_query)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.savefig('%s/%s_%s_cdiff.png'%(directory,target,note),dpi=400,bbox_inches='tight')
            plt.close()

            ## Here I am plotting V vs B-V for HR diagram
            ## Using V and B absolute magnitudes, and best-fit model contrasts to split to individual stars
            v1 = -2.5 * log10(10 ** (-TOT_Mag_absolute[2] / 2.5) / (1 + 10 ** (-cdiff_models[1][2][0] / 2.5)))
            v2 = cdiff_models[1][2][0] + v1
            b1 = -2.5 * log10(10 ** (-TOT_Mag_absolute[1] / 2.5) / (1 + 10 ** (-cdiff_models[1][1][0] / 2.5)))
            b2 = cdiff_models[1][1][0] + b1

            ###############
            ## Make an HR diagram plot
            ###############
            for age,ff in zip(age_best,feh_best):
                paramBest = [np.array([age,ff])]
                isoBest = [Mist_iso.isochrone(param[0], param[1]) for param in paramBest]

                V_best = []
                B_best = []
                for i, iso in enumerate(isoBest):
                    Mbol_V = bc_grid_V.interp([iso['Teff'],iso['logg'],ff,Av]).ravel()
                    Mbol_B= bc_grid_B.interp([iso['Teff'],iso['logg'],ff,Av]).ravel()
                    V_best.append(iso['Mbol'] - Mbol_V)
                    B_best.append(iso['Mbol'] - Mbol_B)

                iso_start = 50
                iso_end = 500
                for i, iso in enumerate(isoBest):
                    ## make sure model matches data magnitudes
                    modelx = B_best[i][iso_start:iso_end] - V_best[i][iso_start:iso_end]
                    modely = V_best[i][iso_start:iso_end]
                    plt.plot(modelx, modely)#,label=f"log age = {log_age_start + log_age_size * i} "
                    #plt.annotate("%s"%np.around(age_best[i],2),xy=(modelx.values[-1], modely.values[-1]-0.2))

            ## make plot
            xval1 = ufloat((b1 - v1).nominal_value, (b1+v1).std_dev)  ## component 1
            yval1 = v1 
            xval2 = ufloat((b2 - v2).nominal_value, (b2+v2).std_dev)  ## component 2
            yval2 = v2
            plt.errorbar(xval1.nominal_value, yval1.nominal_value,
                         xerr=xval1.std_dev, yerr=yval1.std_dev,
                         fmt='+',color="darkgreen",markersize=10)
            plt.errorbar(xval2.nominal_value, yval2.nominal_value,
                         xerr=xval2.std_dev, yerr=yval2.std_dev,
                         fmt='+',color="darkgreen",markersize=10)

            xmin = min(xval1,xval2).nominal_value
            xmax = max(xval1,xval2).nominal_value
            ymin = min(yval1,yval2).nominal_value
            ymax = max(yval1,yval2).nominal_value

            #plt.xlim(xmin-1,xmax+1)
            #plt.ylim(ymin-3,ymax+3)
            plt.xlabel('B-V')
            plt.ylabel('V')
            plt.title('%s'%target_query)
            plt.gca().invert_yaxis()
            plt.savefig('%s/%s_%s_hrdiagram.png'%(directory,target,note),dpi=400,bbox_inches='tight')
            plt.close()

            ## Save txt file with best orbit
            msum = mass1_best + mass2_best
            f = open("%s/%s_%s_photometry_masses.txt"%(directory,target,note),"w+")
            f.write("# feh Msum (solar) M1(solar) M2(solar)\r\n")
            f.write("%s %s %s %s\r\n"%(feh_best[0],msum[0],mass1_best[0],mass2_best[0]))
            f.write("%s %s %s %s\r\n"%(feh_best[1],msum[1],mass1_best[1],mass2_best[1]))
            f.write("%s %s %s %s\r\n"%(feh_best[2],msum[2],mass1_best[2],mass2_best[2]))
            f.close()
    
    except:
        print('Target HD%s FAILED'%target_hd)
        failed_targets.append(target_hd)

print('List of failed targets = ')
print(failed_targets)