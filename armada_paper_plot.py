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

###########################################
## SETUP PATHS
###########################################

path = '/Users/tgardne/ARMADA_orbits'
path_etalon = '/Users/tgardne/etalon_epochs/etalon_fits/etalon_factors_fit.txt'
path_wds = '/Users/tgardne/wds_targets'
path_orb6 = '/Users/tgardne/catalogs/orb6orbits.sql.txt'

###########################################
## Specify Targets
###########################################

targets = ['2772','6456','10453','11031','17094','17904','27176','29316',
           '31093','34319','36058','37269','38545','38769','41040','43358',
           '43525','46273','47105','49643','60107','64235','75974','78316','82446',
           '87652','87822','107259','112846','114993','118889',
           '127726','128415','129246_J','133955','137909','140159','140436',
           '144892','145589','148283','153370','154569','156190','158140',
           '160935','163346','166045','178475','179950','185404','189037','189340','195206',
           '198183','201038','217676','217782','224512']
date = ''

remake_plots = input('Remake the orbit plots? (y / [n]): ')

if remake_plots == 'y':

    failed_targets = []
    for target_hd in targets:
        try:
            query = Simbad.query_objectids('HD %s'%target_hd)
            for item in query:
                if 'HIP' in item[0]:
                    target = item[0].split()[1]
                    print('HIP %s'%target)
                if 'WDS' in item[0]:
                    target_wds = item[0][5:15]
                    print('WDS %s'%target_wds)

            try:
                print(target_wds)
            except:
                print('No WDS number queried')
                target_wds = input('Enter WDS: ')

            directory='%s/HD%s_chi2err/'%(path,target_hd)

            ###########################################
            ## Read in ARMADA data
            ###########################################

            file=open('%s/HD_%s_armada.txt'%(directory,target_hd))
            weight=1

            t,p,theta,error_maj,error_min,error_pa,error_deg = read_data(file,weight)
            file.close()

            ## FIXME: make it easier to choose vlti data
            #vlti = input('Add indices for vlti (y/n)? ')
            vlti = 'n'
            if vlti=='y':
                vlti_idx = input('enter indices (e.g. 1 2 3): ').split(' ')
                vlti_idx = np.array([int(i) for i in vlti_idx])
            else:
                vlti_idx = []

            xpos=p*np.sin(theta)
            ypos=p*np.cos(theta)

            vlti_mask = np.ones(len(t),dtype=bool)
            vlti_mask[vlti_idx] = False

            ###########################################
            ## Read in WDS data - and plot to check
            ###########################################
            #input_wds = input('Include WDS? (y/n): ')
            input_wds = 'y'
            if input_wds=='y':
                file=open('%s/HD_%s_wds.txt'%(directory,target_hd))
                t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds = read_data(file,1)
                file.close()
                xpos_wds=p_wds*np.sin(theta_wds)
                ypos_wds=p_wds*np.cos(theta_wds)
            else:
                t_wds = np.array([np.nan])
                p_wds = np.array([np.nan])
                theta_wds = np.array([np.nan])
                error_maj_wds = np.array([np.nan])
                error_min_wds = np.array([np.nan])
                error_pa_wds = np.array([np.nan])
                error_deg_wds = np.array([np.nan])
                xpos_wds=p_wds*np.sin(theta_wds)
                ypos_wds=p_wds*np.cos(theta_wds)
                print('NO WDS DATA')

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

            vlti_mask_all = np.ones(len(t_all),dtype=bool)
            vlti_mask_all[vlti_idx] = False

            ## Get best orbit
            orbit_file = open('%s/%s__orbit_ls.txt'%(directory,target_hd))
            lines = orbit_file.readlines()
            P_best = float(lines[2].split()[0])
            a_best = float(lines[2].split()[1])
            e_best = float(lines[2].split()[2])
            i_best = float(lines[2].split()[3])
            w_best = float(lines[2].split()[4])
            bigw_best = float(lines[2].split()[5])
            T_best = float(lines[2].split()[6])
            orbit_file.close()

            ## Grab MCMC chains
            chains = np.load('%s/HD%s__chains.npy'%(directory,target_hd))
            w_chain = chains[:,0]
            bigw_chain = chains[:,1]
            inc_chain = chains[:,2]
            e_chain = chains[:,3]
            a_chain = chains[:,4]
            P_chain = chains[:,5]
            T_chain = chains[:,6]

            w_mcmc = np.std(chains[:,0])
            bigw_mcmc = np.std(chains[:,1])
            inc_mcmc = np.std(chains[:,2])
            e_mcmc = np.std(chains[:,3])
            a_mcmc = np.std(chains[:,4])
            P_mcmc = np.std(chains[:,5])
            T_mcmc = np.std(chains[:,6])

            ## select random orbits from chains
            idx = np.random.randint(0,len(chains),size=100)
            fig,ax=plt.subplots()

            for orbit in idx:
                tmod = np.linspace(min(t_all),min(t_all)+2*P_best,1000)
                ra,dec,rapoints,decpoints = orbit_model(a_chain[orbit],e_chain[orbit],inc_chain[orbit],
                                                            w_chain[orbit],bigw_chain[orbit],P_chain[orbit],
                                                            T_chain[orbit],t_all,tmod)
                ax.plot(ra, dec, '--',color='lightgrey')

            tmod = np.linspace(min(t_all),min(t_all)+P_best,1000)
            ra,dec,rapoints,decpoints = orbit_model(a_best,e_best,i_best,
                                                    w_best,bigw_best,P_best,
                                                    T_best,t_all,tmod)
            ax.plot(xpos_all[len(xpos):], ypos_all[len(xpos):], 'o', label='WDS',color='grey')
            if len(vlti_idx)>0:
                ax.plot(xpos[vlti_idx],ypos[vlti_idx],'*', label='ARMADA-VLTI',color='red')
                ax.plot(xpos[vlti_mask],ypos[vlti_mask],'+', label='ARMADA-CHARA',color='blue')
            else:
                ax.plot(xpos,ypos,'+', label='ARMADA',color='red')
            ax.plot(0,0,'*',color='g')
            ax.plot(ra, dec, '--',color='g')
            #plot lines from data to best fit orbit
            i=0
            while i<len(decpoints):
                x=[xpos_all[i],rapoints[i]]
                y=[ypos_all[i],decpoints[i]]
                ax.plot(x,y,color="black")
                i+=1
            #ax.set_xlabel('dRA (mas)')
            #ax.set_ylabel('dDEC (mas)')
            ax.invert_xaxis()
            ax.axis('equal')
            ax.set_title('HD%s Outer Orbit'%target_hd)
            ax.legend()
            plt.savefig('%s/HD%s_%s_mcmc_paper.png'%(directory,target_hd,date),dpi=400,bbox_inches='tight')
            plt.close()

            print('--'*10)
        except:
            failed_targets.append(target_hd)
            print('--'*10)
            continue

    print("Failed Targets =  ")
    print(failed_targets)

pdf_plots = []
plot_idx = 0
for target_hd in targets:
    directory='%s/HD%s_chi2err/'%(path,target_hd)
    pdf_plots.append('%s/HD%s__mcmc_paper.png'%(directory,target_hd))

plot_number = len(pdf_plots)
print('Number of plots = ', plot_number)

number_pages = int(plot_number/16)
number_leftover = plot_number%16

for page_idx in np.arange(number_pages):
    
    img_idx = np.arange(page_idx*16,page_idx*16+16,1)

    fig,ax = plt.subplots(4,4,figsize=(20,20))
    ax = ax.flatten()

    for idx,i in enumerate(img_idx):
        img = plt.imread(pdf_plots[i])
        ax[idx].imshow(img)
        ax[idx].axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)

    fig.text(0.5, 0.04, '$\Delta$ RA (mas)', ha='center')
    fig.text(0, 0.5, '$\Delta$ DEC (mas)', va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig('/Users/tgardne/Desktop/paper_plot_%s.pdf'%page_idx,bbox_inches='tight')

## Last page -- the leftovers

page_idx = number_pages
img_idx = np.arange(page_idx*16,page_idx*16+number_leftover,1)  

fig,ax = plt.subplots(4,4,figsize=(20,20))
ax = ax.flatten()
for idx,i in enumerate(img_idx):
    img = plt.imread(pdf_plots[i])
    ax[idx].imshow(img)
    ax[idx].axis('off')
    plt.subplots_adjust(wspace=0,hspace=0)

fig.text(0.5, 0.04, '$\Delta$ RA (mas)', ha='center')
fig.text(0, 0.5, '$\Delta$ DEC (mas)', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig('/Users/tgardne/Desktop/orbits_paper_plot_%s.pdf'%page_idx,bbox_inches='tight')