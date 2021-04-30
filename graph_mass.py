import csv
import matplotlib.pyplot as plt
import numpy as np

file = "/Users/adam.scovera/Documents/Astro/BEPResearch_Data/ARMADA_targets_updated.csv"

mdynHipEst = []
mdynHipSp = []
mdynHipSpec = []
mdynGaiaEst = []
mdynGaiaSp = []
mdynGaiaSpec = []
estHip = []
spHip = []
specHip = []
estGaia = []
spGaia = []
specGaia = []

with open(file) as csvfile:
    read = csv.reader(csvfile)
    for row in read:
        
        '''
        if (row[0] != "HIP" and row[2] != "1" and row[1] != ""):
            if (row[11] != "--" and row[17] != "0" and row[17] != ""):
                mdynHipEst.append(float(row[15]))
                estHip.append(float(row[17]))
            if (row[11] != "--" and row[18] != "0" and row[18] != "" and float(row[15]) < 30):
                mdynHipSp.append(float(row[15]))
                spHip.append(float(row[18]))
            if (row[11] != "--" and row[19] != "0" and row[19] != "#VALUE!" and float(row[15]) < 30):
                mdynHipSpec.append(float(row[15]))
                specHip.append(float(row[19]))
                
                
            if (row[12] != "--" and row[17] != "0" and row[17] != ""):
                mdynGaiaEst.append(float(row[16]))
                estGaia.append(float(row[17]))
            if (row[12] != "--" and row[18] != "0" and row[18] != "" and float(row[16]) < 30):
                mdynGaiaSp.append(float(row[16]))
                spGaia.append(float(row[18]))
            if (row[12] != "--" and row[19] != "0" and row[19] != "#VALUE!" and float(row[16]) < 30):
                mdynGaiaSpec.append(float(row[16]))
                specGaia.append(float(row[19]))
            '''
           

maxX = max(mdynHipSpec)
maxY = max(specHip)
plt.plot([0, 1.1*max(maxX,maxY)], [0, 1.1*max(maxX,maxY)], color = 'r')
plt.scatter(mdynHipSpec, specHip)
plt.axis([0, 1.1*maxX, 0, 1.1*maxY])
plt.title("Mass Estimates Using Hipparcos VS. Speckle Estimate")
plt.xlabel("mDyn")
plt.ylabel("Speckle Estimate")
#plt.legend()
plt.savefig('speckle.png')
plt.show()

'''
maxX = max(max(mdynHipEst),max(mdynGaiaEst))
maxY = max(max(estHip),max(estGaia))
plt.plot([0, 1.1*max(maxX,maxY)], [0, 1.1*max(maxX,maxY)], color = 'r')
plt.scatter(mdynHipEst, estHip, label="Hipparcos")
plt.scatter(mdynGaiaEst, estGaia, label="Gaia")
plt.axis([0, 1.1*maxX, 0, 1.1*maxY])
plt.title("Hipparcos Estimates VS. MSum Estimate")
plt.xlabel("mDyn")
plt.ylabel("MSum Estimate")
plt.legend()
plt.savefig('msum.png')
plt.show()

maxX = max(max(mdynHipSp),max(mdynGaiaSp))
maxY = max(max(spHip),max(spGaia))
plt.plot([0, 1.1*max(maxX,maxY)], [0, 1.1*max(maxX,maxY)], color = 'r')
plt.scatter(mdynHipSp, spHip, label="Hipparcos")
plt.scatter(mdynGaiaSp, spGaia, label="Gaia")
plt.axis([0, 1.1*maxX, 0, 1.1*maxY])
plt.title("Hipparcos & Gaia Estimates VS. Spec Type Estimate")
plt.xlabel("mDyn")
plt.ylabel("Spec Type Estimate")
plt.legend()
plt.savefig('spec_type.png')
plt.show()

maxX = max(max(mdynHipSpec),max(mdynGaiaSpec))
maxY = max(max(specHip),max(specGaia))
plt.plot([0, 1.1*max(maxX,maxY)], [0, 1.1*max(maxX,maxY)], color = 'r')
plt.scatter(mdynHipSpec, specHip, label="Hipparcos")
plt.scatter(mdynGaiaSpec, specGaia, label="Gaia")
plt.axis([0, 1.1*maxX, 0, 1.1*maxY])
plt.title("Hipparcos & Gaia Estimates VS. Speckle Estimate")
plt.xlabel("mDyn")
plt.ylabel("Speckle Estimate")
plt.legend()
plt.savefig('speckle.png')
plt.show()

with open(file) as csvfile:
    read = csv.reader(csvfile)
    for row in read:
        if (row[15] != "Mdyn Hipparcos" and row[15] != "#DIV/0!" and row[15] != "" and row[1] != "--" and row[2] != "1" and float(row[1]) < 75 and row[17] != "0"):
            if (row[17] != "" and row[17] != "0" and row[15] != "#VALUE!" and row[15] != "#DIV/0" and float(row[15]) > 0.5):
                mdynHip.append(float(row[15]))
                estimateHip.append(float(row[17]))
            if (row[17] != "" and row[16] != "0" and row[16] != "#VALUE!" and row[16] != "#DIV/0!" and float(row[16]) > 0.5 and float(row[16]) < 50):
                mdynGaia.append(float(row[16]))
                estimateGaia.append(float(row[17]))
                
plt.plot([0,12],[0,12],color = 'r')
plt.scatter(mdynHip,estimateHip,label='Hipparcos')
plt.scatter(mdynGaia,estimateGaia,label='Gaia')
plt.legend()
plt.axis([0,12,0,12])
plt.title("All Data")
plt.xlabel("mDyn")
plt.ylabel("Estimates")
plt.savefig('all.png')
plt.show()

plt.plot([0,12],[0,12], color = 'r')
plt.scatter(estimateHip,mdynHip,label='Hipparcos')
plt.scatter(estimateGaia,mdynGaia,label='Gaia')
plt.axis([0,12,0,12])
plt.legend()
plt.title("All Data - Axes Flipped")
plt.xlabel("mDyn")
plt.ylabel("Estimates")
plt.savefig('all-axes-flipped.png')
plt.show()
for i in range(len(mdynSp)-1):
    if mdynSp[i] > 150: # For some reason the last valueof this array does not want to be accessed here
        mdynSp.pop(i)
        spType.pop(i)
        
plt.plot([0,41],[0,41],color='r')
plt.scatter(mdynEst,estimate,label='Est')
plt.scatter(mdynSp,spType,label='Sp')
plt.legend()
plt.title("1 Outlier Revomed")
plt.xlabel("mDyn")
plt.ylabel("Estimates")
plt.savefig('1large_removed.png')
plt.show()

for i in range(len(mdynSp)-1):
    if mdynSp[i] > 40: # For some reason the last valueof this array does not want to be accessed here
        mdynSp.pop(i)
        spType.pop(i)
for i in range(len(mdynEst)-1):
    if mdynEst[i] > 40:
        mdynEst.pop(i)
        estimate.pop(i)

plt.plot([0,20],[0,20],color='r')
plt.scatter(mdynEst,estimate,label='Est')
plt.scatter(mdynSp,spType,label='Sp')
plt.legend()
plt.title("2 Outliers Removed")
plt.xlabel("mDyn")
plt.ylabel("Estimates")
plt.savefig('2large_removed.png')
plt.show()

for i in range(len(mdynSp)-1):
    if mdynSp[i] > 15: # For some reason the last valueof this array does not want to be accessed here
        mdynSp.pop(i)
        spType.pop(i)

for i in range(len(mdynEst)):
    if mdynEst[i] > 15:
        mdynEst.pop(i)
        estimate.pop(i)

plt.plot([0,13],[0,13],color='r')
plt.scatter(mdynEst,estimate,label='Est')
plt.scatter(mdynSp,spType,label='Sp')
plt.legend()
plt.title("All Outliers Removed")
plt.xlabel("mDyn")
plt.ylabel("Estimates")
plt.savefig('all_ol_removed.png')
plt.show()

plt.plot([0,17],[0,17],color='r')
plt.scatter(estimate,mdynEst,label='Est')
plt.scatter(spType,mdynSp,label='Sp')
plt.legend()
plt.ylabel("mDyn")
plt.xlabel("Estimates")
plt.show()
'''
