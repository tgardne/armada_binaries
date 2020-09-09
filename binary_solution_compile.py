######################################################################
## Tyler Gardner
## 
## Get orbit solutions and compile to single txt file
######################################################################

import os

dir = input('dir of fits: ')
target_id = input('target_id: ')
triple = input('triple? (y/n) ')

if triple=='y':
    f = open("/Users/tgardne/ARMADA_orbits/%s_triple.txt"%target_id,"w+")
    f.write("# date mjd sep12 pa12 err_maj12 err_min12 err_pa12 sep13 pa13 err_maj13 err_min13 err_pa13\r\n")
else:
    f = open("/Users/tgardne/ARMADA_orbits/%s_chi2err.txt"%target_id,"w+")
    f.write("# date mjd sep pa err_maj err_min err_pa\r\n")

for file in os.listdir(dir):
    if triple=='y':
        if file.endswith("triple.txt"):
            filename = open(os.path.join(dir, file))
            for line in filename.readlines():
                if line.startswith('#'):
                    continue
                date = line.split()[0]
                mjd = line.split()[1]
                sep12 = line.split()[2]
                pa12 = line.split()[3]
                err_maj12 = line.split()[4]
                err_min12 = line.split()[5]
                err_pa12 = line.split()[6]
                sep13 = line.split()[7]
                pa13 = line.split()[8]
                err_maj13 = line.split()[9]
                err_min13 = line.split()[10]
                err_pa13 = line.split()[11]
                f.write("%s %s %s %s %s %s %s %s %s %s %s %s\r\n"%(date,mjd,sep12,pa12,err_maj12,err_min12,err_pa12,sep13,pa13,err_maj13,err_min13,err_pa13))
    else:
        if file.endswith("chi2err.txt"):
            filename = open(os.path.join(dir, file))
            for line in filename.readlines():
                if line.startswith('#'):
                    continue
                date = line.split()[0]
                mjd = line.split()[1]
                sep = line.split()[2]
                pa = line.split()[3]
                err_maj = line.split()[4]
                err_min = line.split()[5]
                err_pa = line.split()[6]
                f.write("%s %s %s %s %s %s %s\r\n"%(date,mjd,sep,pa,err_maj,err_min,err_pa))
f.close()

if triple == 'y':
    print('done......')
else:
    f = open("/Users/tgardne/ARMADA_orbits/%s_bootstrap.txt"%target_id,"w+")
    f.write("# date mjd sep pa err_maj err_min err_pa\r\n")

    for file in os.listdir(dir):
        if file.endswith("bootstrap.txt"):
            filename = open(os.path.join(dir, file))
            for line in filename.readlines():
                if line.startswith('#'):
                    continue
                date = line.split()[0]
                mjd = line.split()[1]
                sep = line.split()[2]
                pa = line.split()[3]
                err_maj = line.split()[4]
                err_min = line.split()[5]
                err_pa = line.split()[6]
                f.write("%s %s %s %s %s %s %s\r\n"%(date,mjd,sep,pa,err_maj,err_min,err_pa))
    f.close()

