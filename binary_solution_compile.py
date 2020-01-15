######################################################################
## Tyler Gardner
## 
## Get orbit solutions and compile to single txt file
######################################################################

import os

dir = input('dir of fits: ')
target_id = input('target_id: ')

f = open("/Users/tgardne/ARMADA_orbits/%s.txt"%target_id,"w+")
f.write("# date mjd sep pa err_maj err_min err_pa\r\n")

for file in os.listdir(dir):
    if file.endswith(".txt"):

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

