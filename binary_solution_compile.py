######################################################################
## Tyler Gardner
## 
## Get orbit solutions and compile to single txt file
######################################################################

import os

dir = input('dir of fits: ')
target_id = input('target_id: ')
triple = input('triple? (y/n) ')
quad = input('quad? (y/n)')
subtract = input('subtract err pa from 360deg? (y/[n])')

if triple=='y':
    f = open("/Users/tgardner/ARMADA_orbits/%s_triple.txt"%target_id,"w+")
    f.write("# date mjd sep12 pa12 err_maj12 err_min12 err_pa12 sep13 pa13 err_maj13 err_min13 err_pa13\r\n")
else:
    if quad=='y':
        f = open("/Users/tgardner/ARMADA_orbits/%s_quad.txt"%target_id,"w+")
        f.write("# date mjd sep12 pa12 err_maj12 err_min12 err_pa12 sep13 pa13 err_maj13 err_min13 err_pa13 sep14 pa14 err_maj14 err_min14 err_pa14\r\n")
    else:
        f = open("/Users/tgardner/ARMADA_orbits/%s_chi2err.txt"%target_id,"w+")
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
                if subtract=='y':
                    err_pa12 = 360 - float(line.split()[6])
                else:
                    err_pa12 = line.split()[6]
                sep13 = line.split()[7]
                pa13 = line.split()[8]
                err_maj13 = line.split()[9]
                err_min13 = line.split()[10]
                if subtract=='y':
                    err_pa13 = 360 - float(line.split()[11])
                else:
                    err_pa13 = line.split()[11]
                f.write("%s %s %s %s %s %s %s %s %s %s %s %s\r\n"%(date,mjd,sep12,pa12,err_maj12,err_min12,err_pa12,sep13,pa13,err_maj13,err_min13,err_pa13))
    else:
        if quad=='y':
            if file.endswith("quad.txt"):
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
                    if subtract=='y':
                        err_pa12 = 360 - float(line.split()[6])
                    else:
                        err_pa12 = line.split()[6]
                    sep13 = line.split()[7]
                    pa13 = line.split()[8]
                    err_maj13 = line.split()[9]
                    err_min13 = line.split()[10]
                    if subtract=='y':
                        err_pa13 = 360 - float(line.split()[11])
                    else:
                        err_pa13 = line.split()[11]
                    sep14 = line.split()[12]
                    pa14 = line.split()[13]
                    err_maj14 = line.split()[14]
                    err_min14 = line.split()[15]
                    if subtract=='y':
                        err_pa14 = 360 - float(line.split()[16])
                    else:
                        err_pa14 = line.split()[16]
                    f.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\r\n"%(date,mjd,sep12,pa12,err_maj12,err_min12,err_pa12,sep13,pa13,err_maj13,err_min13,err_pa13,sep14,pa14,err_maj14,err_min14,err_pa14))
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
                    if subtract=='y':
                        err_pa12 = 360 - float(line.split()[6])
                    else:
                        err_pa12 = line.split()[6]
                    f.write("%s %s %s %s %s %s %s\r\n"%(date,mjd,sep,pa,err_maj,err_min,err_pa12))
f.close()

if triple == 'y' or quad == 'y':
    print('done......')
else:
    f = open("/Users/tgardner/ARMADA_orbits/%s_bootstrap.txt"%target_id,"w+")
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
                if subtract=='y':
                    err_pa12 = 360 - float(line.split()[6])
                else:
                    err_pa12 = line.split()[6]
                f.write("%s %s %s %s %s %s %s\r\n"%(date,mjd,sep,pa,err_maj,err_min,err_pa))
    f.close()

