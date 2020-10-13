##########################
## Read in astrometric data from
## ARMADA
## WDS
## or orbital elements from orb6
######################### 

import numpy as np

def read_data(file,weight):
    p=[]
    degrees=[]  #text file is in degrees
    t=[]
    t_date=[]
    error_maj=[]
    error_min=[]
    error_pa=[]
    for line in file.readlines():
        if line.startswith('#'):
            continue
        p.append(float(line.split()[2]))
        degrees.append(float(line.split()[3]))
        t.append(float(line.split()[1]))
        t_date.append(line.split()[0])
        if line.split()[4]=='--':
            error_maj.append(0.1)
            error_min.append(0.1)
            error_pa.append(0)
        else:
            error_maj.append(float(line.split()[4]))
            error_min.append(float(line.split()[5]))
            epa = float(line.split()[6])
            #if epa>0:
            #    epa=90-epa
            #else:
            #    epa=abs(epa)+90
            error_pa.append(epa)
    file.close()
    degrees=np.array(degrees)
    p=np.array(p)
    t=np.array(t)
    theta=np.array(degrees)*(np.pi/180)

    error_maj=weight*np.asarray(error_maj)
    error_min=weight*np.asarray(error_min)

    error_deg=np.asarray(error_pa)
    error_pa=error_deg*np.pi/180.
    
    return(t,p,theta,error_maj,error_min,error_pa,error_deg)

def read_wds(file,weight,fileTypes):
    p_wds=[]
    degrees_wds=[]  #text file is in degrees
    t_wds=[]
    error_maj_wds=[]
    error_min_wds=[]
    error_pa_wds=[]
    type=[]
            
    print()
    print('Select which collection types to use.')
    print('Put nothing to use all.')
    print('To select only some, list the ones you want with spaces between them. e.g. \'M S I\' for the types of M, S, and I.')
    print('Coillection types for this target:')
    print(fileTypes)
    dtypes = input('Input: ')
    
    for line in file.readlines():
        if line.startswith('#'):
            continue

        #if float(line.split()[0])<1985:
        #    continue
        #if line.split()[4]=='.':
        #    continue
        #print(line.split('\s+')[0][111:113])

        if dtypes=='':
            p_wds.append(float(line.split()[3]))
            degrees_wds.append(float(line.split()[1]))
            t_wds.append(float(line.split()[0]))
            type.append(line.split('\s+')[0][111])

            if line.split('\s+')[0][111]=='S':
                error_maj_wds.append(1)
                error_min_wds.append(1)
                error_pa_wds.append(0)
            else:
                error_maj_wds.append(5)
                error_min_wds.append(5)
                error_pa_wds.append(0)
        else:
            types = dtypes.split()
            for t in types:
                if line.split('\s+')[0][111]==t:
                    p_wds.append(float(line.split()[3]))
                    degrees_wds.append(float(line.split()[1]))
                    t_wds.append(float(line.split()[0]))
                    type.append(t)
    
                    error_maj_wds.append(1)
                    error_min_wds.append(1)
                    error_pa_wds.append(0)
                else:
                    continue
    file.close()

    degrees_wds=np.asarray(degrees_wds)
    p_wds=np.asarray(p_wds)*1000
    t_wds=np.asarray(t_wds)*365.2422-678940.37
    theta_wds=np.asarray(degrees_wds)*(np.pi/180)

    error_maj_wds=weight*np.asarray(error_maj_wds)
    error_min_wds=weight*np.asarray(error_min_wds)

    error_deg_wds=np.asarray(error_pa_wds)
    error_pa_wds=error_deg_wds*np.pi/180.
    
    return(t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds,type)

def read_orb6(target,file):

    orb6=open(file)

    for line in orb6.readlines():
    
        if line.startswith('#'):
            continue

        if target == str(line.split('|')[6]):
        
            print('found target!')
    
            if str(line.split('|')[15])=='a':
                a = float(line.split('|')[14])*1000
            else:
                if str(line.split('|')[15])=='m' or str(line.split('|')[15])=='M':
                    a = float(line.split('|')[14])
                else:
                    if str(line.split('|')[15])=='u':
                        a = float(line.split('|')[14])/1000
    
            inc = float(line.split('|')[17])*np.pi/180
            e = float(line.split('|')[25])
    
            if str(line.split('|')[12])=='m':
                P = float(line.split('|')[11])/60/24
            else:
                if str(line.split('|')[12])=='h':
                    P = float(line.split('|')[11])/24
                else:
                    if str(line.split('|')[12])=='d':
                        P = float(line.split('|')[11])
                    else:
                        if str(line.split('|')[12])=='y':
                            P = float(line.split('|')[11])*365
                        else:
                            if str(line.split('|')[12])=='c':
                                P = float(line.split('|')[11])*365*1000
        
            if str(line.split('|')[23])=='m':
                T = float(line.split('|')[22])
            else:
                if str(line.split('|')[23])=='d':
                    T = float(line.split('|')[22])-2400000.5
                else:
                    if str(line.split('|')[23])=='y':
                        T = float(line.split('|')[22])*365.2422-678940.37
                    else:
                        if str(line.split('|')[23])=='c':
                            T = float(line.split('|')[22])*365.2422-678940.37 #NEED TO CHECK !!!!!
                            print('CHECK THIS ONE!!')
        
            omega = float(line.split('|')[27])*np.pi/180
        
            bigomega = float(line.split('|')[19])*np.pi/180
    
    print('--------------------------')
    print('a(mas),P(yr),e,i(deg),omega(deg),bigomega(deg),T(mjd)')
    print(a,P/365,e,inc*180/np.pi,omega*180/np.pi,bigomega*180/np.pi,T)
    print('--------------------------')
    
    return(a,P,e,inc,omega,bigomega,T)
    
def get_types(file):
    targetTypes = []
    for line in file.readlines():
        if line.startswith('#'):
            continue
        t = line.split('\s+')[0][111]
        if t in targetTypes:
            continue
        else:
            targetTypes.append(t)
    file.close()
    return targetTypes
