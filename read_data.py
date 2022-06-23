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

def read_data2(file,weight):
    p=[]
    degrees=[]  #text file is in degrees
    t=[]
    t_date=[]
    error_maj=[]
    error_min=[]
    error_pa=[]
    
    p2=[]
    degrees2=[]  #text file is in degrees
    error_maj2=[]
    error_min2=[]
    error_pa2=[]
    
    for line in file.readlines():
        if line.startswith('#'):
            continue
        p.append(float(line.split()[2]))
        degrees.append(float(line.split()[3]))
        p2.append(float(line.split()[7]))
        degrees2.append(float(line.split()[8]))
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
            error_maj2.append(float(line.split()[9]))
            error_min2.append(float(line.split()[10]))
            epa2 = float(line.split()[11])
            #if epa>0:
            #    epa=90-epa
            #else:
            #    epa=abs(epa)+90
            error_pa.append(epa)
            error_pa2.append(epa2)
    file.close()
    degrees=np.array(degrees)
    p=np.array(p)
    t=np.array(t)
    theta=np.array(degrees)*(np.pi/180)
    
    degrees2=np.array(degrees2)
    p2=np.array(p2)
    theta2=np.array(degrees2)*(np.pi/180)

    error_maj=weight*np.asarray(error_maj)
    error_min=weight*np.asarray(error_min)
    error_deg=np.asarray(error_pa)
    error_pa=error_deg*np.pi/180.
    
    error_maj2=weight*np.asarray(error_maj2)
    error_min2=weight*np.asarray(error_min2)
    error_deg2=np.asarray(error_pa2)
    error_pa2=error_deg2*np.pi/180.
    
    return(t,p,theta,error_maj,error_min,error_pa,error_deg,p2,theta2,error_maj2,error_min2,error_pa2,error_deg2)

def read_data3(file,weight):
    p=[]
    degrees=[]  #text file is in degrees
    t=[]
    t_date=[]
    error_maj=[]
    error_min=[]
    error_pa=[]
    
    p2=[]
    degrees2=[]  #text file is in degrees
    error_maj2=[]
    error_min2=[]
    error_pa2=[]
    
    p3=[]
    degrees3=[]  #text file is in degrees
    error_maj3=[]
    error_min3=[]
    error_pa3=[]
    
    for line in file.readlines():
        if line.startswith('#'):
            continue
        p.append(float(line.split()[2]))
        degrees.append(float(line.split()[3]))
        p2.append(float(line.split()[7]))
        degrees2.append(float(line.split()[8]))
        p3.append(float(line.split()[12]))
        degrees3.append(float(line.split()[13]))
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
            error_maj2.append(float(line.split()[9]))
            error_min2.append(float(line.split()[10]))
            epa2 = float(line.split()[11])
            error_maj3.append(float(line.split()[14]))
            error_min3.append(float(line.split()[15]))
            epa3 = float(line.split()[16])
            #if epa>0:
            #    epa=90-epa
            #else:
            #    epa=abs(epa)+90
            error_pa.append(epa)
            error_pa2.append(epa2)
            error_pa3.append(epa3)
    file.close()
    degrees=np.array(degrees)
    p=np.array(p)
    t=np.array(t)
    theta=np.array(degrees)*(np.pi/180)
    
    degrees2=np.array(degrees2)
    p2=np.array(p2)
    theta2=np.array(degrees2)*(np.pi/180)
    
    degrees3=np.array(degrees3)
    p3=np.array(p3)
    theta3=np.array(degrees3)*(np.pi/180)

    error_maj=weight*np.asarray(error_maj)
    error_min=weight*np.asarray(error_min)
    error_deg=np.asarray(error_pa)
    error_pa=error_deg*np.pi/180.
    
    error_maj2=weight*np.asarray(error_maj2)
    error_min2=weight*np.asarray(error_min2)
    error_deg2=np.asarray(error_pa2)
    error_pa2=error_deg2*np.pi/180.
    
    error_maj3=weight*np.asarray(error_maj3)
    error_min3=weight*np.asarray(error_min3)
    error_deg3=np.asarray(error_pa3)
    error_pa3=error_deg3*np.pi/180.
    
    return(t,p,theta,error_maj,error_min,error_pa,error_deg,p2,theta2,error_maj2,error_min2,error_pa2,error_deg2,p3,theta3,error_maj3,error_min3,error_pa3,error_deg3)

def read_rv(file,weight=1):
    rv=[]
    t=[]
    err=[]
    datetype = input('MJD? (y,[n]): ')
    for line in file.readlines():
        if line.startswith('#'):
            continue
        rv.append(float(line.split()[1]))
        if datetype=='y':
            t.append(float(line.split()[0])) ##txt file in MJD
        else:
            t.append(float(line.split()[0])-0.5) ##txt file in HJD
        try:
            err.append(float(line.split()[2]))
        except:
            err.append(1)
    file.close()
    rv=np.array(rv)
    t=np.array(t)
    err=weight*np.asarray(err)    
    return(t,rv,err)

def read_wds(file,weight,dtype):
    p_wds=[]
    degrees_wds=[]  #text file is in degrees
    t_wds=[]
    error_maj_wds=[]
    error_min_wds=[]
    error_pa_wds=[]
    for line in file.readlines():
        if line.startswith('#'):
            continue
        #if float(line.split()[0])<1985:
        #    continue
        #if line.split()[4]=='.':
        #    continue
    
        #print(line.split('\s+')[0][111:113])
        if dtype=='':
            p_wds.append(float(line.split()[3]))
            degrees_wds.append(float(line.split()[1]))
            t_wds.append(float(line.split()[0]))

            #error_maj_wds.append(1)
            #error_min_wds.append(1)
            #error_pa_wds.append(0)

            if line.split('\s+')[0][111]=='S':
                error_maj_wds.append(1)
                error_min_wds.append(1)
                error_pa_wds.append(0)
            else:
                error_maj_wds.append(5)
                error_min_wds.append(5)
                error_pa_wds.append(0)
        else:
            if line.split('\s+')[0][111]==dtype:
                p_wds.append(float(line.split()[3]))
                degrees_wds.append(float(line.split()[1]))
                t_wds.append(float(line.split()[0]))
    
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
    
    return(t_wds,p_wds,theta_wds,error_maj_wds,error_min_wds,error_pa_wds,error_deg_wds)

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
    
            inc = float(line.split('|')[17])
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
        
            omega = float(line.split('|')[27])
        
            bigomega = float(line.split('|')[19])
    
    print('--------------------------')
    print('a(mas),P(yr),e,i(deg),omega(deg),bigomega(deg),T(mjd)')
    print(a,P/365,e,inc,omega,bigomega,T)
    print('--------------------------')
    
    return(a,P,e,inc,omega,bigomega,T)