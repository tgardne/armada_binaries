## Inputs: hour angle, declination, chara xyz positions
## Outputs: u,v,baseline lengths, baseline angles

import numpy as np
import itertools
from itertools import combinations

class uv_calc:
    
    def geocnvrt(self,h0,i0,arr0):
        #calculate Geocentric X-Y-Z from East-West-Up coordinate
        b=[[-np.sin(h0),-np.sin(i0)*np.cos(h0),np.cos(i0)*np.cos(h0)],
           [np.cos(h0),-np.sin(i0)*np.sin(h0),np.cos(i0)*np.sin(h0)],
           [0.0,np.cos(i0),np.sin(i0)]]
        arr=np.dot(arr0,np.array(b))
        return(arr)

    def geocoord(self,xyz,lat,long):
        lat=lat/180*np.pi
        long=long/180*np.pi
        new_xyz=[]
        for item in xyz:
            new_arr=self.geocnvrt(long,lat,item)
            new_xyz.append(new_arr)
        new_xyz=np.array(new_xyz)
        return(new_xyz)

    def calc_uv(self,ha,baseline_vector,lat,dec):
        b_east=baseline_vector[0]
        b_north=baseline_vector[1]
        b_up=baseline_vector[2]
    
        ## convert to radians
        ha_rads=ha*np.pi/12
        dec_rads=dec*np.pi/180
        lat_rads=lat*np.pi/180
    
        ## convert from (east,north,up) to (x,y,z)
        bx=-np.sin(lat_rads)*b_north+np.cos(lat_rads)*b_up
        by=b_east
        bz=np.cos(lat_rads)*b_north+np.sin(lat_rads)*b_up
    
        cosza=np.sin(lat_rads)*np.sin(dec_rads)+np.cos(lat_rads)*np.cos(dec_rads)*np.cos(ha_rads)
        za=np.arccos(cosza)*180/np.pi
    
        ## convert bx,by,bz to (u,v,w)
        u=np.sin(ha_rads)*bx+np.cos(ha_rads)*by
        v=-np.sin(dec_rads)*np.cos(ha_rads)*bx+np.sin(dec_rads)*np.sin(ha_rads)*by+np.cos(dec_rads)*bz
        w=np.cos(dec_rads)*np.cos(ha_rads)*bx-np.cos(dec_rads)*np.sin(ha_rads)*by+np.sin(dec_rads)*bz
    
        return([u,v])

    def grab_chara_uv(self,ha,dec,tels):
        ## get telescope positions from chara_xyz.txt file
        ## closing triangle used is given by 'tels' input (e.g. tels=['S1', 'E1', 'W1'])
        
        #print('tels check')
        #print(tels)
        #print('-----------')
        
        telescope_positions=[]
        
        #for line in file.readlines()[1:]:
        #    if line[0:2]==tels[0] or line[0:2]==tels[1] or line[0:2]==tels[2]:
        #        tel_vals=[]
        #        tel_vals.append(float(line.split()[1])/1e6)
        #        tel_vals.append(float(line.split()[2])/1e6)
        #        tel_vals.append(float(line.split()[3])/1e6)
        #        telescope_positions.append(tel_vals)
               
        for item in tels:
            file=open('/Users/tgardne/binary_interferometry/chara_xyz.txt')
            for line in file.readlines()[1:]:
                if line[0:2]==item:
                    tel_vals=[]
                    tel_vals.append(float(line.split()[1])/1e6)
                    tel_vals.append(float(line.split()[2])/1e6)
                    tel_vals.append(float(line.split()[3])/1e6)
               
                    telescope_positions.append(tel_vals)
            file.close()
        
        file.close()
        telescope_positions=np.array(telescope_positions)

        chara_lat=34.2259 #deg
        chara_long=-118.0571 #deg
    
        ## Convert Tel positions to XYZ
        xyz=self.geocoord(telescope_positions,chara_lat,chara_long)
    
        u=[]
        v=[]
        ## Go through each baseline
        for item in [(0,1),(1,2),(2,0)]:
            bv=[-telescope_positions[item[0]][0]+telescope_positions[item[1]][0],-telescope_positions[item[0]][1]+telescope_positions[item[1]][1],-telescope_positions[item[0]][2]+telescope_positions[item[1]][2]]
            if bv==[0.0, 0.0, 0.0]:
                continue
            else:
                #print(bv)
                point=self.calc_uv(ha,bv,chara_lat,dec)
                u.append(point[0])
                v.append(point[1])
    
        u=np.array(u)
        v=np.array(v)
    
        b_lengths=np.sqrt(u**2+v**2)
        b_angles=np.arctan2(v,u)*180./np.pi
        
        #print('uv check')
        #print(u,v)
        #print('-----------')
    
        return(u,v,b_lengths,b_angles)
