## For a binary star with UD for each
##
## Inputs: u,v,dRA,dDE,flux ratio,UD1,UD2
## NOTE: u,v should be in units of wavelength (i.e. u/lambda)
## NOTE: dRA,dDE,UD1,UD2 are in units of mas
##
## Outputs: complex visibility

import numpy as np
import itertools
from scipy import special
from itertools import combinations

class binary_disks_vector:

    def mas2rad(self,mas):
        rad=mas/1000.
        rad=rad/3600.
        rad=rad*np.pi/180.
        return(rad)

    def binary(self,u,v,sep,pa,ratio,ud1,ud2,bw):
        
        delta_dec=self.mas2rad(sep*np.sin((pa+90)*np.pi/180))
        delta_ra=-self.mas2rad(sep*np.cos((pa+90)*np.pi/180))
        
        secondary_flux = 1/(1+ratio)
        primary_flux = 1-secondary_flux
        
        diameter1 = self.mas2rad(ud1)
        diameter2 = self.mas2rad(ud2)
        x1 = np.pi*diameter1*np.sqrt(u**2+v**2)
        x2 = np.pi*diameter2*np.sqrt(u**2+v**2)
    
        ## bessel 1st order
        x1=x1+1e-15
        x2=x2+1e-15
        f1 = 2*special.jn(1,x1)/x1
        f2 = 2*special.jn(1,x2)/x2

        ## bandwidth smearing
        vis_bw = np.sinc(bw*(u*delta_ra+v*delta_dec))
        complex_vis = primary_flux*f1 + vis_bw*secondary_flux*f2*np.exp(-2*np.pi*1j*(u*delta_ra+v*delta_dec))
                         
        return(complex_vis)
    
    ## This one just takes d_ra,d_dec instead:
    def binary2(self,u,v,ra,dec,ratio,ud1,ud2,bw):
        
        delta_dec=self.mas2rad(dec)
        delta_ra=self.mas2rad(ra)
        
        secondary_flux = 1/(1+ratio)
        primary_flux = 1-secondary_flux
        
        diameter1 = self.mas2rad(ud1)
        diameter2 = self.mas2rad(ud2)
        x1 = np.pi*diameter1*np.sqrt(u**2+v**2)
        x2 = np.pi*diameter2*np.sqrt(u**2+v**2)
        
        ## bessel 1st order
        x1=x1+1e-15
        x2=x2+1e-15
        f1 = 2*special.jn(1,x1)/x1
        f2 = 2*special.jn(1,x2)/x2

        ## bandwidth smearing
        vis_bw = np.sinc(bw*(u*delta_ra+v*delta_dec))
        complex_vis = primary_flux*f1 + vis_bw*secondary_flux*f2*np.exp(-2*np.pi*1j*(u*delta_ra+v*delta_dec))
                
        return(complex_vis)

    ## for an elongated disk for central star
    def binary_ellipse(self,u,v,sep,pa,ratio,ud1,fd,axis_ratio,inc,angle,ud2,bw):
        
        delta_dec=self.mas2rad(sep*np.sin((pa+90)*np.pi/180))
        delta_ra=-self.mas2rad(sep*np.cos((pa+90)*np.pi/180))
        
        secondary_flux = 1/(1+ratio)
        primary_flux = 1-secondary_flux

        fd = 1/(1+fd)
        fs = 1 - fd
        
        ## primary disk
        minor = self.mas2rad(2*ud1/(1+axis_ratio))
        major = minor*axis_ratio
        diameter1 = self.mas2rad(ud1)
        angle = 90-angle
        uprime = u*np.cos(angle*np.pi/180)+v*np.sin(angle*np.pi/180)
        vprime = v*np.cos(angle*np.pi/180)*np.cos(inc*np.pi/180)-u*np.sin(angle*np.pi/180)*np.cos(inc*np.pi/180)
        uprime = uprime*major/diameter1
        vprime = vprime*minor/diameter1
        xd = np.pi*diameter1*np.sqrt(uprime**2+vprime**2)

        ## companion as UD
        diameter2 = self.mas2rad(ud2)
        x1 = np.pi*diameter2*np.sqrt(u**2+v**2)
        x2 = np.pi*diameter2*np.sqrt(u**2+v**2)
    
        ## bessel 1st order
        x1=x1+1e-15
        x2=x2+1e-15
        xd=xd+1e-15
        f1 = 2*special.jn(1,x1)/x1
        f2 = 2*special.jn(1,x2)/x2
        f3 = 2*special.jn(1,xd)/xd

        ## bandwidth smearing
        vis_bw = np.sinc(bw*(u*delta_ra+v*delta_dec))
        complex_vis = fs*f1 + fd*f3 + vis_bw*secondary_flux*f2*np.exp(-2*np.pi*1j*(u*delta_ra+v*delta_dec))
                         
        return(complex_vis)

    def triple(self,u,v,sep12,pa12,sep13,pa13,ratio12,ratio13,ud1,ud2,ud3,bw):
    
        f1 = 1/(1+1/ratio12+1/ratio13)
        f2 = 1/(1+ratio12+ratio12/ratio13)
        f3 = 1-f1-f2
    
        delta_dec12=self.mas2rad(sep12*np.sin((pa12+90)*np.pi/180))
        delta_ra12=-self.mas2rad(sep12*np.cos((pa12+90)*np.pi/180))
    
        delta_dec13=self.mas2rad(sep13*np.sin((pa13+90)*np.pi/180))
        delta_ra13=-self.mas2rad(sep13*np.cos((pa13+90)*np.pi/180))
    
        diameter1 = self.mas2rad(ud1)
        diameter2 = self.mas2rad(ud2)
        diameter3 = self.mas2rad(ud3)
        x1 = np.pi*diameter1*np.sqrt(u**2+v**2)
        x2 = np.pi*diameter2*np.sqrt(u**2+v**2)
        x3 = np.pi*diameter3*np.sqrt(u**2+v**2)
    
        ## bessel 1st order
        v1 = 2*special.jn(1,x1)/x1
        v2 = 2*special.jn(1,x2)/x2
        v3 = 2*special.jn(1,x3)/x3

        ## bandwidth smearing
        vis_bw12 = np.sinc(bw*(u*delta_ra12+v*delta_dec12))
        vis_bw13 = np.sinc(bw*(u*delta_ra13+v*delta_dec13))
    
        complex_vis = (f1*v1 + vis_bw12*f2*v2*np.exp(-2*np.pi*1j*(u*delta_ra12+v*delta_dec12)) + vis_bw13*f3*v3*np.exp(-2*np.pi*1j*(u*delta_ra13+v*delta_dec13)))/(f1+f2+f3)
                   
        return(complex_vis)

    def triple2(self,u,v,delta_ra12,delta_dec12,delta_ra13,delta_dec13,ratio12,ratio13,ud1,ud2,ud3,bw):
    
        f1 = 1/(1+1/ratio12+1/ratio13)
        f2 = 1/(1+ratio12+ratio12/ratio13)
        f3 = 1-f1-f2

        delta_dec12=self.mas2rad(delta_dec12)
        delta_ra12=self.mas2rad(delta_ra12)
    
        delta_dec13=self.mas2rad(delta_dec13)
        delta_ra13=self.mas2rad(delta_ra13)
    
        diameter1 = self.mas2rad(ud1)
        diameter2 = self.mas2rad(ud2)
        diameter3 = self.mas2rad(ud3)
        x1 = np.pi*diameter1*np.sqrt(u**2+v**2)
        x2 = np.pi*diameter2*np.sqrt(u**2+v**2)
        x3 = np.pi*diameter3*np.sqrt(u**2+v**2)
    
        ## bessel 1st order
        v1 = 2*special.jn(1,x1)/x1
        v2 = 2*special.jn(1,x2)/x2
        v3 = 2*special.jn(1,x3)/x3

        ## bandwidth smearing
        vis_bw12 = np.sinc(bw*(u*delta_ra12+v*delta_dec12))
        vis_bw13 = np.sinc(bw*(u*delta_ra13+v*delta_dec13))
    
        complex_vis = (f1*v1 + vis_bw12*f2*v2*np.exp(-2*np.pi*1j*(u*delta_ra12+v*delta_dec12)) + vis_bw13*f3*v3*np.exp(-2*np.pi*1j*(u*delta_ra13+v*delta_dec13)))/(f1+f2+f3)
                   
        return(complex_vis)
