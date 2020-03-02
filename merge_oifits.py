######################################################################
## Tyler Gardner
##
## Merge oifits files into single file
## Using JDM's IDL routine
##
######################################################################

import numpy as np
import matplotlib.pyplot as plt
eachindex = lambda lst: range(len(lst))
import os
import numpy_indexed as npi
import matplotlib as mpl
from astropy.io import fits
import pidly

## select night, target
dir=input('Path to oifits directory: ')
target_id=input('Target ID (e.g. HD_206901): ')
date=input('Date (e.g. 2018Jul19): ')

## select destination
dir2=input('Path to destination: ')

## Using oifits library
filelist = []

for file in os.listdir(dir):
    if file.endswith("_oifits.fits") or file.endswith("_viscal.fits") or file.endswith("_singlescivis.fits"):
        filename = os.path.join(dir, file)
        oifile = fits.open(filename)
        obj = oifile[0].header['OBJECT']
        print(str(obj))
        if str(target_id)==str(obj):
            filelist.append(filename)

### merge the files - using John's IDL routine
#outfile = os.path.join(dir2, '%s_%s.fits'%(target_id,date))
#infiles = filelist
#
#idl = pidly.IDL('idl /Users/tgardne/mirc6b_idl_startup.pro')
#idl.pro('merge_oidata',outfile=outfile,infiles=infiles)
#idl.close()

## merge the files - using John Young C tools
info_list = ['oifits-merge','%s/%s_%s.fits'%(dir2,target_id,date)]+filelist
print(' '.join(info_list))
os.system(' '.join(info_list))