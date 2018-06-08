""" 
Filename: load.py
Author: Mathew Bub
Last Revision Date: 2018-06-08

This module contains memory-efficient tools for loading the Gaia DR2 RV
catalogue. These tools are derived from those found in the gaia_tools package.
"""
import os
import numpy as np
import astropy.io.fits as pyfits
from gaia_tools.load import path, download

def gaiarv(fields=None, parallax_cut=True):
    file_paths = path.gaiarvPath()
    if not np.all([os.path.exists(file_path) for file_path in file_paths]):
        download.gaiarv()
    
    if fields is None:
        data = np.lib.recfunctions.stack_arrays(
            [pyfits.getdata(file_path, ext=1) for file_path in file_paths],
            autoconvert=True)
    else:  
        if parallax_cut and 'parallax_over_error' not in fields:
            fields.append('parallax_over_error')
        data = np.lib.recfunctions.stack_arrays(
            [np.array(pyfits.getdata(file_path, ext=1))[fields] for file_path 
             in file_paths], autoconvert=True)
    
    if parallax_cut:
        data = data[data['parallax_over_error'] > 5]
        
    return data

