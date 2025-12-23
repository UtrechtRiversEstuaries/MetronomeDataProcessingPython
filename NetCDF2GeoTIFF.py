"""
This script translates a NetCDF DEM into a GeoTIFF suitable for GIS applications

@author: Eise Nota
FINALIZED MAY 2025
"""

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

# %% importing libraries
from IPython import get_ipython
import os

# Set base directory of the network drive
pwd = r"\\..."

# Import functions
os.chdir(pwd + r"\python\Laserscan_scripts\Functions_laserscan") # Setting the functions directory
from Function_NetCDF2GeoTIFF import NetCDF2geoTIFF

#%% Set the specific DEM information you want to translate
Expnr = '054'              # As string
pilot = 1                  # In integer, 0 if no pilot
start_cycle = '00000'      # As string, e.g. 00000 or 00000_dry
end_cycle = '01000'    # String of cycle where you want to end # End with complete numbers as in folder structure
baseModel = '11'           # As string
gridRes = 5                # mm
relPerc = 'z50'            # Relevant percentile; leave at 'z50'
origin = [0,3]             # Origin (X,Y) of the desired geoTIFF at west-north , default [0,3]

#%% Determine the files and directories
# Is it a pilot?    
if pilot == 0:
    rootAndWriteFolder = pwd + '\\experiments\\Exp' + Expnr + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(gridRes) + 'mm\\'
else: 
    rootAndWriteFolder = pwd + '\\experiments\\Exp' + Expnr + '\\Pilot' + str(pilot) + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(gridRes) + 'mm\\'   

#%% Loading input directories
# Only .nc files are relevant
relfiles = [file for file in os.listdir(rootAndWriteFolder) if file.endswith('.nc')]

chosen_files = [] # Relevant folders will be stored here
      
# Create dictionary of relevant folders per camera based on start_cycle and end_cycle
for i in relfiles:
    # Only append if the folders are between start_cycle and end_cycle
    # First i should be in startcycle
    if start_cycle in i: # First start_cycle is discovered
        chosen_files.append(i)
        # If start_cycle and end_cycle are the same, break the for-loop
        if start_cycle == end_cycle:
            break
    # Then subsequent folders are added
    elif len(chosen_files) > 0 and end_cycle not in i:
        chosen_files.append(i)
    # Until finally the end_cycle is appended
    elif end_cycle in i: # First start_cycle is discovered
        chosen_files.append(i)
        # End the for loop
        break

#%% Now loop over the chosen files
for ncFile in chosen_files:
    # Determine the name of the tif file
    tifFile = ncFile.replace("nc","TIF") 
    # Translate the dem into GeoTIFF
    NetCDF2geoTIFF(ncFile,rootAndWriteFolder,tifFile,relPerc,origin,gridRes)