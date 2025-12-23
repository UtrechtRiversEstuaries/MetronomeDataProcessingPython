'''
Function to translate a NetCDF file with into a GeoTIFF format, suitable for GIS applications.

Python script written by Eise W. Nota (finalized MAY 2025) 
'''

# Importing packages
import rasterio
import netCDF4 as nc
import numpy as np

def NetCDF2geoTIFF(ncFile,rootAndWriteFolder,tifFile,relPerc,origin,gridRes):
    '''
    Parameters
    ----------
    ncFile              : String of input NetCDF DEM
    rootAndWriteFolder  : Folder to save the GeoTIFF DEM
    tifFile             : String with GeoTIFF DEM output file name
    relPerc             : Relevant percentile, by default 'z50'
    origin              : Origin (X,Y) of the desired geoTIFF, default [0,0]
    gridRes             : Grid resolution in mm
    '''
    
    #%% Load the NetCDF demFile
    demNetCDF = nc.Dataset(rootAndWriteFolder + ncFile)
    
    #%% Extract relevant information
    # Load the Z data
    zValues = np.array(demNetCDF.get_variables_by_attributes(name='Z-axis')[0])
    
    # How many percentiles does the file contain?
    pcs = np.array(demNetCDF.get_variables_by_attributes(name='Z percentiles')[0])
    # Find the right one
    i=0
    # Loop trough percentiles
    for pc in pcs:
        if pc == relPerc:
            zValues = zValues[:,:,i].reshape((zValues.shape[0],zValues.shape[1]))
        i +=1
    
    # We can close the file now
    demNetCDF.close()
    
    # Store data to 32-bit, required for rasterio GeoTIFF
    zValues = np.float32(zValues)
    # Flip zValues
    zValues = np.flip(zValues,0)
    
    #%% Now store into GeoTIFF
    # Define coordinate transform
    transform = rasterio.transform.from_origin(
        west=origin[0], 
        north=origin[1], 
        xsize=gridRes/1000, # convert to m 
        ysize=gridRes/1000 # convert to m 
    )
    
    # Define new tiff
    new_tiff = rasterio.open(
        rootAndWriteFolder + tifFile, 'w', 
        driver = 'GTiff',
        height = zValues.shape[0],
        width = zValues.shape[1],
        count = 1,
        nodata = -9999,
        dtype = zValues.dtype,
        transform = transform,
        compress='lzw'
    )
    
    # Write tiff
    new_tiff.write(zValues,1)
    
    # Close tiff
    new_tiff.close()
    
    print(tifFile + ' has been saved and stored as GeoTiff file')
    