# MetronomeDataProcessingPython
Files for processing raw data from the tidal tilting flume 'The Metronome' in Python

## DEM processing
* ### NetCDF2Plots.py

This file creates basic plots from Metronome DEMs in NetCDF format

## Water level data processing
* ### WaterLevelProcessing.py

This script processes the water level measurements from the Metronome.

The raw water level data is read from the corresponding directory and then:
* filtered for outliers
* averaged over the usually three measured cycles
* sorted to align all data equally with the tilt
* smoothed
* put in relation to the still water level
* saved as netCDF-file
* plotted

#### Note: 
* The data should be saved in subfolders thst are named e.g. 
     '1500cycles_tilting' and there should be at least one measurement of the 
     still water level
* The final data does not contain the y-position of the sensors for the 
     different measurements. Make sure to have this data elsewhere.
* The folder 'water_level' should already exist in the directory 
     'processed_data' of the corresponding experiment, otherwise the processed
     data is not saved.
* The raw data files should be named as follows: 
     <cyclesnumber>_<x-metres>.<x-decimetres>_<tilting or still>_<automatic time stamp>_LT.s<sensornumber>.csv

## Other files
The other scripts necessary to process the raw Metronome data do not yet exist in Python. Some are well written in Matlab, others need a major overhaul when they are implemented in Python
