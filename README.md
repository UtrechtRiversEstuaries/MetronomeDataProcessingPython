# MetronomeDataProcessingPython
Files for processing raw data from the tidal tilting flume 'The Metronome' in Python

The main branch has the following sub-branches, with the following files:

## Overhead_scripts
* ### Overhead_Orthomosaics_Metashape_Timelapse.py
Computes orthomosaics from overhead series through base model alignment and combines them into a single timelapse video
* ### Overhead_Single_Orthomosaic_Metashape.py
Computes a single orthomosaics from a specified overhead series (Experiment ...; cycle .....)
* ### Function_Metashape_Overhead.py
Ccontains the function that translate and Overhead series into an orthomosaic
* ### Function_colour_correction_overhead.py
Contains the function that debayers raw overhead imagery
* ### Function_create_timelapse_overhead.py
Contains the function that creates the timelapse of Overhead series orthomosaics
* ### Function_flatten_list.py
Contains the function that flattens arrays of Overhead data require to create an orthomosaic

## Laserscan_scripts
* ### Laserscan2DEM.py
Computes DEMs from laserscans through Base model geometry-based laserscan method and plots them in three possible ways.
* ### NetCDF2DoD.py
Computes DoDs from two different DEMs. It is compatible for different DEM methods
* ### NetCDF2GeoTIFF.py
Translates our NetCDF format to a GeoTIFF compatible for GIS applications
* ### Function_Laserscan_DEM.py
Contains the functions that translate gridded laserscan data into DEMs through Base model geometry-based laserscan method
* ### Function_NetCDF2GeoTIFF.py
Contains the function that translates our NetCDF format to a GeoTIFF compatible for GIS applications
* ### Function_undistort.py
Contains the functions that apply distortion correction to raw laserscan data
* ### NetCDF2Plots.py
This file creates basic plots from Metronome DEMs in NetCDF format.

## DSLR_scripts
* ### DSLR_Orthomosaics_Metashapte.py
Computes orthomosaics from DSLR surveys through base model alignment
* ### DSLR_Orthomosaics_DEMs_Metashape.py
Computes orthomosaics and DEMs for Individually aligned and Base model-aligned DSLR surveys
* ### Function_GeoTIFF2NetCDF.py
Contains the function that translates an Agisoft GeoTIFF into the NetCDF format we use
* ### Function_Metashape_DSLR.py
Contains the functions that translate DSLR surveys into orthomosaics and DEMs

## Water-level-processing
* ### WaterLevelProcessing.py
This script processes the water level measurements from the Metronome.

# Software requirements

* python 3.10 (not tested for later versions)
* Agisoft Metashape v. 1.8.5 (tested for v. 2.0.4)

## NetCDF file structure
The .nc (NetCDF) files for the DEMs can be read using scripting software such as MATLAB, Python or R. Important is to know how these NetCDF files are constructed. They contain the following variables:
X-axis: (start,end,step) of the entire X-axis. In numpy you can extract the whole axis as follows: np.arange(xAxis[0],xAxis[1],xAxis[2])
Y-axis: (start,end,step) of the entire Y-axis. In numpy you can extract the whole axis as follows: np.arange(yAxis[0],yAxis[1],yAxis[2])
Z percentiles: The Z-percentile values that are stored in the NetCDF file. This is by default only the median (i.e. [50])
Z-axis: List of Gridded Z-data (meshgrid described in X-axis and Y-axis), stored per percentile. As the default is only median, this list contains by default only one grid.

