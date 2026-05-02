# MetronomeDataProcessingPython

Files for processing raw data from the tidal tilting flume 'The Metronome' in Python

The main branch has the following sub-branches, with the following files:


## python\_demos\_for\_loading\_dataset

Contains demo scripts to load data from the dataset paper (to be submitted)

* ### expMetaData.xlsx

Contains all relevant metadata for each of the experiments, required as input for the python demo scripts below. Each scripts follows the Yoda repository directory structure

* ### load\_DEMs.py

Demo script to load laserscan DEMs

* ### load\_Orthomosaics\_1Hz.py

Demo script to load 1Hz Overhead orthomosaics

* ### load\_Orthomosaics\_DSLR.py

Demo script to load DSLR orthomosaics

* ### load\_Orthomosaics\_PIV.py

Demo script to load PIV Overhead orthomosaics

* ### load\_Orthomosaics\_interval.py

Demo script to load interval Overhead orthomosaics

* ### load\_WaterDepthMaps\_1Hz.py

Demo script to load water depth maps derived from 1Hz Overhead orthomosaics, including masks

* ### load\_WaterDepthMaps\_interval.py

Demo script to load water depth maps derived from interval Overhead orthomosaics, including masks


## Overhead\_scripts

* ### Overhead\_Orthomosaics\_Metashape\_Timelapse.py

Computes orthomosaics from overhead series through base model alignment and combines them into a single timelapse video

* ### Overhead\_Single\_Orthomosaic\_Metashape.py

Computes a single orthomosaics from a specified overhead series (Experiment ...; cycle .....)

* ### RandomForest2WaterDepthMaps.py

Computes water depth maps (or derived elevation) from overhead orthomosaics through the random forest models and combines them into a single timelapse video

* ### Function\_Metashape\_Overhead.py

Contains the function that translate and Overhead series into an orthomosaic

* ### Function\_colour\_correction\_overhead.py

Contains the function that debayers raw overhead imagery and the functions that transform RGB orthomosaics into various color spaces

* ### Function\_create\_timelapse\_overhead.py

Contains the function that creates the timelapse of Overhead series orthomosaics

* ### Function\_create\_timelapse\_overhead\_depthMaps.py

Contains the function that creates the timelapse of Overhead series water depth maps (or derived elevation) through Random Forest models

* ### Function\_flatten\_list.py

Contains the function that flattens arrays of Overhead data require to create an orthomosaic

## Laserscan\_scripts

* ### Laserscan2DEM.py

Computes DEMs from laserscans through Base model geometry-based laserscan method and plots them in three possible ways.

* ### NetCDF2DoD.py

Computes DoDs from two different DEMs. It is compatible for different DEM methods

* ### NetCDF2GeoTIFF.py

Translates our NetCDF format to a GeoTIFF compatible for GIS applications

* ### Function\_Laserscan\_DEM.py

Contains the functions that translate gridded laserscan data into DEMs through Base model geometry-based laserscan method

* ### Function\_NetCDF2GeoTIFF.py

Contains the function that translates our NetCDF format to a GeoTIFF compatible for GIS applications

* ### Function\_undistort.py

Contains the functions that apply distortion correction to raw laserscan data

* ### NetCDF2Plots.py

This file creates basic plots from Metronome DEMs in NetCDF format.

## DSLR\_scripts

* ### DSLR\_Orthomosaics\_Metashapte.py

Computes orthomosaics from DSLR surveys through base model alignment

* ### DSLR\_Orthomosaics\_DEMs\_Metashape.py

Computes orthomosaics and DEMs for Individually aligned and Base model-aligned DSLR surveys

* ### Function\_GeoTIFF2NetCDF.py

Contains the function that translates an Agisoft GeoTIFF into the NetCDF format we use

* ### Function\_Metashape\_DSLR.py

Contains the functions that translate DSLR surveys into orthomosaics and DEMs

## Water-level-processing

* ### WaterLevelProcessing.py

This script processes the water level measurements from the Metronome.

## RandomForestTrainingValidation

* ### 1\_Overhead\_Orthomosaics\_Smooth.py

Apply Gaussian smoothing of Overhead orthomosaics

* ### 2\_Overhead\_Orthomosaics\_RGB2ColorSpace.py

Transform smoothened RGB orhtomosaics into the relevant colorspaces

* ### 3\_ConstructRandomForestModels.py

Train the Random Forest Classifier and Regressor models with the Validation datasets of the various dye concentrations.

* ### 4\_RandomForestValidationCalibrationSetup.py

Validate the Random Forest models with the Training and Validation dastsets of their respective dye concentrations

* ### 5\_RandomForestCrossModelValidation.py

Apply cross-model validation of the Random Forest models with the Training and Validation datssets of the other dye concentrations

* ### 6\_RandomForestValidationExperimentalDatasets.py

Validate the Random Forest models with the Experimental datssets

# Software requirements

* python 3.10 (not tested for later versions). Environment (yml) file included in the main branch
* Agisoft Metashape v. 1.8.5 (tested for v. 2.0.4)

## NetCDF file structure

The .nc (NetCDF) files for the DEMs, masks and water depth maps can be read using scripting software such as MATLAB, Python or R. Important is to know how these NetCDF files are constructed. The Python demos provide code to loas these .nc files. They contain the following variables:

X-axis: (start,end,step) of the entire X-axis. In numpy you can extract the whole axis as follows: np.arange(xAxis[0],xAxis[1],xAxis[2])

Y-axis: (start,end,step) of the entire Y-axis. In numpy you can extract the whole axis as follows: np.arange(yAxis[0],yAxis[1],yAxis[2])

Z percentiles (only for DEMs): The Z-percentile values that are stored in the NetCDF file. This is by default only the median (i.e. [50])

Z-axis (for DEMs): List of Gridded elevation data (meshgrid described in X-axis and Y-axis), stored per percentile. As the default is only median, this list contains by default only one grid

Z-axis (for water depth maps): List of Gridded water depth data (meshgrid described in X-axis and Y-axis). The dry cells in these water depth maps are represented by 0

mask (only for masks): binary map yielding 1 for the relevant area (active area of the estuaries) and 0 for the masked out data (inactive bed + location bridge + delta + sea)

# Available datasets

* https://public.yoda.uu.nl/geo/UU01/SGM22N.html

Data supplement to "Remote sensing of a gantry-equipped facility: optimizing accuracy by integrating SfM photogrammetry and laserscan computer graphics through fixed base model geometry" (https://doi.org/10.1016/j.jag.2026.105098).



* https://public.yoda.uu.nl/geo/UU01/2XBVKK.html

Data supplement to "Quantitative water depth determination in large experimental timeseries through combining spectrophotometry and machine learning" (https://doi.org/10.22541/essoar.177082648.88604562/v1).


* DATASET PAPER TO BE SUBMITTED AND SOON AVAILABLE ON YODA



