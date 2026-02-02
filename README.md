# MetronomeDataProcessingPython

Files for processing raw data from the tidal tilting flume 'The Metronome' in Python

The main branch has the following sub-branches, with the following files:

## Overhead\_scripts

* ### Overhead\_Orthomosaics\_Metashape\_Timelapse.py

Computes orthomosaics from overhead series through base model alignment and combines them into a single timelapse video

* ### Overhead\_Single\_Orthomosaic\_Metashape.py

Computes a single orthomosaics from a specified overhead series (Experiment ...; cycle .....)

* ### RandomForest2WaterDepthMaps.py

Computeswater depth maps (or derived elevation maps) from overhead orthomosaics through the Random Forest models and combines them into a single timelapse video

* ### Function\_Metashape\_Overhead.py

Ccontains the function that translate and Overhead series into an orthomosaic

* ### Function\_colour\_correction\_overhead.py

Contains the function that debayers raw overhead imagery and to conduct color space transformations from RGB to various color spaces

* ### Function\_create\_timelapse\_overhead.py

Contains the function that creates the timelapse of Overhead series orthomosaics

* ### Function\_create\_timelapse\_overhead\_depthMaps.py

Contains the function that creates the timelapse of Overhead series orthomosaics including water depth maps (or derived elevation maps) through the Random Forest models

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

Apply Gaussian smoothening to input Overhead orthomosaics

* ### 2\_Overhead\_Orthomosaics\_RGB2ColorSpace.py

Transform RGB orthomosaics into various colorspaces

* ### 3\_ConstructRandomForestModels.py

Tran the Random Forest (dry/wet) classifier and (water depth at wet) regressor models on the different dye concentration Training datasets

* ### 4\_RandomForestValidataionCalibrationSetup.py

Conduct the validations on the Random Forest models using the Training and Validation datasets

* ### 5\_RandomForestCrossModelValidation.py

Conduct the cross-model validations of the Random Forest models using the Training and Validation datasets of different dye concentrations

* ### 6\_RandomForestValidationExperimentalDatasets.py

Conduct the validations of the Random Forest models using the Experimental datasets



# Software requirements

* python 3.10 (not tested for later versions). Environment (yml) file included in the main branch
* Agisoft Metashape v. 1.8.5 (tested for v. 2.0.4)

## NetCDF file structure

The .nc (NetCDF) files for the DEMs can be read using scripting software such as MATLAB, Python or R. Important is to know how these NetCDF files are constructed. They contain the following variables:
X-axis: (start,end,step) of the entire X-axis. In numpy you can extract the whole axis as follows: np.arange(xAxis\[0],xAxis\[1],xAxis\[2])
Y-axis: (start,end,step) of the entire Y-axis. In numpy you can extract the whole axis as follows: np.arange(yAxis\[0],yAxis\[1],yAxis\[2])
Z percentiles: The Z-percentile values that are stored in the NetCDF file. This is by default only the median (i.e. \[50])
Z-axis: List of Gridded Z-data (meshgrid described in X-axis and Y-axis), stored per percentile. As the default is only median, this list contains by default only one grid.

# Available datasets

* https://public.yoda.uu.nl/geo/UU01/SGM22N.html

Data supplement to "Remote sensing of a gantry-equipped facility: optimizing accuracy by integrating SfM photogrammetry and laserscan computer graphics through fixed base model geometry" ( http://ssrn.com/abstract=5649495).

