"""
This script translates a NetCDF DEM into a GeoTIFF suitable for GIS applications

Note: 'Matlab' method equals the linearly processed laserscan DEM.

@author: Eise Nota
FINALIZED MAY 2025
"""

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

# %% importing libraries
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from IPython import get_ipython
import os
import csv
import pandas as pd

# Set base directory of the network drive
pwd = r"\\..."

#%% Set the DEMs that you want to use to make the DoDs from
# Because DEMs can be constructed through various methods, we need to specify all necessary information for both DEMs
# First set information that must be the same for both DEMs
gridRes = 5                 # mm
statistics = 0              # Set to 1 if you want to calculate the statistics of the DoD
crossSection = 0            # Set to 1 if you want to calculate a cross section of the DoD
if crossSection == 1: # Set X-coordinate you want to do (for calibration data, this is at 0.5 cm, which would be 0.5025 for a grid size of 5 mm)
    crossSectionX = 0.5025  # Possibly

#%% DEM0 specific information
Expnr0 = '054'              # As string
pilot0 = 1                  # In integer, 0 if no pilot
cycle0 = '00000'            # As string, e.g. 00000 or 00000_dry
method0 = 0                 # 0 == 'LaserScan'; 1 == 'Agisoft'; 2 == 'Matlab'

# For Python-processed DEMs, a baseModel is used as well
# The default is '11'
if method0 == 0 or method0 == 1:
    baseModel0 = '11'       # As string, False if not used in Agisoft DEM

# For LaserScan-processed DEMs, elevation data is stored in percentiles (by default we use the z50)
if method0 == 0 or method0 == 2:
    plotvar0 = 'z50'        # As string starting with z (e.g.'z50' or 'avg') NOTE: 'avg' only for method 0
    
# For Agisoft-processed DEMs, a depth map accuracy is used (by default we use 'medium')
if method0 == 1:
    accuracy0 = 'ultra'    # 'lowest'; 'low'; 'medium'; 'high'; 'ultra'

#%% DEM1 specific information
Expnr1 = '054'              # As string
pilot1 = 1                  # In integer, 0 if no pilot
cycle1 = '00000'            # As string, e.g. 00000 or 00000_dry
method1 = 1                 # 0 == 'LaserScan'; 1 == 'Agisoft'; 2 == 'Matlab'

# For Python-processed DEMs, a baseModel is used as well
# The default is '11'
if method1 == 0 or method1 == 1:
    baseModel1 = '11'       # As string, False if not used in Agisoft DEM

# For LaserScan-processed DEMs, elevation data is stored in percentiles (by default we use the z50)
if method1 == 0 or method1 == 2:
    plotvar1 = 'z50'        # As string starting with z (e.g.'z50' or 'avg') NOTE: 'avg' only for method 0
    
# For Agisoft-processed DEMs, a depth map accuracy is used (by default we use 'medium')
if method1 == 1:
    accuracy1 = 'ultra'    # 'lowest'; 'low'; 'medium'; 'high'; 'ultra'
    
#%% Specify the figure parameters
figwidth = 30               # inches
labelsize = 15              # for labels x-, y-axes and colorbar
ticksize = 12.5             # for label ticks
titlesize = 20              # for title
vmax = 1               # Maximum DoD value to be plotted in cm
vmin = -1                # Minimum DoD value to be plotted in cm

#%% Determine the files and directories
# Start with DEM0
if method0 == 0:            # LaserScan
    # Is it a pilot?    
    if int(pilot0) == 0:
        directory0 = pwd + '\\Exp' + Expnr0 + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel0 + '\\Res' + str(gridRes) + 'mm\\'
        # DEM name depends on whether variable is statistical elevation or avg
        if plotvar0 == 'avg':
            DEM0 = 'Exp' + Expnr0 + '_' + cycle0 + '_LaserScan_avg_DEM.nc'
        else:
            DEM0 = 'Exp' + Expnr0 + '_' + cycle0 + '_LaserScan_DEM.nc'
        writeFolder = pwd + '\\Exp' + Expnr0 + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_BaseModel' + baseModel0 + '_res' + str(gridRes) + 'mm\\'
    else: 
        directory0 = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel0 + '\\Res' + str(gridRes) + 'mm\\'
        # DEM name depends on whether variable is statistical elevation or avg
        if plotvar0 == 'avg':
            DEM0 = 'Exp' + Expnr0 + '_Pilot' + str(pilot0) + '_' + cycle0 + '_LaserScan_avg_DEM.nc'
        else:
            DEM0 = 'Exp' + Expnr0 + '_Pilot' + str(pilot0) + '_' + cycle0 + '_LaserScan_DEM.nc'        
        writeFolder = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_BaseModel' + baseModel0 + '_res' + str(gridRes) + 'mm\\'

elif method0 == 1:          # Agisoft
    # Is it a pilot?    
    if int(pilot0) == 0:
        # Is a real BaseModel used?
        if baseModel0 != False:
            directory0 = pwd + '\\Exp' + Expnr0 + '\\processed_data\\DEMs\\Agisoft\\BaseModel' + baseModel0 + '\\Res' + str(gridRes) + 'mm\\'
            DEM0 = 'Exp' + Expnr0 + '_' + cycle0 + '_' + accuracy0 + '_Agisoft_DEM.nc' 
            writeFolder = pwd + '\\Exp' + Expnr0 + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_BaseModel' + baseModel0 + '_res' + str(gridRes) + 'mm\\'
        else: # No BaseModel
            directory0 = pwd + '\\Exp' + Expnr0 + '\\processed_data\\DEMs\\Agisoft\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
            DEM0 = 'Exp' + Expnr0 + '_' + cycle0 + '_' + accuracy0 + '_Agisoft_DEM.nc' 
            writeFolder = pwd + '\\Exp' + Expnr0 + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_IndividualAlignment_res' + str(gridRes) + 'mm\\'
    else: # Pilot
        # Is a real BaseModel used?
        if baseModel0 != False:
            directory0 = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\processed_data\\DEMs\\Agisoft\\BaseModel' + baseModel0 + '\\Res' + str(gridRes) + 'mm\\'
            DEM0 = 'Exp' + Expnr0 + '_Pilot' + str(pilot0) + '_' + cycle0 + '_' + accuracy0 + '_Agisoft_DEM.nc'
            writeFolder = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_BaseModel' + baseModel0 + '_res' + str(gridRes) + 'mm\\'
        else: # No BaseModel
            directory0 = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\processed_data\\DEMs\\Agisoft\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
            DEM0 = 'Exp' + Expnr0 + '_Pilot' + str(pilot0) + '_' + cycle0 + '_' + accuracy0 + '_Agisoft_DEM.nc'
            writeFolder = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_IndividualAlignment_res' + str(gridRes) + 'mm\\'

elif method0 == 2:          # Matlab
    # Is it a pilot?    
    if int(pilot0) == 0:
        directory0 = pwd + '\\Exp' + Expnr0 + '\\processed_data\\DEMs\\Matlab\\NetCDFFiles\\Res' + str(gridRes) + 'mm\\'
        DEM0 = 'Exp' + Expnr0 + '_' + cycle0 + '_Matlab_DEM.nc' 
        writeFolder = pwd + '\\Exp' + Expnr0 + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_res' + str(gridRes) + 'mm\\'
    else: 
        directory0 = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\processed_data\\DEMs\\Matlab\\NetCDFFiles\\Res' + str(gridRes) + 'mm\\'
        DEM0 = 'Exp' + Expnr0 + '_Pilot' + str(pilot0) + '_' + cycle0 + '_Matlab_DEM.nc'
        writeFolder = pwd + '\\Exp' + Expnr0 + '\\Pilot' + str(pilot0) + '\\derived_data\\DoDs\\' + DEM0[:-3] + '_res' + str(gridRes) + 'mm\\'
else: # Method doesn't exist
    raise Exception("Method for creating DEM0 doesn't exist")
    
# Continue with DEM1
if method1 == 0:            # LaserScan
    # Is it a pilot?    
    if int(pilot1) == 0:
        directory1 = pwd + '\\Exp' + Expnr1 + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel1 + '\\Res' + str(gridRes) + 'mm\\'
        if plotvar1 == 'avg':
            DEM1 = 'Exp' + Expnr1 + '_' + cycle1 + '_LaserScan_avg_DEM.nc'
        else:
            DEM1 = 'Exp' + Expnr1 + '_' + cycle1 + '_LaserScan_DEM.nc'
    else: 
        directory1 = pwd + '\\Exp' + Expnr1 + '\\Pilot' + str(pilot1) + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel1 + '\\Res' + str(gridRes) + 'mm\\'
        if plotvar1 == 'avg':
            DEM1 = 'Exp' + Expnr1 + '_Pilot' + str(pilot1) + '_' + cycle1 + '_LaserScan_avg_DEM.nc'
        else:
            DEM1 = 'Exp' + Expnr1 + '_Pilot' + str(pilot1) + '_' + cycle1 + '_LaserScan_DEM.nc'

elif method1 == 1:          # Agisoft
    # Is it a pilot?    
    if int(pilot1) == 0:
        # Is a real BaseModel used?
        if baseModel1 != False:
            directory1 = pwd + '\\Exp' + Expnr1 + '\\processed_data\\DEMs\\Agisoft\\BaseModel' + baseModel1 + '\\Res' + str(gridRes) + 'mm\\'
        else: # No BaseModel
            directory1 = pwd + '\\Exp' + Expnr1 + '\\processed_data\\DEMs\\Agisoft\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
        DEM1 = 'Exp' + Expnr1 + '_' + cycle1 + '_' + accuracy1 + '_Agisoft_DEM.nc' 
    else: # Pilot
        # Is a real BaseModel used?
        if baseModel1 != False:
            directory1 = pwd + '\\Exp' + Expnr1 + '\\Pilot' + str(pilot1) + '\\processed_data\\DEMs\\Agisoft\\BaseModel' + baseModel1 + '\\Res' + str(gridRes) + 'mm\\'
        else: # No BaseModel
            directory1 = pwd + '\\Exp' + Expnr1 + '\\Pilot' + str(pilot1) + '\\processed_data\\DEMs\\Agisoft\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
        DEM1 = 'Exp' + Expnr1 + '_Pilot' + str(pilot1) + '_' + cycle1 + '_' + accuracy1 + '_Agisoft_DEM.nc'

elif method1 == 2:          # Matlab
    # Is it a pilot?    
    if int(pilot1) == 0:
        directory1 = pwd + '\\Exp' + Expnr1 + '\\processed_data\\DEMs\\Matlab\\NetCDFFiles\\Res' + str(gridRes) + 'mm\\'
        DEM1 = 'Exp' + Expnr1 + '_' + cycle1 + '_Matlab_DEM.nc' 
    else: 
        directory1 = pwd + '\\Exp' + Expnr1 + '\\Pilot' + str(pilot1) + '\\processed_data\\DEMs\\Matlab\\NetCDFFiles\\Res' + str(gridRes) + 'mm\\'
        DEM1 = 'Exp' + Expnr1 + '_Pilot' + str(pilot1) + '_' + cycle1 + '_Matlab_DEM.nc'
else: # Method doesn't exist
    raise Exception("Method for creating DEM1 doesn't exist")    
    
# What is de DoD name?
DoD_name = 'DoD_' + DEM1[:-3] + '-' + DEM0[:-3] + '.PNG'

#%% Load the DEM files
DEM0file = nc.Dataset(directory0 + DEM0)
DEM1file = nc.Dataset(directory1 + DEM1)

# Extract the X-Y information
xAxis0 = np.array(DEM0file .get_variables_by_attributes(name='X-axis')[0])
yAxis0 = np.array(DEM0file .get_variables_by_attributes(name='Y-axis')[0])
xAxis1 = np.array(DEM1file .get_variables_by_attributes(name='X-axis')[0])
yAxis1 = np.array(DEM1file .get_variables_by_attributes(name='Y-axis')[0])

# Verify whether the axes are equal
# We need to round numbers here to account for numerical differences between Python and matlab
if np.round(xAxis0[0],5) != np.round(xAxis1[0],5) or np.round(xAxis0[1],5) != np.round(xAxis1[1],5) or np.round(xAxis0[2],5) != np.round(xAxis1[2],5):
    raise Exception('The X-axes of both DEMs are not equal to each other')
if np.round(yAxis0[0],5) != np.round(yAxis1[0],5) or np.round(yAxis0[1],5) != np.round(yAxis1[1],5) or np.round(yAxis0[2],5) != np.round(yAxis1[2],5):
    raise Exception('The Y-axes of both DEMs are not equal to each other')
        
# Convert the X/Y-axis information to plottable values
# As both axes are equal, we can just take DEM0 as example
xValues = np.arange(xAxis0[0],xAxis0[1],xAxis0[2])
yValues = np.arange(yAxis0[0],yAxis0[1],yAxis0[2])
xx, yy = np.meshgrid(xValues,yValues)

# Now load the Z data
zValues0 = np.array(DEM0file.get_variables_by_attributes(name='Z-axis')[0])
zValues1 = np.array(DEM1file.get_variables_by_attributes(name='Z-axis')[0])

# For LaserScan-processed DEMs, elevation data is stored in percentiles (by default we use the z50)
if method0 == 0 or method0 == 2:
    # How many percentiles does the file contain?
    pcs = np.array(DEM0file.get_variables_by_attributes(name='Z percentiles')[0])
    # Find the right one
    i=0
    # Note that if plotvar is 'avg', the stored variable name is actually 'z50'. Updatr
    if plotvar0 == 'avg':
        plotvar0 = 'z50'
    # Loop trough percentiles
    for pc in pcs:
        if pc == plotvar0:
            #  Matlab data has to be imported differently
            if method0 == 0:
                zValues0 = zValues0[:,:,i].reshape((zValues0.shape[0],zValues0.shape[1]))
            else:
                zValues0 = np.transpose(zValues0)
                zValues0 = zValues0[i,:,:].reshape((zValues0.shape[1],zValues0.shape[2]))
        i += 1

if method1 == 0 or method1 == 2:
    # How many percentiles does the file contain?
    pcs = np.array(DEM1file.get_variables_by_attributes(name='Z percentiles')[0])
    # Find the right one
    i=0
    # Note that if plotvar is 'avg', the stored variable name is actually 'z50'. Updatr
    if plotvar1 == 'avg':
        plotvar1 = 'z50'
    # Loop trough percentiles
    for pc in pcs:
        if pc == plotvar1:
            #  Matlab data has to be imported differently
            if method1 == 0:
                zValues1 = zValues1[:,:,i].reshape((zValues1.shape[0],zValues1.shape[1]))
            else:
                zValues1 = np.transpose(zValues1)
                zValues1 = zValues1[i,:,:].reshape((zValues1.shape[1],zValues1.shape[2]))
        i += 1

# We can close the files again
DEM0file.close()
DEM1file.close()

# %% Calculate DoD
DoD = zValues1 - zValues0
# Rescale DoD to cm
DoD = (DoD*100)

# Create writefolder if it doesn't exist
if not os.path.exists(writeFolder):
    os.makedirs(writeFolder)

#%% Make plot
figsize=(figwidth, figwidth*(3/20)) # scale y- and x-axes
ax = plt.figure(figsize=figsize)

plt.pcolormesh(xx, yy, DoD, 
               cmap="Spectral", 
               rasterized=True,
               vmax=vmax, 
               vmin=vmin)
plt.xlim(0,20)
plt.tight_layout()
plt.tick_params(labelsize=ticksize)
plt.xlabel('x (m)', size=labelsize)
plt.ylabel('y (m)', size=labelsize)
plt.title(DoD_name[:-4], size=titlesize, pad=15)
# Add colorbar
clb = plt.colorbar(location='right', pad = 0.01)
clb.set_label('elevation difference (cm)', size=labelsize)
clb.ax.tick_params(labelsize=ticksize)

# Store image
plt.savefig(writeFolder + DoD_name, bbox_inches='tight')

# Show image
plt.show()

#%% Determine statistics of the DoD, if desired

if statistics == 1:
    # For convenience, we just skip the river input, delta and sea (as the grass shows largest deviations)
    # Let's say we exclude the first 0.50m from the river side, the last 2m from the seaside and 5cm from both edges
    yValuesReduced = yValues[(0.05 <= yValues) & (yValues <= 2.95)]
    xValuesReduced = xValues[(0.5 <= xValues) & (xValues <= 18)]
    y0 = np.where(yValuesReduced[0]==yValues)[0][0]; y1 = np.where(yValuesReduced[-1]==yValues)[0][0]+1
    x0 = np.where(xValuesReduced[0]==xValues)[0][0]; x1 = np.where(xValuesReduced[-1]==xValues)[0][0]+1
    DoDreduced = DoD[y0:y1,x0:x1]
    
    #%% Store numbers as well
    DoDreduced2 = DoDreduced

    # And determine statistical parameters from the DoD
    average = np.nanmean(DoDreduced2)
    median = np.nanmedian(DoDreduced2)
    stdev = np.nanstd(DoDreduced2)
    maxx = np.nanmax(DoDreduced2)
    minn = np.nanmin(DoDreduced2)
    perc99 = np.nanpercentile(DoDreduced2,99)
    perc01 = np.nanpercentile(DoDreduced2,1)
    perc9999 = np.nanpercentile(DoDreduced2,99.99)
    perc0001 = np.nanpercentile(DoDreduced2,0.01)
    perc999999 = np.nanpercentile(DoDreduced2,99.9999)
    perc000001 = np.nanpercentile(DoDreduced2,0.0001)
      
    # Store in CSV file
    # Continue with laserCam
    with open(writeFolder + 'DoD_' + DEM1[:-6] + 'Statistics.CSV', 'w', newline='') as statisticsFile:
        statisticsWriter = csv.writer(statisticsFile)
        statisticsWriter.writerow(['average'] + [str(average)]) 
        statisticsWriter.writerow(['median'] + [str(median)])
        statisticsWriter.writerow(['stdev'] + [str(stdev)])
        statisticsWriter.writerow(['max'] + [str(maxx)])
        statisticsWriter.writerow(['min'] + [str(minn)])
        statisticsWriter.writerow(['perc99'] + [str(perc99)])
        statisticsWriter.writerow(['perc01'] + [str(perc01)])
        statisticsWriter.writerow(['perc99.99'] + [str(perc9999)])
        statisticsWriter.writerow(['perc0.01'] + [str(perc0001)])
        statisticsWriter.writerow(['perc99.9999'] + [str(perc999999)])
        statisticsWriter.writerow(['perc0.0001'] + [str(perc000001)])
        
    #%% Determine the histogram of reduced histogram # Exclude the outer percentiles
    counts, bins = np.histogram(DoDreduced,bins=100,range=(perc01,perc99))

    output_file = writeFolder + 'DoD_' + DEM1[:-6] + 'Histogram.png'
    figsize=(figwidth/2, figwidth/4) # scale y- and x-axes
    ax = plt.figure(figsize=figsize)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.tight_layout()
    plt.tick_params(labelsize=ticksize*1.5)
    plt.xlabel('DoD value (cm)', size=labelsize*2)
    plt.ylabel('count', size=labelsize*2)
    plt.title(DoD_name[:-4] + '_Histogram', size=titlesize, pad=15)
    # Store image
    plt.savefig(output_file, bbox_inches='tight')

    # Show image
    plt.show()
    
    #%% Visualize with a cumulative distribution function
    DoDreduced2sorted = np.sort(DoDreduced2, axis=None)
    norm_cdf = np.arange(1, len(DoDreduced2sorted) + 1) / len(DoDreduced2sorted)
    xSeries = (DoDreduced2sorted-np.nanmean(DoDreduced2sorted))*10

    output_file = writeFolder + 'DoD_' + DEM1[:-6] + 'CDF.png'
    figsize=(figwidth/2, figwidth/4) # scale y- and x-axes
    ax = plt.figure(figsize=figsize)
    plt.plot(xSeries,norm_cdf)
    plt.tight_layout()
    plt.tick_params(labelsize=ticksize*1.5)
    plt.xlabel('DoD value - mean DoD value (mm)', size=labelsize*2)
    plt.xlim(np.nanpercentile(xSeries,0.1),np.nanpercentile(xSeries,99.9))
    plt.ylabel('Cumulative probability', size=labelsize*2)
    plt.title(DoD_name[:-4] +  ' Cumulatibe Distribution Function', size=titlesize, pad=15)
    # Store image
    plt.savefig(output_file, bbox_inches='tight')
    
#%% Calculate cross section if desired
if crossSection == 1:
    # First find which X-coordinate equals the desired column
    xColumn = np.where(xValues==crossSectionX)[0][0]
    DoDsection = DoD[:,xColumn]
    # What is the median of the DoDsection?
    DoDsectionMedian = np.median(DoDsection)
    perc03 = np.percentile(DoDsection,3)
    perc97 = np.percentile(DoDsection,97)
    medianPlot = np.full(len(DoDsection),DoDsectionMedian)
    
    # We would like to calibrate the kappa here as well
    medianPlotReduced = np.array(pd.DataFrame(DoDsection))/100 # make it independent from DoD DoDsection and /100 for back to m
    # Get rid of the percentile extremes for better fitting
    medianPlotReduced[medianPlotReduced<(perc03/100)] = np.nan
    medianPlotReduced[medianPlotReduced>(perc97/100)] = np.nan
    # Ignore NaNs
    idx = np.isfinite(yValues) & np.isfinite(medianPlotReduced)[:,0]
    # Find slope
    slopeSectionFunction = np.polyfit(yValues[idx],medianPlotReduced[idx],1)
    slopePlot = slopeSectionFunction[1]+slopeSectionFunction[0]*yValues
    y0Offset = slopeSectionFunction[1][0]-(DoDsectionMedian/100)
    
    # Plot
    figsize=(figwidth, 3*figwidth*(3/20)) # scale y- and x-axes
    fig, ax = plt.subplots(3, 1, figsize=figsize,sharex='col')
    fig.suptitle(DoD_name[:-4] + 'Cross section at x = ' + str(crossSectionX), size=titlesize)
    
    ax[0].plot(yValues,DoDsection,label='DoD values')
    ax[0].plot(yValues,medianPlot,'--',label='median = ' + str(DoDsectionMedian))
    #ax[0].tight_layout()
    ax[0].tick_params(labelsize=ticksize)
    #ax[0].xlabel('y (m)', size=labelsize)
    ax[0].set_ylabel('DoD value (m)', size=labelsize)
    ax[0].legend()
    
    # Repeat with tighter y-axis
    ax[1].plot(yValues,DoDsection,label='DoD values')
    ax[1].plot(yValues,medianPlot,'--',label='median = ' + str(DoDsectionMedian))
    ax[1].set_ylim(perc03,perc97)
    ax[1].tick_params(labelsize=ticksize)
    ax[1].set_xlabel('y (m)', size=labelsize)
    ax[1].set_ylabel('DoD value (cm)', size=labelsize)
    ax[1].legend()
    
    # Show additional kappa fit
    ax[2].plot(yValues,medianPlotReduced,label='DoD values')
    ax[2].plot(yValues,medianPlot/100,'--',label='median = ' + str(DoDsectionMedian))
    ax[2].plot(yValues,slopePlot,label='slope = ' + str(slopeSectionFunction[0][0]))
    ax[2].plot([yValues[0],yValues[0]],[(DoDsectionMedian/100),slopePlot[0]],label='offset at Y=0 = ' + str(y0Offset))
    ax[2].set_ylim((perc03/100),(perc97/100))
    ax[2].tick_params(labelsize=ticksize)
    ax[2].set_xlabel('y (m)', size=labelsize)
    ax[2].set_ylabel('DoD value (m)', size=labelsize)
    ax[2].legend()
    
    # Add legend
    plt.legend()

    cross_section_name = 'DoD_' + DEM1[:-6] + 'cross-section_at_X_' + str(crossSectionX) + 'm.PNG'
    # Store image
    plt.savefig(writeFolder + cross_section_name, bbox_inches='tight')

    # Show image
    plt.show()

    # Calculate the phi offset
    