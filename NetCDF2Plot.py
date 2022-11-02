# -*- coding: utf-8 -*-
"""
This script creates a basic plot from processed Metronome DEM data 
in a netCDF file
Created on Fri Sep 16 15:36:37 2022

@author: Jan-Eike Rossius
"""

# %% importing libraries
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython import get_ipython; 
import math  
import ctypes
import statistics

# %% clearing workspace
get_ipython().magic('reset -sf') #clear workspace
plt.close('all') #close all figures/windows

# %% loading data and defining variable to plot
#filepath = r'C:\Users\7062052\OneDrive - Universiteit Utrecht\Open Science Metronome\Code\Exp049DEMs.nc' #setting the path to the file
#parent = nc.Dataset(filepath) #opening the file
#plotvar = 'z50' #define the name of the variable from the file that should be plotted

# %% Input - to be filled in individually
# your personal working directory in which you use the standard folder structure:
pwd = r'C:\Users\7062052\OneDrive - Universiteit Utrecht\Open Science Metronome\MetronomeWorkingDirectory'
# the current experiment (three digit number, in case of pilots use e.g. '052\Pilot1' as this string is used for directory paths)
exp = '052\Pilot2'
# define the name of the variable from the file that should be plotted
plotvar='z50'

# %% Initialisation
# %% Genaral

expalt=exp.replace("\\", "-") # replacing \ in exp string to have a string that is nicer for naming files etc.

# open the file
parent = nc.Dataset(pwd + r'\01metronome_experiments\Exp' + exp + r'\processed_data\DEMs\laser_scanner\Exp' + expalt + r'DEMs.nc')

# Get information about the contents of the file.
dimsparent = parent.dimensions #extract dimensions from parten directory of file
groups = parent.groups #extract groups from parten directory of file
if len(dimsparent) == 0: #if no dimensions exist in parent directory
    DEMgroup = parent.groups['DEMs'] #search for group named DEMs
else:
    DEMgroup = parent #if there are dimensions in teh parent directory, the DEM data is assumed to be stored in the parent directory
  
# %% Get dimension data
xdim = DEMgroup.get_variables_by_attributes(axis='X')[0] #extract x-dimension (along flume)
ydim = DEMgroup.get_variables_by_attributes(axis='Y')[0] #extract y-dimension (across flume)
tdim = DEMgroup.get_variables_by_attributes(axis='T')[0] #extract time-dimension

#check whether all necessary dimensions exist
existtest = int('xdim' in locals())
existtest = existtest + int('ydim' in locals())
existtest = existtest + int('tdim' in locals())

#if not, throw errors       
if existtest == 0:
    raise ValueError('All dimesions do not contain axis attribute')
elif existtest<3 & existtest>0:
    raise ValueError('Only one or two dimensions contain axis attributes. It should be three')

# %% Get variable data
#extract the variable that should be plotted based on the name defined at the beginning of the script
plotvar = DEMgroup.get_variables_by_attributes(name=plotvar)[0]

# %% Determining figure and plot size
# Getting the DPI of the screen
LOGPIXELSX = 88
LOGPIXELSY = 90
dc = ctypes.windll.user32.GetDC(0)
horzdpi = ctypes.windll.gdi32.GetDeviceCaps(dc, LOGPIXELSX)
vertdpi = ctypes.windll.gdi32.GetDeviceCaps(dc, LOGPIXELSY)
ctypes.windll.user32.ReleaseDC(0, dc)
screenDPI = statistics.mean([horzdpi, vertdpi])

# Getting the screen size in pixels
user32 = ctypes.windll.user32
screenheight = user32.GetSystemMetrics(1)
screenwidth = user32.GetSystemMetrics(0)

figheight=screenheight-150 # set height of the figure to 150 pixels less than the height of the screen
numplotsvert=math.ceil(len(tdim)/2) # number of rows of plots
# check whether there is an even or odd number of plots necessary
if numplotsvert==len(tdim)/2:
    even=True
else:
    even=False

padding=75 # space at the sides of the figure used for axis labels etc.
vertpixplot=(figheight-padding)/(numplotsvert+1*int(even)) # determine number of vertical pixels per subplot. padding is left for axis description. Space of one plot row is left for colourbar if there's an even number of plots necessary.

if getattr(xdim,'units')!=getattr(ydim,'units'): # check if units of both spatial axes are equal
    raise ValueError('x- and y-dimensions do not have the same units. Adjust code manually!') # if not code needs to be adjusted as this is ssumed in the fllowing section
else:
    pixperunit=vertpixplot/(max(ydim)-min(ydim)) # caculate number of pixels available per unit length to determine width of plots and figure

horzpixplot=(max(xdim)-min(xdim))*pixperunit # calculate number of horizontal pixels per plot
figwidth=padding+2*horzpixplot # set figure width to twice the plot width plus padding for axis labels

if figwidth > screenwidth: # in case the figure gets too wide, base the figure size on the screen width instead
    figwidth = screenwidth-50
    horzpixplot = (figwidth-padding)/2
    pixperunit = horzpixplot/(max(xdim)-min(xdim))
    vertpixplot = pixperunit*(max(ydim)-min(ydim))
    figheight = vertpixplot*(numplotsvert+1*int(even))+padding

# Creating figure
fig = plt.figure(figsize=(figwidth/screenDPI, figheight/screenDPI), dpi=screenDPI)

# Plot size and padding relative to figure size
vertrelplot=vertpixplot/figheight
horzrelplot=horzpixplot/figwidth
vertrelpad=padding/figheight
horzrelpad=padding/figwidth

# %% Actual plotting
# Defining the colormap
valrange = np.percentile(plotvar,[2, 12, 17, 20, 25, 35, 40, 60, 75, 85, 98]) # percentile values for an adjusted colormap
demcolorpos = valrange-valrange[0] # setting lowest value to zero and adjust the others
demcolorpos = demcolorpos/demcolorpos[-1] # norming all values on the range 0-1
# color dictionary for the colormap, ranging from blue via white, green and yellow to brown
cdict = {'red':    [[demcolorpos[0],  0.0, 0.0],
                   [demcolorpos[1],  0.121568627, 0.121568627],
                   [demcolorpos[2],  0.0, 0.0],
                   [demcolorpos[3],  0.596078431, 0.596078431],
                   [demcolorpos[4],  0.866666667, 0.866666667],
                   [demcolorpos[5],  0.215686275, 0.215686275],
                   [demcolorpos[6],  0.0, 0.0],
                   [demcolorpos[7],  0.57254902, 0.57254902],
                   [demcolorpos[8],  0.894117647, 0.894117647],
                   [demcolorpos[9], 0.749019608, 0.749019608],
                   [demcolorpos[10], 0.51372549, 0.51372549]],
         'green':  [[demcolorpos[0],  0.125490196, 0.125490196],
                    [demcolorpos[1],  0.305882353, 0.305882353],
                    [demcolorpos[2],  0.690196078, 0.690196078],
                    [demcolorpos[3],  0.752941176, 0.752941176],
                    [demcolorpos[4],  0.921568627, 0.921568627],
                    [demcolorpos[5],  0.337254902, 0.337254902],
                    [demcolorpos[6],  0.690196078, 0.690196078],
                    [demcolorpos[7],  0.815686275, 0.815686275],
                    [demcolorpos[8],  0.764705882, 0.764705882],
                    [demcolorpos[9],  0.560784314, 0.560784314],
                    [demcolorpos[10],  0.235294118, 0.235294118]],
         'blue':   [[demcolorpos[0],  0.376470588, 0.376470588],
                    [demcolorpos[1],  0.470588235, 0.470588235],
                    [demcolorpos[2],  0.941176471, 0.941176471],
                    [demcolorpos[3],  0.894117647, 0.894117647],
                    [demcolorpos[4],  0.968627451, 0.968627451],
                    [demcolorpos[5],  0.137254902, 0.137254902],
                    [demcolorpos[6],  0.31372549, 0.31372549],
                    [demcolorpos[7],  0.31372549, 0.31372549],
                    [demcolorpos[8],  0.290196078, 0.290196078],
                    [demcolorpos[9],  0.0, 0.0],
                    [demcolorpos[10],  0.047058824, 0.047058824]]}
DEMcmap = LinearSegmentedColormap('DEMcmap', segmentdata=cdict, N=256) # creating the colormap
font = {'size': 10} # font characteristics
fontheight = 15 # Number of pixels for one line of text, used in time step labelling
ax = [None] * len(tdim) # pre-allocating array for axes objects

for i in range(len(tdim)): # looping through the time-dimension
    if i+1<=numplotsvert: # define positions of plots in figure for the left column
        posx=horzrelpad*0.75
        posy=1-vertrelpad*0.25-((i+1+int(even))*vertrelplot)
    else: # define positions of plots in figure for the right column
        posx=horzrelpad*0.75+horzrelplot
        posy=1-vertrelpad*0.25-((i-numplotsvert+2)*vertrelplot)

    ax[i]=fig.add_axes([posx, posy, horzrelplot, vertrelplot]) # creating subplot
    img = ax[i].pcolorfast(xdim,ydim,plotvar[i,:,:],cmap=DEMcmap,vmin=valrange[0], vmax=valrange[-1]) # plotting the data with DEMcolormap and normalize colors to 2nd and 98th percentile
    
    ax[i].text(min(xdim)+((max(xdim)-min(xdim))/100),max(ydim)-((fontheight/vertpixplot)*(max(ydim)-min(ydim))),"{:04g}".format(tdim[i]),fontdict=font) # label the value of the time step
    ax[i].text(min(xdim)+((max(xdim)-min(xdim))/100),max(ydim)-2*((fontheight/vertpixplot)*(max(ydim)-min(ydim))),getattr(tdim,'units'),fontdict=font) # label the unit of the time step
    
    if i>=numplotsvert:
        plt.yticks(visible = False) # no y-axis label for right column of plots
    else:
        plt.ylabel('y [' + getattr(ydim,'units') + ']', fontdict=font) # label the y-axis with corresponding unit
    
    if i==numplotsvert-1 or i==len(tdim)-1: 
        plt.xlabel('x [' + getattr(xdim,'units') + ']', fontdict=font) # label the x-axis with corresponding unit for bottom plots
    else:
        plt.xticks(visible = False) # no x-axis label for plots not at the bottom

# Create axes for colorbar placement depending on even or odd number of plots    
if even:
    cax = fig.add_axes([horzrelpad*0.75, 1-vertrelpad*0.25-vertrelplot*0.3, 2*horzrelplot, vertrelplot*0.3])
else:
    cax = fig.add_axes([horzrelpad*0.75+1.1*horzrelplot, 1-vertrelpad*0.25-vertrelplot*0.3, 0.9*horzrelplot, vertrelplot*0.3])
# Adding the colorbar
fig.colorbar(img,cax=cax,label=getattr(plotvar,'long_name') + ' [' + getattr(plotvar,'units') + ']', orientation="horizontal")
   
# %% Closing file
parent.close()

