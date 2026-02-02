'''
This script contains the functions for creating the water depth map movie through generating orthomosaics in Metashape

This script also contains the functions for applying Gaussion blur to the overhead orthomosaics, and for storing the resulting depthMaps as netCDF files

Python script written by Eise W. NOTA (finalized SEPTEMBER 2025)

'''

# Importing packages
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image
import io
import netCDF4 as nc
import datetime
from pandas import Timedelta
import os
import joblib

# Load internal functions
from Function_colour_correction_overhead import rgb_to_LAB, rgb_to_Luv, rgb_to_HSV, rgb_to_YCrCb

def create_metronome_movie_depthMaps(rel_orthos,nr_cycles,start_cycle,cycledurat,Expnr,pilot,figWidth,figHeight,
                                     target_image_size,writeFolder,orthoFolder,orthoRes,output_path,startXY,endXY,
                                     startXY2,endXY2,fps,maskDir,clfDir,regrDir,videoStyle,finalWeirLevel,demfloodLevel,
                                     imagery2beProcessed,fancyVideo,illOrthoFolder):
    
    #%% Raise exception if '1Hz' and 'waterDepthElevation' are both set
    if imagery2beProcessed == '1Hz' and videoStyle == 'waterDepthElevation' or imagery2beProcessed == '1Hz' and videoStyle == 'combined':
        raise Exception("Using both '1Hz' and 'waterDepthElevation' doesn't make sense")
                   
    #%% Create a writer object with the desired output path, FPS, and codec
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='mp4')
    
    #%%Load the relevant information before looping over the orthos
    # Mask
    maskNC = nc.Dataset(maskDir, 'r')
    # Extract the relevant information
    mask = np.array(maskNC.get_variables_by_attributes(name='mask')[0])
    # We can close the file again
    maskNC.close()
    # Flip vertically for ortho
    mask = np.flipud(mask)
    # Make mask suited for 3D
    orthoMask0 = np.zeros([np.size(mask,0),np.size(mask,1),3])
    orthoMask0[:,:,0] = mask; orthoMask0[:,:,1] = mask; orthoMask0[:,:,2] = mask
    # Also create a stacked mask
    maskStacked = mask.reshape(-1)
    boolMaskStacked = maskStacked.astype(bool)
    # For the waterdepthDEMs, delete columns with all zeros for mask
    if videoStyle == 'waterDepthElevation' or videoStyle == 'combined':
        mask2 = mask[:,~np.all(mask == 0, axis = 0)]
    
    # Set zeros to Nan
    nanmask = mask.astype(float)
    nanmask[nanmask==0] = np.nan
    # Delete columns with all nans
    nanmask = nanmask[:,~np.all(np.isnan(nanmask), axis=0)]
    
    # Classifier model
    dtnow = datetime.datetime.now()
    clf = joblib.load(clfDir) 
    dtnow2 = datetime.datetime.now()
    dtnow3 = round((dtnow2 - dtnow).total_seconds())
    print("Clf model loaded in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")

    # Regression model
    dtnow = datetime.datetime.now()
    regr = joblib.load(regrDir) 
    dtnow2 = datetime.datetime.now()
    dtnow3 = round((dtnow2 - dtnow).total_seconds())
    print("Regr model loaded in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
    
    #%% Iterate over the relevant orthos using the averaged_over as iteration step
    for currentOrtho in rel_orthos:
        # Indicate the current stitching sequence    
        print('Current ortho = ' + currentOrtho)
        
        # What is the current netCDF file?
        currentNetCDF = currentOrtho.replace('.JPG','.nc')
        
        # Compute if it doesn't already exist
        if not os.path.exists(writeFolder + currentNetCDF):
            # Load ortho
            ortho = cv2.imread(orthoFolder + currentOrtho)
            # Apply mask to ortho
            maskedOrtho = np.array(ortho*orthoMask0,dtype='uint8')
            # Apply the gaussian blur to maskedOrtho 
            gausOrtho = applyGaussianBlur(maskedOrtho) # Function below
            # Apply the colorspace transformations
            orthoStacked = rgb2colorspaces(gausOrtho) # Function below
            # Calculate depth map # Function below
            calculateDepthMap(orthoStacked,clf,regr,maskStacked,boolMaskStacked,np.shape(ortho),writeFolder,
                              currentNetCDF,mask,startXY,endXY,orthoRes) 
            
        #%% Load currentNetCDF
        netCDFFile = nc.Dataset(writeFolder + currentNetCDF)
        # Extract the relevant information
        xAxis = np.array(netCDFFile.get_variables_by_attributes(name='X-axis')[0])
        yAxis = np.array(netCDFFile.get_variables_by_attributes(name='Y-axis')[0])
        zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
        zData =  zData[:,:,0].reshape((zData.shape[0],zData.shape[1]))
        # We can close the file again
        netCDFFile.close()
         
        # Convert the X/Y-axis information to plottable values
        xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
        yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
        xx, yy = np.meshgrid(xValues,yValues)
        # Flip yy
        yy = np.flipud(yy)   
            
        #%%Define label for frame
        if imagery2beProcessed == 'fixedCycle':
            if pilot == 0:
                currentCycleStr = currentNetCDF[11:16]
            else:
                currentCycleStr = currentNetCDF[18:23]
            currentCycle = int(currentCycleStr)
            duration = currentCycle * cycledurat # current total seconds of experiment
        elif imagery2beProcessed == '1Hz':
            if pilot == 0:
                currentCycleStr = currentNetCDF[23:28]
                currentSecond = float(currentNetCDF[29:33].replace('_','.'))
            else:
                currentCycleStr = currentNetCDF[30:35]
                currentSecond = float(currentNetCDF[36:40].replace('_','.'))
            currentCycle = float(currentCycleStr)
            duration = currentCycle * cycledurat + currentSecond

        # This depends on the videoStyle
        if videoStyle == 'waterDepth':
            if pilot == 0:
                if imagery2beProcessed == 'fixedCycle':
                    label = "Exp" + Expnr + "     Water depth map     Cycle = " + currentCycleStr + "     (" + str(Timedelta(seconds=duration)) + ")"
                elif imagery2beProcessed == '1Hz':
                    label = "Exp" + Expnr + "     Cycle = " + currentCycleStr + "     Seconds = " + "{:04.1f}".format(currentSecond)
            else: # Pilot
                if imagery2beProcessed == 'fixedCycle':
                    label = "Exp" + Expnr + "     Pilot " + str(pilot) + "     Water depth map     Cycle = " + currentCycleStr + "     (" + str(Timedelta(seconds=duration)) + ")"    
                elif imagery2beProcessed == '1Hz':
                    label = "Exp" + Expnr + "     Pilot " + str(pilot) + "     Water depth map     Cycle = " + currentCycleStr + "     Seconds = " + "{:04.1f}".format(currentSecond)
        elif videoStyle == 'waterDepthElevation':
            if pilot == 0:
                label = "Exp" + Expnr + "     Elevation from Water depth map     Cycle = " + currentCycleStr + "     (" + str(Timedelta(seconds=duration)) + ")"
            else: # Pilot
                label = "Exp" + Expnr + "     Pilot " + str(pilot) + "     Elevation from Water depth map     Cycle = " + currentCycleStr + "     (" + str(Timedelta(seconds=duration)) + ")"    
        elif videoStyle == 'combined':
            if pilot == 0:
                label = "Exp" + Expnr + "     Water depth map & Elevation     Cycle = " + currentCycleStr + "     (" + str(Timedelta(seconds=duration)) + ")"
            else: # Pilot
                label = "Exp" + Expnr + "     Pilot " + str(pilot) + "     Water depth map & Elevation     Cycle = " + currentCycleStr + "     (" + str(Timedelta(seconds=duration)) + ")"    
            
            
        #%% To add axes to the mean image, we want to plot it in matplotlib
        # Adjust in case we want to make a zoomed-in video
        if startXY2 == startXY and endXY == endXY2:
            # Is it a fancy video?
            if fancyVideo != 1:
                # This depends on the videoStyle
                if videoStyle == 'waterDepth':
                    # We don't want to plot everything in the "Plots" section of Spyder, so turn interactive mode off
                    with plt.ioff():
                        fig, ax = plt.subplots(figsize = (figWidth,figHeight),layout='constrained') # Mannually determing larger figsize. Optionally also incorporate this as input variable
                        im = ax.pcolormesh(xx, yy, zData*100,
                                       cmap="Blues",
                                       rasterized=True,
                                       vmax=7, 
                                       vmin=0)
                        ax.set_xlim(startXY[0],endXY[0])
                        ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
                        ax.set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=15)
                        ax.set_ylim(startXY[1],endXY[1])
                        ax.set_yticks([0,1,2,3])
                        ax.set_yticklabels(['0','1','2','3 m'],fontsize=15)
                        ax.set_title(label,fontsize=25)
                        # Add colorbar
                        clb = fig.colorbar(im, ax=ax,location='right', pad = 0.005, shrink = 0.9)
                        clb.set_label('Water depth (cm)', size=20)
                        clb.ax.tick_params(labelsize=15)
                
                elif videoStyle == 'waterDepthElevation':
                    # Implement water levels from weir
                    orthoDEM = zData*-1+finalWeirLevel
                    # Set all mask values to flood level
                    orthoDEM[mask2 == 0] = demfloodLevel
                    
                    with plt.ioff():
                        fig, ax = plt.subplots(figsize = (figWidth,figHeight),layout='constrained') # Mannually determing larger figsize. Optionally also incorporate this as input variable
                        im = ax.pcolormesh(xx, yy, orthoDEM*100,
                                       cmap="terrain",
                                       rasterized=True,
                                       vmax=10, 
                                       vmin=1)
                        ax.set_xlim(startXY[0],endXY[0])
                        ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
                        ax.set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=15)
                        ax.set_ylim(startXY[1],endXY[1])
                        ax.set_yticks([0,1,2,3])
                        ax.set_yticklabels(['0','1','2','3 m'],fontsize=15)
                        ax.set_title(label,fontsize=25)
                        # Add colorbar
                        clb = fig.colorbar(im, ax=ax,location='right', pad = 0.005, shrink = 0.9)
                        clb.set_label('Elevation (cm)', size=20)
                        clb.ax.tick_params(labelsize=15)
                        
            # Is it a fancy video?
            elif fancyVideo == 1:
                
                # Load the illustrative orthomosaic
                illOrtho = cv2.imread(illOrthoFolder + currentOrtho)
                
                # Plotting statements
                h, w = illOrtho.shape[:2]
                ypixelstep = h/3 # Divide by 3 to plot each metre along y-axis
                xpixelstep = w/10 # Divide by 10 to plot each 2 metres along y-axis
                
                # Apply nanmask to depthmap to redetermine classifier map
                clfMap = nanmask*zData
                # Flip true and false
                clfMap = -1*(clfMap-1)
                # Set wet values to nan
                clfMap[clfMap<1] = np.nan
                
                # This depends on the videoStyle
                if videoStyle == 'waterDepth':
                    # We don't want to plot everything in the "Plots" section of Spyder, so turn interactive mode off
                    with plt.ioff():
                        
                        fig, ax = plt.subplots(2,1,figsize = (figWidth,figHeight),layout='constrained') # Mannually determing larger figsize. Optionally also incorporate this as input variable
                        
                        # Start with ortho
                        ax[0].imshow(illOrtho[:, :, ::-1],interpolation=None)
                        ax[0].set_xticks(np.arange(0,w+xpixelstep,xpixelstep))
                        ax[0].set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=25)
                        ax[0].set_yticks(np.arange(0,h+ypixelstep,ypixelstep))
                        ax[0].set_yticklabels(['3 m','2','1','0'],fontsize=25)
                        
                        # Then plot fancy depth map
                        im = ax[1].pcolormesh(xx, yy, zData*100,
                                              cmap="Blues",
                                              rasterized=True,
                                              vmax=7, 
                                              vmin=0)
                        # Then the dry cells
                        im2 = ax[1].pcolormesh(xx, yy, clfMap,
                                                cmap=matplotlib.colors.ListedColormap(['palegoldenrod']),
                                                rasterized=True)
                        ax[1].set_aspect('equal',anchor='NW')
                        ax[1].set_xlim(startXY[0],endXY[0])
                        ax[1].set_xticks([0,2,4,6,8,10,12,14,16,18,20])
                        ax[1].set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=25)
                        ax[1].set_ylim(startXY[1],endXY[1])
                        ax[1].set_yticks([0,1,2,3])
                        ax[1].set_yticklabels(['0','1','2','3 m'],fontsize=25)
                        
                        #ax.set_title(label,fontsize=25)
                        # Define axins for colorbars
                        axins = inset_axes(
                            ax[1],
                            width="70%",  # width: 5% of parent_bbox width
                            height="15%",  # height: 50%
                            loc="lower center",
                            bbox_to_anchor=(0.5, -0.3, 0.5, 0.5),
                            bbox_transform=ax[1].transAxes,
                            borderpad=0,
                        )
                        axins2 = inset_axes(
                            ax[1],
                            width="15%",  # width: 5% of parent_bbox width
                            height="15%",  # height: 50%
                            loc="lower center",
                            bbox_to_anchor=(0, -0.3, 0.5, 0.5),
                            bbox_transform=ax[1].transAxes,
                            borderpad=0,
                        )
                        # Add colorbar for the depth maps
                        clb = fig.colorbar(im, cax=axins,orientation="horizontal") #, pad = 0.005, shrink = 0.5, aspect = 30)
                        clb.set_label('Water depth (cm) at wet cells', size=30)
                        clb.ax.tick_params(labelsize=25)
                        # Add colorbar for the dry cells
                        clb2 = fig.colorbar(im2, cax=axins2,orientation="horizontal") #, pad = 0.005, shrink = 0.5, aspect = 30)
                        clb2.set_label('Dry cells', size=25, labelpad = 20)
                        clb2.set_ticks([])
                        # Set title
                        fig.suptitle(label,fontsize=40,y=0.97)
                
                # This depends on the videoStyle
                elif videoStyle == 'combined':
                    # Implement water levels from weir
                    orthoDEM = zData*-1+finalWeirLevel
                    # Set all mask values to flood level
                    orthoDEM[mask2 == 0] = demfloodLevel
                    
                    # We don't want to plot everything in the "Plots" section of Spyder, so turn interactive mode off
                    with plt.ioff():
                        
                        fig, ax = plt.subplots(3,1,figsize = (figWidth,figHeight),layout='constrained') # Mannually determing larger figsize. Optionally also incorporate this as input variable
                        
                        # Start with ortho
                        ax[0].imshow(illOrtho[:, :, ::-1],interpolation=None)
                        ax[0].set_xticks(np.arange(0,w+xpixelstep,xpixelstep))
                        ax[0].set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=25)
                        ax[0].set_yticks(np.arange(0,h+ypixelstep,ypixelstep))
                        ax[0].set_yticklabels(['3 m','2','1','0'],fontsize=25)
                        
                        # Then plot fancy depth map
                        im = ax[1].pcolormesh(xx, yy, zData*100,
                                              cmap="Blues",
                                              rasterized=True,
                                              vmax=6, 
                                              vmin=0)
                        # Then the dry cells
                        im2 = ax[1].pcolormesh(xx, yy, clfMap,
                                                cmap=matplotlib.colors.ListedColormap(['palegoldenrod']),
                                                rasterized=True)
                        ax[1].set_aspect('equal',anchor='NW')
                        ax[1].set_xlim(startXY[0],endXY[0])
                        ax[1].set_xticks([0,2,4,6,8,10,12,14,16,18,20])
                        ax[1].set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=25)
                        ax[1].set_ylim(startXY[1],endXY[1])
                        ax[1].set_yticks([0,1,2,3])
                        ax[1].set_yticklabels(['0','1','2','3 m'],fontsize=25)
                        
                        #ax.set_title(label,fontsize=25)
                        # Define axins for colorbars
                        axins = inset_axes(
                            ax[1],
                            width="70%",  # width: 5% of parent_bbox width
                            height="15%",  # height: 50%
                            loc="lower center",
                            bbox_to_anchor=(0.5, -0.3, 0.5, 0.5),
                            bbox_transform=ax[1].transAxes,
                            borderpad=0,
                        )
                        axins2 = inset_axes(
                            ax[1],
                            width="15%",  # width: 5% of parent_bbox width
                            height="15%",  # height: 50%
                            loc="lower center",
                            bbox_to_anchor=(0, -0.3, 0.5, 0.5),
                            bbox_transform=ax[1].transAxes,
                            borderpad=0,
                        )
                        # Add colorbar for the depth maps
                        clb = fig.colorbar(im, cax=axins,orientation="horizontal") #, pad = 0.005, shrink = 0.5, aspect = 30)
                        clb.set_label('Water depth (cm) at wet cells', size=30)
                        clb.ax.tick_params(labelsize=25)
                        # Add colorbar for the dry cells
                        clb2 = fig.colorbar(im2, cax=axins2,orientation="horizontal") #, pad = 0.005, shrink = 0.5, aspect = 30)
                        clb2.set_label('Dry cells', size=30, labelpad = 20)
                        clb2.set_ticks([])
                        
                        # Continue with elevation
                        im3 = ax[2].pcolormesh(xx, yy, orthoDEM*100,
                                        cmap="terrain",
                                        rasterized=True,
                                        vmax=9, 
                                        vmin=3)
                        ax[2].set_aspect('equal',anchor='NW')
                        ax[2].set_xlim(startXY[0],endXY[0])
                        ax[2].set_xticks([0,2,4,6,8,10,12,14,16,18,20])
                        ax[2].set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=25)
                        ax[2].set_ylim(startXY[1],endXY[1])
                        ax[2].set_yticks([0,1,2,3])
                        ax[2].set_yticklabels(['0','1','2','3 m'],fontsize=25)
                        # Define axins for colorbars
                        axins3 = inset_axes(
                            ax[2],
                            width="70%",  # width: 5% of parent_bbox width
                            height="15%",  # height: 50%
                            loc="lower center",
                            bbox_to_anchor=(0.15, -0.3, 0.7, 0.5),
                            bbox_transform=ax[2].transAxes,
                            borderpad=0,
                        )
                        # Add colorbar
                        clb3 = fig.colorbar(im3, cax=axins3, orientation="horizontal", pad = 0.005, shrink = 0.9)
                        clb3.set_label('Elevation (cm)', size=30)
                        clb3.ax.tick_params(labelsize=25)
                        
                        # Set title
                        fig.suptitle(label,fontsize=40,y=0.99)
                
        # In case there is a zoomed-in section, adjust manually to the zone of interest
        else:
            raise Exception("Zoomed section has not been implemented yet in the function file")
            
        # Store the matplotlib figure in an array
        with io.BytesIO() as memf: # Use this to only store array in RAM
            fig.savefig(memf, format='JPG',bbox_inches='tight')
            memf.seek(0)
            img = Image.open(memf)
            arr = np.asarray(img)
            #img.show()
            
        plt.close()
        del(fig,ax)
            
        #%% Add array to writer to make movie
        writer.append_data(arr)
    
    # Close the writer to finalize the movie
    writer.close()
    return print("Movie creation completed.")

#%% Other relevant functions
def applyGaussianBlur(maskedOrtho):
    # Define some defaults
    smGsettings = [5,5,0,0]
    borderType = cv2.BORDER_ISOLATED
    
    # To ignore NaNs, we have to conduct the GaussianBlur twice with the maskedOrtho 0 and a boolean map, and then divide
    maskedOrtho0 = maskedOrtho.copy()
    maskedOrtho1 = maskedOrtho.copy()
    maskedOrtho1[maskedOrtho1>0] = 1
    # Set to floats
    maskedOrtho0 = maskedOrtho0.astype(float)
    maskedOrtho1 = maskedOrtho1.astype(float)
    
    # Apply blur
    dst0 = cv2.GaussianBlur(maskedOrtho0 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderType)
    dst1 = cv2.GaussianBlur(maskedOrtho1 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderType)
    
    # Divide
    dst = dst0/dst1 
    
    # Apply mask
    dst[maskedOrtho1==0]=np.nan
    
    # Round values to nearest integer
    dst = np.rint(dst)
    
    # Set back to uint8
    gausOrtho = dst.astype(np.uint8)
    
    return gausOrtho

#%%
def rgb2colorspaces(gausOrtho):
    # Stack information
    orthoStacked = gausOrtho.reshape(-1,1,np.shape(gausOrtho)[-1])
    # Remove masked area (all zeros)
    orthoStacked = orthoStacked[orthoStacked > 0]
    orthoStacked = orthoStacked.reshape(int(len(orthoStacked)/3),1,3)
    
    # Append the bingo2 chart of colorspaces
    # LAB
    orthoStacked = np.append(orthoStacked,rgb_to_LAB(orthoStacked),axis=2)
    # Luv
    orthoStacked = np.append(orthoStacked,rgb_to_Luv(orthoStacked[:,:,:3]),axis=2)
    # YCrCb
    orthoStacked = np.append(orthoStacked,rgb_to_YCrCb(orthoStacked[:,:,:3]),axis=2)
    # HSV
    orthoStacked = np.append(orthoStacked,rgb_to_HSV(orthoStacked[:,:,:3]),axis=2)
    # Now Restructure to 2D
    orthoStacked = orthoStacked.reshape(-1,np.shape(orthoStacked)[-1])
    
    return orthoStacked

#%%
def calculateDepthMap(orthoStacked,clf,regr,maskStacked,boolMaskStacked,orthoShape,writeFolder,
                      currentNetCDF,mask,startXY,endXY,orthoRes):
    # Apply models to orthoStacked
    # Classifier
    clfResult = clf.predict(orthoStacked)    
    # Regressor
    regrResult = regr.predict(orthoStacked)
    # Combine
    combResult = clfResult*regrResult
    # Insert back into the the mask
    combOrtho = maskStacked.copy()
    combOrtho = combOrtho.astype(float)
    combOrtho[boolMaskStacked] = combResult
    # Project back to original shape
    combOrtho = combOrtho.reshape(orthoShape[:2])
    
    #% Store as netCDF # Function below
    combOrtho2NetCDF(combOrtho,writeFolder,currentNetCDF,mask,startXY,endXY,orthoRes)
    
    return print(currentNetCDF + ' is computed.')

#%%
def combOrtho2NetCDF(combOrtho,writeFolder,currentNetCDF,mask,startXY,endXY,orthoRes):
    #% Prepare orthoDepthMap for NetCDf
    # Convert grid resolution from mm to m
    gridResol = orthoRes/1000
    
    # Prepare X/Y information
    # Let the centres of each cell be in the middle
    # Original grid
    originalY = np.arange(startXY[1]+gridResol/2,endXY[1]+gridResol/2,gridResol)
    originalX = np.arange(startXY[0]+gridResol/2,endXY[0]+gridResol/2,gridResol)
    # Reduced grid for X
    reducedX = originalX[~np.all(mask == 0, axis = 0)]
    # Which information to store
    xInformation = np.array([reducedX[0],reducedX[-1]+gridResol/2,gridResol])
    yInformation = np.array([originalY[0],originalY[-1]+gridResol/2,gridResol])
    
    # Percentiles is default, it doesn't mean anything here
    percentiles = [50] 
    
    # Delete columns with all zeros in the depth map
    orthoNC = combOrtho[:,~np.all(mask == 0, axis = 0)]
        
    #% Create NetCDF
    netCDFFile = nc.Dataset(writeFolder + currentNetCDF,'w') 
    netCDFFile.title = currentNetCDF
    # Add axes attributes 
    netCDFFile.createDimension('X', len(xInformation))
    netCDFFile.createDimension('Y', len(yInformation))
    netCDFFile.createDimension('Xrange', len(reducedX))
    netCDFFile.createDimension('Yrange', len(originalY))
    netCDFFile.createDimension('PcRange', len(percentiles))
    X_var = netCDFFile.createVariable('X-axis', np.float64, ('X'))
    X_var[:] = xInformation
    Y_var = netCDFFile.createVariable('Y-axis', np.float64, ('Y'))
    Y_var[:] = yInformation
    
    # Now define over the percentiles as variables
    actualPercentiles = []
    for pc in range(len(percentiles)):
        # What is the string of the current percentile?
        actualPercentiles.append("z" + str(percentiles[pc]))
    
    # Where are we going to store the Z_grid?
    Z_grid = orthoNC[:,:,np.newaxis]
    # Create the netCDF variable of Z and the percentiles
    Z_Var = netCDFFile.createVariable('Z-axis', np.float64, ('Yrange','Xrange','PcRange'))
    pcName_Var = netCDFFile.createVariable('Z percentiles', np.str_, ('PcRange'))
    pcName_Var[:] = np.array(actualPercentiles)
    # Write the Z variables
    Z_Var[:] = Z_grid
    # Now we can close the netCDF file
    netCDFFile.close()
    