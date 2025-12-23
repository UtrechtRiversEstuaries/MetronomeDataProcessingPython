'''
This script contains the functions for creating the movie through generating orthomosaics in Metashape

For now, the 1Hz function is also here, but we'll make a different function for it.

Python script written by Eise W. NOTA (finalized JUNE 2024)

We need to keep a low RAM, so creating the video should be as follows:
- Start with writing the video file
- Start a loop over all the relevant files
- Stitch the first averaged_over amount of overhead imagery
- Average these stitched images
- Delete the single-cycle stitched images
- Use the average image to write into the video file
- Delete the averaged image
- New iteration of loop
- Close video file
'''

# Importing packages
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
#from datetime import time
from datetime import timedelta
import os

# Set base directory of the network drive
pwd =  r"\\..."

# Internal functions can be found in R:\Metronome\python\Functions_Overhead
os.chdir(pwd + r"\\\python\Overhead_scripts\Functions_Overhead") # Setting the functions directory
from Function_Metashape_Overhead import Orthomosaic_Overhead

def create_metronome_movie_Metashape(rel_files,output_path,fps,averaged_over,nr_cycles,start_cycle,jump,cycledurat,form,
                                     Expnr,pilot,figWidth,figHeight,target_image_size,initialWBCSV,skipFolders,foldersToSkip,baseModel,
                                     orthoFolder,copyFolder,calculateMean,calculateMedian,orthoRes,startXY,endXY,markerCSV,tilting_imagery=False):
                       
    #%% Create a writer object with the desired output path, FPS, and codec
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='mp4')
    
    # What is the length of the relevant files (equal for each camera)
    rel_file_length = len(rel_files.setdefault('\cam_1'))
    
    # Indicate for how many sequences you think this is going to take
    expectedCounter = round(rel_file_length/averaged_over)
    print('Expected final sequence counter = ' + str(expectedCounter))
    
    # Set initial values
    sequence = 1 # stitching sequence
    cycle = int(start_cycle) # actual cycle
    cyclestring = f"{cycle:05}"
    
    #%% Iterate over the relevant images using the averaged_over as iteration step
    for rel_sequence in range(0,rel_file_length,averaged_over):
        # Indicate the current stitching sequence    
        print('Current stitching sequence = ' + str(sequence) + ' (' + str(round(sequence/expectedCounter*100,1)) + ' %)')
        
        # First iterate through the files to the orthomosaics in the averaged_over amount of files 
        counter = 0 
        # List of ortho names
        orthoNames = []
        # Create the orthomosaics
        for l in range(averaged_over): # Now consider the averaging of subsequent cycles
            print(counter)
            m = rel_sequence + l # Add both indexes with each other
            # Agisoft needs an image name to export the raster
            if pilot == 0:
                image_name = 'Exp' + Expnr + '_RGB_' + cyclestring +'.JPG'
            else:
                image_name = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_RGB_' + cyclestring +'.JPG'
            print(image_name)
            orthoNames.append(image_name)
            
            # Create the Orthomosaic if it doesn't already exist in the orthoFolder
            if not os.path.exists(orthoFolder + image_name):
                # Create a list of the image names for using the orthomosaic function
                input_files = [rel_files['\cam_1'][m],rel_files['\cam_2'][m],rel_files['\cam_3'][m],rel_files['\cam_4'][m],
                               rel_files['\cam_5'][m],rel_files['\cam_6'][m],rel_files['\cam_7'][m]]
                # Create orthomosaic through Metashape
                Orthomosaic_Overhead(input_files, baseModel, orthoFolder, copyFolder, image_name, initialWBCSV, orthoRes, startXY, endXY, markerCSV, tilting_imagery)
                
            # Update cycle and counter if averaged_over is larger than 1
            cycle += 1
            counter += 1
            cyclestring = f"{cycle:05}"
        
        # Update cycle and cyclestring for the next iteration
        cycle = cycle-averaged_over # Take cycle back to the first cycle of this mean_ortho
        cyclestring = f"{cycle:05}"
        
        # Load the orthomosaics into Python
        orthomosaics = [] # Save the orthomosaics in this list
        for o in range(len(orthoNames)):
            orthomosaics.append(cv2.imread(orthoFolder +  orthoNames[o]))
            
        #%% Now use the stitched images to calculate the mean or median
        if calculateMean == 1:
            h, w = orthomosaics[0].shape[:2]
            mean_ortho = np.zeros((h, w, 3), float)
            
            for im in orthomosaics:
                imarr = np.array(im, dtype=float)
                mean_ortho = mean_ortho + imarr
            mean_ortho = mean_ortho / averaged_over
            
            del(orthomosaics,im,imarr,h,w)
        elif calculateMedian == 1:
            mean_ortho = np.stack(orthomosaics)
            mean_ortho = np.median(mean_ortho, axis = 0)
            
        mean_ortho = mean_ortho.astype(np.uint8)
            
        #%%Define label for frame
        duration = cycle * cycledurat # current total seconds of experiment
        if pilot == 0:
            label = "Exp" + Expnr + "     Cycle = " + cyclestring + "     (" + str(timedelta(seconds=duration)) + ")"
        else: # Pilot
            label = "Exp" + Expnr + "     Pilot " + str(pilot) + "     Cycle = " + cyclestring  + "     (" + str(timedelta(seconds=duration)) + ")"    

        #%% To add axes to the mean image, we want to plot it in matplotlib
        # Plotting statements
        h, w = mean_ortho.shape[:2]
        ypixelstep = h/3 # Divide by 3 to plot each metre along y-axis
        xpixelstep = w/10 # Divide by 10 to plot each 2 metres along y-axis
        
        # We don't want to plot everything in the "Plots" section of Spyder, so turn interactive mode off
        with plt.ioff():
            fig, ax = plt.subplots(figsize = (figWidth,figHeight)) # Mannually determing larger figsize. Optionally also incorporate this as input variable
            ax.imshow(mean_ortho[:, :, ::-1],interpolation=None)
            ax.set_xticks(np.arange(0,w+xpixelstep,xpixelstep))
            ax.set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=15)
            ax.set_yticks(np.arange(0,h+ypixelstep,ypixelstep))
            ax.set_yticklabels(['3 m','2','1','0'],fontsize=15)
            ax.set_title(label,fontsize=25)
           
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
        
        del(mean_ortho,h,w,memf,img,arr)
        
        # Update the actual cycle. This depends on whether folders are skipped
        if skipFolders == 1:
            # First do normal updating of the cycles with jump
            cycle += jump
            # Test if the new cycle occurs in any of the deleted folders
            for folder in foldersToSkip:
                if int(folder[:5]) <= cycle <= int(folder[6:]):
                    # Update cycle according to the next relevant file number
                    # Only if there is a next iteration
                    if m < rel_file_length-averaged_over:
                        # What is the directory of next image used?
                        next_image_dir = rel_files['\cam_1'][m+1]
                        # Split up the directory to extract the folder and filename
                        path = os.path.normpath(next_image_dir)
                        path = path.split(os.sep)
                        # We also need the cycle of the first file of the path
                        folder_dir = next_image_dir.replace(path[-1],'')
                        firstFile_folder_dir = os.listdir(folder_dir)
                        firstFile_folder_dir.sort()
                        # We only need the first file
                        firstFile_folder_dir = firstFile_folder_dir[0]
                        # Look at the cycle
                        firstFileCycle = firstFile_folder_dir[2:6]
                        # Directories and filenames can change, but we can be certain that:
                        # (1) The penultimate element in the list is the next cycle folder; first 5 digits are next cycle
                        cycleFolder = int(path[-2][:5])
                        # (2) The final element in the list is the filename; 3-6 digits indicate the cycle number 
                        # (3) The first file of the folder may not start with index 0001, so we have to subtract this one
                        cycleNumber = int(path[-1][2:6])-int(firstFileCycle)
                        # Update to the actual new cycle
                        cycle = cycleFolder + cycleNumber
                        
                        # Delete the unneccessary stuff
                        del(next_image_dir,path,cycleFolder,cycleNumber)
            del(folder)
        else: # just use the jump
            cycle += jump # Update the actual cycle
        cyclestring = f"{cycle:05}"
        
        # Update sequence with jump
        sequence += 1
    
    # Close the writer to finalize the movie
    writer.close()

    return "Movie creation completed."