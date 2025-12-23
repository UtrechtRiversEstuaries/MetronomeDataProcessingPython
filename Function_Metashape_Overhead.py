'''
Function to create an overhead orthomosaic from the Overhead cameras of the 
Metronome in the Earth Simulation Laboratory

Python script written by Eise W. Nota (finalized JULY 2024)

Produces intrinsically and extrinsically corrected stitched images through co-alignment of a BaseModel from Overhead and DSLR imagery.

Note: BaseModel00 has a different structure than other baseModels for which this code doesn't work. Older code can be found in the Python archive.
'''

# Importing packages
import os
#import pandas as pd
import numpy as np
import cv2

# Set the agisoft_LICENSE environment variable
os.environ["..."] ='...' 
import Metashape as Metashape

# Option to check if the license was loaded correctly
Metashape.license.valid

# Set base directory of the network drive
pwd =  r"\\..."

os.chdir(pwd + r"\\\python\Overhead_scripts\Functions_Overhead") # Setting the functions directory
from Function_colour_correction_overhead import debayer_bmp

#def Orthomosaic_Overhead(input_files, writeFolder, copyFolder, image_name, brightness_factors, white_balancing_pixel, orthoRes, markerCSV, tilting_imagery=False):
def Orthomosaic_Overhead(input_files, baseModel, writeFolder, copyFolder, image_name, initialWBCSV, orthoRes, startXY, endXY, markerCSV, tilting_imagery=False):
    '''
    Parameters
    ----------
    input_files     : Source images of the seven Overhead cameras for orthomosaic
    writeFolder     : Folder to store orthomosaic
    copyFolder      : Folder to load baseModel.psx for co-alignment
    image_name      : String with output file name
    initialWBCSV    : Initial white balancing parameters for all overheads
    orthoRes        : Pixel size in mm
    startXY         : Start coordinates of X and Y
    endXY           : End coordinates of X and Y
    markerCSV       : Pixel coordinates of all markers in the overhead imagery
    tilting_imagery : Boolean, whether the Metronome is horizontal (False) or tilting (True)
    '''
    
    # Check if the copy folder exists
    if not os.path.exists(copyFolder):
        raise FileNotFoundError(f"Copy folder not found: {copyFolder}")
        
    # Check if there are 7 image files in the folder
    if np.size(input_files) != 7:
        raise Exception("The input overhead files are not 7 images")
    
    # Load the images
    raw_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in input_files]
    print("overhead images loaded")
    
    # Debayer, conduct camera-specific color correction and store the desired images
    cam = 1
    for i in range(len(raw_images)):
        writefile = copyFolder + '\\' + str(cam) + '.PNG' # filename as camnum
        deb_image = debayer_bmp(raw_images[i]) # debayer image
        # Apply initial camera-specific white-balancing
        WB_correction = np.array(initialWBCSV.iloc[i]) # Convert to array first
        WB_image = deb_image*WB_correction # Apply
        cv2.imwrite(writefile,WB_image) # store debayered and corrected image in copyFolder
        cam += 1 # update camnum
        del(deb_image) # clear ram
    del(raw_images,i,cam,writefile) # clear ram
    
    # List corrected image directories
    cor_files = [os.path.join(copyFolder, file) for file in os.listdir(copyFolder) if file.endswith('.PNG')]
    print("overhead images debayered and corrected")
    
    # Open BaseModel in Agisoft
    doc = Metashape.Document()
    doc.open(copyFolder + "\\AgisoftBaseModel" + baseModel + ".psx")
    print("base model opened")
    
    # Load the debayered images into the BaseModel
    chunk = doc.chunk
    # Add photos to the chunk
    chunk.addPhotos(cor_files)
    print("photos added")
    
    # Define the ranges of the BaseModel imagery
    baseModelOverheadCams = []
    baseModelDSLRCams = []
    baseModelLaserCams = []
    overheadCams = []
    counter = 0
    # Loop through all cameras
    for camera in chunk.cameras:
        # Test if the camera is BaseModel overhead
        if camera.label[0] == 'O':
            baseModelOverheadCams.append(counter)
        # Test if the camera is BaseModel DSLR
        elif camera.label[0] == 'D':
            baseModelDSLRCams.append(counter)
        # Test if the camera is BaseModel laser camera
        elif camera.label[0] == 'L':
            baseModelLaserCams.append(counter)
        # Others are newly added DSLR images
        else:
            overheadCams.append(counter)
        # Update counter
        counter += 1
    del(counter)
        
    # Define the ranges of the BaseModel imagery
    oldOverheadRange = range(baseModelOverheadCams[0],baseModelOverheadCams[-1]+1) # BaseModel overhead photos
    DSLRRange = range(baseModelDSLRCams[0],baseModelDSLRCams[-1]+1) # BaseModel DSLR photos
    LaserRange = range(baseModelLaserCams[0],baseModelLaserCams[-1]+1) # BaseModel DSLR photos
    
    # To be sure of consistency, we'll digitize markers but it should not be necessary for most overhead images
    # Digitizing markers based on pixel coordinates is not going to work if we have tilting imagery (else you get accordeon videos)
    if tilting_imagery == False:
        # Disable DSLR cameras
        for cam in DSLRRange:
            chunk.cameras[cam].enabled=False
        del(cam)
        
        # Disable Laser cameras
        for cam in LaserRange:
            chunk.cameras[cam].enabled=False
        del(cam)
        
        # Enable BaseModel overhead imagery to optimize alignment
        for cam in oldOverheadRange:
            chunk.cameras[cam].enabled=True
        del(cam)
        
        # Loop through the CSV file to digitize the GCP coordinates
        for i in range(np.size(markerCSV,0)):
            # Which marker do we want to digitize?
            marker = chunk.markers[markerCSV.at[i,"markerindex"]]
            # Setting the projection accuracy is not necessary. However if we want this in the future, this is the code.
            #marker.reference.accuracy = Metashape.Vector((markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"]))
            # For which camera?
            camera = chunk.cameras[markerCSV.at[i,"camindex"]+overheadCams[0]] # Add overheadCams[0] to get the actual index
            # What is the X pixel value?
            Xpixel = markerCSV.at[i,"Xpixel"]
            # What is the Y pixel value?
            Ypixel = markerCSV.at[i,"Ypixel"]
            # Now digitize
            marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([Xpixel,Ypixel]), True)
            del(marker,camera,Xpixel,Ypixel)   
        
        print("GCP's digitized")
        
    # If there is tilting imagery, we have to fool agisoft to let it believe that the Metronome is static, while the cameras are tilting, so without GCPs
    else: # tilting_imagery == True
        # Enable DSLR cameras in this case
        for cam in DSLRRange:
            chunk.cameras[cam].enabled=True
        del(cam)
        
        # Disable overhead imagery to prevent agisoft to make tie points with the non-moving fence that have the exact same pixel coordinates for the overhead images
        for cam in oldOverheadRange:
            chunk.cameras[cam].enabled=False
        del(cam)
        
        print("GCP's not digitized")
     
    # Matching photos and optimizing the cameras by aligning it with the reference enabled imagery
    chunk.matchPhotos(downscale=1,generic_preselection=True,reference_preselection=True,reference_preselection_mode=Metashape.ReferencePreselectionEstimated,keep_keypoints=True,reset_matches=False)
    chunk.alignCameras(adaptive_fitting = True)
    print("cameras aligned and optimized")
    
    # Enable DSLR and disable Overhead cameras for Agisoft color correction
    for cam in DSLRRange:
        chunk.cameras[cam].enabled=True
    for cam in oldOverheadRange:
        chunk.cameras[cam].enabled=False
    del(cam)
    
    # Calibrate Colors with tie points and white balancing
    chunk.calibrateColors(source_data=Metashape.TiePointsData,white_balance=True)
    print("color calibrated")

    # Disable DSLR cameras
    for cam in DSLRRange:
        chunk.cameras[cam].enabled=False
    del(cam)
    
    # Build the Orthomosaic
    resM = orthoRes/1000 # Divide by 1000 to go from mm to m
    chunk.buildOrthomosaic(fill_holes=True,resolution=resM) 
    print("orthomosaic built")

    # Export the orthomosaic to a file with compression settings
    c = Metashape.ImageCompression()
    c.TiffCompressionLZW
    c.jpeg_quality = 99
    
    # Defining which part of the orthomosaic will be saved. We're only interested in the cartesion coordinates of x: 0-20 and y: 0-3 m
    b = Metashape.BBox()
    b.min = Metashape.Vector(startXY) # min X and min Y
    b.max = Metashape.Vector(endXY) # max X and max Y
    b.size = 2
    
    # For some unexplainable reason, the amount of pixels is not always constant depending on resolution,
    # So we need to specify the amount of pixels when exporting the raster
    totalWidth = endXY[0]-startXY[0]
    totalHeight = endXY[1]-startXY[1]
    pixelWidth = int(totalWidth/resM) # Round to integers
    pixelHeigth = int(totalHeight/resM) # Round to integers
    
    # Exporting the orthomosaic with predefined settings
    # Some alledgly redundant settings are added to assure a consistent pixel size
    chunk.exportRaster(path=(writeFolder + image_name),
                            image_format = Metashape.ImageFormatJPEG, 
                            raster_transform = Metashape.RasterTransformNone,
                            image_compression = Metashape.ImageCompression(),
                            region = b,
                            resolution = resM,
                            block_width = pixelWidth,
                            block_height = pixelHeigth,
                            width = pixelWidth,
                            height = pixelHeigth,
                            white_background=True,
                            north_up = False)
    
    # Close doc without saving
    del(doc)
    
    return print("Orthomosaic is saved in " + writeFolder)
