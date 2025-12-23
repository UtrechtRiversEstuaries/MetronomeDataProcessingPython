'''
Function to create a DSLR orthomosaic from the Overhead cameras of the 
Metronome in the Earth Simulation Laboratory

Python script written by Brechtje A. Van Amstel and Eise W. Nota (finalized JUNE 2024)

'''

# Importing packages
import os
import shutil
import numpy as np

# Set the agisoft_LICENSE environment variable
os.environ["..."] ='...' 
import Metashape as Metashape

# Option to check if the license was loaded correctly
Metashape.license.valid

#%% Orthomosaics only
def Orthomosaic_DSLR(input_files, baseModel, writeFolder, copyFolder, image_name, orthoRes, startXY, endXY, markerCSV):
    '''
    Parameters
    ----------
    input_files : Source images for orthomosaic
    baseModel   : BaseModel used for co-alignment
    writeFolder : Folder to save the orthomosaics
    copyFolder  : Folder to load baseModel.psx for co-alignment
    image_name  : String with output file name
    orthoRes    : Pixel size in mm
    startXY     : Start coordinates of X and Y
    endXY       : End coordinates of X and Y
    markerCSV   : Pixel coordinates of all markers in the DSLR imagery
    '''
    
    # Check if the copy folder exists
    if not os.path.exists(copyFolder):
        raise FileNotFoundError(f"Copy folder not found: {copyFolder}")
    
    # Open BaseModel in Agisoft
    doc = Metashape.Document()
    doc.open(copyFolder + "\\AgisoftBaseModel" + baseModel + ".psx")
    print("base model opened")
    
    # Load the DSLR images into the BaseModel
    chunk = doc.chunk
    # Add photos to the chunk
    chunk.addPhotos(input_files)
    print("photos added")
    
    # Define the ranges of the BaseModel imagery
    baseModelOverheadCams = []
    baseModelDSLRCams = []
    baseModelLaserCams = []
    DSLRCams = []
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
            DSLRCams.append(counter)
        # Update counter
        counter += 1
    del(counter)
        
    # Define the ranges of the BaseModel imagery
    OverheadRange = range(baseModelOverheadCams[0],baseModelOverheadCams[-1]+1) # BaseModel overhead photos
    oldDSLRRange = range(baseModelDSLRCams[0],baseModelDSLRCams[-1]+1) # BaseModel DSLR photos
    LaserRange = range(baseModelLaserCams[0],baseModelLaserCams[-1]+1) # BaseModel DSLR photos
    #inputRange = range(DSLRCams[0],DSLRCams[-1]+1) # Input DSLR photos
    
    # Loop through the CSV file to digitize the GCP coordinates
    for i in range(np.size(markerCSV,0)):
        # Which marker do we want to digitize?
        marker = chunk.markers[markerCSV.at[i,"markerindex"]]
        # Setting the projection accuracy is not necessary. However if we want this in the future, this is the code.
        #marker.reference.accuracy = Metashape.Vector((markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"]))
        # For which camera?
        camera = chunk.cameras[markerCSV.at[i,"camindex"]+DSLRCams[0]] # Add DSLRCams[0] to get the actual index
        # What is the X pixel value?
        Xpixel = markerCSV.at[i,"Xpixel"]
        # What is the Y pixel value?
        Ypixel = markerCSV.at[i,"Ypixel"]
        # Now digitize
        marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([Xpixel,Ypixel]), True)
        del(marker,camera,Xpixel,Ypixel)   
    
    print("GCP's digitized")
    
    # Disable Laser cameras
    for cam in LaserRange:
        chunk.cameras[cam].enabled=False
    del(cam)
    
    # Matching photos and optimizing the cameras by aligning it with the reference baseModel imagery enabled
    # Enable DSLR cameras in this case
    for cam in oldDSLRRange:
        chunk.cameras[cam].enabled=True
    del(cam)
    for cam in OverheadRange:
        chunk.cameras[cam].enabled=True
    del(cam)
    
    # Matching photos and optimizing the cameras by aligning it with the reference enabled imagery
    chunk.matchPhotos(downscale=1,generic_preselection=True,reference_preselection=True,reference_preselection_mode=Metashape.ReferencePreselectionSource,keep_keypoints=True,reset_matches=False)
    chunk.alignCameras(adaptive_fitting = True)
    print("cameras aligned and optimized")
    
    # Disable the cameras again
    for cam in oldDSLRRange:
        chunk.cameras[cam].enabled=False
    del(cam)
    for cam in OverheadRange:
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
    
    return "Orthomosaic is saved in" + writeFolder

#%% Orthomosaics + DEMs using BaseModel alignment
def Orthomosaic_DEM_DSLR(input_files, baseModel, writeOrthoFolder, writeDEMFolder, copyFolder, ortho_name, demTif_name, demOrtho_name, gridRes, startXY, endXY, markerCSV, depthMapSetting):
    '''
    Parameters
    ----------
    input_files         : Source images for orthomosaic
    baseModel           : BaseModel used for co-alignment
    writeOrthoFolder    : Folder to save the orthomosaics
    writeDEMFolder      : Folder to save the DEMs
    copyFolder          : Folder to load baseModel.psx for co-alignment
    ortho_name          : String with orthomosaic output file name
    demTif_name         : String with GeoTIFF DEM output file name
    demOrtho_name       : String with ortho DEM output file name
    gridRes             : Pixel/grid size in mm
    startXY             : Start coordinates of X and Y
    endXY               : End coordinates of X and Y
    markerCSV           : Pixel coordinates of all markers in the DSLR imagery
    depthMapSetting     : Setting for generating new depth maps
    '''
    
    # Check if the copy folder exists
    if not os.path.exists(copyFolder):
        raise FileNotFoundError(f"Copy folder not found: {copyFolder}")
    
    # Provide the integer for the downscaleFactor based on the depthMapSetting
    if depthMapSetting == 'ultra' or depthMapSetting == 'Ultra':
        downscaleFactor = 1
    elif depthMapSetting == 'high' or depthMapSetting == 'High':
        downscaleFactor = 2
    elif depthMapSetting == 'medium' or depthMapSetting == 'Medium':
        downscaleFactor = 4
    elif depthMapSetting == 'low' or depthMapSetting == 'Low':
        downscaleFactor = 8
    elif depthMapSetting == 'lowest' or depthMapSetting == 'Lowest':
        downscaleFactor = 16
    else:
        raise Exception("The defined depthMapSetting does not exist")
        
    # Open BaseModel in Agisoft
    doc = Metashape.Document()
    doc.open(copyFolder + "\\AgisoftBaseModel" + baseModel + ".psx")
    print("base model opened")
    
    # Load the DSLR images into the BaseModel
    chunk = doc.chunk
    # Add photos to the chunk
    chunk.addPhotos(input_files)
    print("photos added")
    
    # Define the ranges of the BaseModel imagery
    baseModelOverheadCams = []
    baseModelDSLRCams = []
    baseModelLaserCams = []
    DSLRCams = []
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
            DSLRCams.append(counter)
        # Update counter
        counter += 1
    del(counter)
        
    # Define the ranges of the BaseModel imagery
    OverheadRange = range(baseModelOverheadCams[0],baseModelOverheadCams[-1]+1) # BaseModel overhead photos
    oldDSLRRange = range(baseModelDSLRCams[0],baseModelDSLRCams[-1]+1) # BaseModel DSLR photos
    LaserRange = range(baseModelLaserCams[0],baseModelLaserCams[-1]+1) # BaseModel DSLR photos
    
    # Loop through the CSV file to digitize the GCP coordinates
    for i in range(np.size(markerCSV,0)):
        # Which marker do we want to digitize?
        marker = chunk.markers[markerCSV.at[i,"markerindex"]]
        # Setting the projection accuracy is not necessary. However if we want this in the future, this is the code.
        #marker.reference.accuracy = Metashape.Vector((markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"]))
        # For which camera?
        camera = chunk.cameras[markerCSV.at[i,"camindex"]+DSLRCams[0]] # Add DSLRCams[0] to get the actual index
        # What is the X pixel value?
        Xpixel = markerCSV.at[i,"Xpixel"]
        # What is the Y pixel value?
        Ypixel = markerCSV.at[i,"Ypixel"]
        # Now digitize
        marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([Xpixel,Ypixel]), True)
        del(marker,camera,Xpixel,Ypixel)   
    print("GCP's digitized")
    
    # Disable Laser cameras
    for cam in LaserRange:
        chunk.cameras[cam].enabled=False
    del(cam)
    
    # Matching photos and optimizing the cameras by aligning it with the reference baseModel imagery enabled
    # Enable DSLR cameras in this case
    for cam in oldDSLRRange:
        chunk.cameras[cam].enabled=True
    del(cam)
    for cam in OverheadRange:
        chunk.cameras[cam].enabled=True
    del(cam)
    
    # Matching photos and optimizing the cameras by aligning it with the reference enabled imagery
    chunk.matchPhotos(downscale=1,generic_preselection=True,reference_preselection=True,reference_preselection_mode=Metashape.ReferencePreselectionSource,keep_keypoints=True,reset_matches=False)
    chunk.alignCameras(adaptive_fitting = True)
    print("cameras aligned and optimized")
    
    # Disable the cameras again
    for cam in oldDSLRRange:
        chunk.cameras[cam].enabled=False
    del(cam)
    for cam in OverheadRange:
        chunk.cameras[cam].enabled=False
    del(cam)
    
    # Build the Orthomosaic
    resM = gridRes/1000 # Divide by 1000 to go from mm to m
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
    chunk.exportRaster(path=(writeOrthoFolder + ortho_name),
                            source_data =  Metashape.OrthomosaicData,
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
    print("orthomosaic is saved in" + writeOrthoFolder)
    
    # With the input DSLR cameras aligned, we can replace the depth maps with only the input cameras
    chunk.buildDepthMaps(downscale=downscaleFactor, filter_mode=Metashape.MildFiltering)
    print("new depth maps generated")
    
    # Construct new point cloud
    chunk.buildPointCloud(source_data=Metashape.DepthMapsData)
    print("new point cloud built")
    
    # Construct DEM
    chunk.buildDem(source_data = Metashape.PointCloudData,
                   interpolation = Metashape.EnabledInterpolation,
                   region = b,
                   resolution = resM)
    print("DEM built")
    
    # Export DEM
    chunk.exportRaster(path=(writeDEMFolder + demTif_name),
                       source_data = Metashape.ElevationData,
                       image_format = Metashape.ImageFormatTIFF, 
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
 
    # Create the new orthomosaic that is associated to this DEM
    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData,fill_holes=True,resolution=resM) 
    print("DEM orthomosaic built")
    
    # Export the orthomosaic
    chunk.exportRaster(path=(writeDEMFolder + demOrtho_name),
                            source_data =  Metashape.OrthomosaicData,
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
    
    return "GeoTIFF and orthomosaic of DEM are saved in" + writeDEMFolder

#%% Orthomosaics + DEMs using individual alignment
def Orthomosaic_DEM_DSLR_indv(input_files, writeOrthoFolder, writeDEMFolder, ortho_name, demTif_name, demOrtho_name, gridRes, startXY, endXY, markerCSV, markerPos, depthMapSetting):
    '''
    Parameters
    ----------
    input_files         : Source images for orthomosaic
    writeOrthoFolder    : Folder to save the orthomosaics
    writeDEMFolder      : Folder to save the DEMs
    ortho_name          : String with orthomosaic output file name
    demTif_name         : String with GeoTIFF DEM output file name
    demOrtho_name       : String with ortho DEM output file name
    gridRes             : Pixel/grid size in mm
    startXY             : Start coordinates of X and Y
    endXY               : End coordinates of X and Y
    markerCSV           : Pixel coordinates of all markers in the DSLR imagery
    depthMapSetting     : Setting for generating new depth maps
    '''
    
    # Provide the integer for the downscaleFactor based on the depthMapSetting
    if depthMapSetting == 'ultra' or depthMapSetting == 'Ultra':
        downscaleFactor = 1
    elif depthMapSetting == 'high' or depthMapSetting == 'High':
        downscaleFactor = 2
    elif depthMapSetting == 'medium' or depthMapSetting == 'Medium':
        downscaleFactor = 4
    elif depthMapSetting == 'low' or depthMapSetting == 'Low':
        downscaleFactor = 8
    elif depthMapSetting == 'lowest' or depthMapSetting == 'Lowest':
        downscaleFactor = 16
    else:
        raise Exception("The defined depthMapSetting does not exist")
        
    # Create a new Metashape document
    doc = Metashape.Document()
    doc.clear()
    print("new metashape document created")
    
    # Save the project in PSX format before further processing, important before creating orthomosaic
    project_path = writeOrthoFolder + "project.psx" 
    doc.save(project_path)
    
    # Create a chunk
    chunk = doc.addChunk()
    
    # Add photos to the chunk
    chunk.addPhotos(input_files)
    print("photos added")
    
    # Define markers
    for i in range(np.size(markerPos,0)):
        # Get current marker details
        #markerLabel = markerPos.at[i,"marker_label"]
        markerX = markerPos.at[i,"X"]
        markerY = markerPos.at[i,"Y"]
        markerZ = markerPos.at[i,"Z"]
        chunk.addMarker(Metashape.Vector([markerX,markerY,markerZ]))
    
    # Likely the code below can be deleted
    # # If the dummyModel contains 81 images, this needs to be reduced to the DSLR survey length (which will now only work for the standard of 41 images)
    # if markerCSV.max().at["camindex"] != len(input_files)-1:
    #     # Are the lengths 81 and 41?
    #     if markerCSV.max().at["camindex"] == 80 and len(input_files) == 41:
    #         # Only store odd-indexed numbers
    #         for i in range(len(markerCSV)):
    #             if markerCSV.loc[i,"camindex"] % 2 == 1:
    #                 # Drop row
    #                 markerCSV.drop(i, inplace=True)
    #             else: # Update camindex by dividing by 2
    #                 markerCSV.loc[i,"camindex"] = markerCSV.loc[i,"camindex"]/2
    #         # Reset indices in markerCSV
    #         markerCSV = markerCSV.reset_index(drop=True)
    #     else:
    #         raise Exception('Combination of lengths of DSLRcsv and input_files are not considered')
    
    # Loop through the CSV file to digitize the GCP coordinates
    for i in range(np.size(markerCSV,0)):
        # Which marker do we want to digitize?
        marker = chunk.markers[markerCSV.at[i,"markerindex"]]
        # Setting the projection accuracy is not necessary. However if we want this in the future, this is the code.
        #marker.reference.accuracy = Metashape.Vector((markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"],markerCSV.at[i,"accuracy"]))
        # For which camera?
        camera = chunk.cameras[markerCSV.at[i,"camindex"]] # Add DSLRCams[0] to get the actual index
        # What is the X pixel value?
        Xpixel = markerCSV.at[i,"Xpixel"]
        # What is the Y pixel value?
        Ypixel = markerCSV.at[i,"Ypixel"]
        # Now digitize
        marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([Xpixel,Ypixel]), True)
        # enable        
        marker.reference.enabled = True
        del(marker,camera,Xpixel,Ypixel)   
    print("GCP's digitized")
    
    # Matching photos and aligning the cameras
    chunk.matchPhotos(downscale=1,generic_preselection=True,reference_preselection=True)
    chunk.alignCameras(adaptive_fitting = True)
    print("cameras aligned and optimized")
    
    # With the input DSLR cameras aligned, we can create the depth maps with only the input cameras
    chunk.buildDepthMaps(downscale=downscaleFactor, filter_mode=Metashape.MildFiltering)
    print("depth maps generated")
    
    # Construct new point cloud
    chunk.buildPointCloud(source_data=Metashape.DepthMapsData)
    print("point cloud built")
    
    # Build model
    chunk.buildModel(surface_type = Metashape.Arbitrary, 
                      interpolation = Metashape.EnabledInterpolation, 
                      face_count = Metashape.LowFaceCount, 
                      source_data = Metashape.DepthMapsData, 
                      vertex_colors = True)
    print("mesh created")
    
    # Export the orthomosaic to a file with compression settings
    resM = gridRes/1000 # Divide by 1000 to go from mm to m
    c = Metashape.ImageCompression()
    c.TiffCompressionLZW
    c.jpeg_quality = 99
    
    # Defining which part of the orthomosaic will be saved. We're only interested in the cartesion coordinates of x: 0-20 and y: 0-3 m
    b = Metashape.BBox()
    b.min = Metashape.Vector(startXY) # min X and min Y
    b.max = Metashape.Vector(endXY) # max X and max Y
    b.size = 2
    
    # Construct DEM
    chunk.buildDem(source_data = Metashape.ModelData,
                       interpolation = Metashape.EnabledInterpolation,
                       region = b,
                       resolution = resM)
    print("DEM built")
    
    # Construct orthomosaic
    chunk.buildOrthomosaic(surface_data=Metashape.ModelData)
    print("orthomosaic built")
    
    # For some unexplainable reason, the amount of pixels is not always constant depending on resolution,
    # So we need to specify the amount of pixels when exporting the raster
    totalWidth = endXY[0]-startXY[0]
    totalHeight = endXY[1]-startXY[1]
    pixelWidth = int(totalWidth/resM) # Round to integers
    pixelHeigth = int(totalHeight/resM) # Round to integers
    
    # Exporting the orthomosaic with predefined settings
    # Some alledgly redundant settings are added to assure a consistent pixel size
    doc.chunk.exportRaster(path=(writeOrthoFolder + ortho_name),
                            source_data =  Metashape.OrthomosaicData,
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
    print("orthomosaic is saved in" + writeOrthoFolder)
    
    # Export DEM
    chunk.exportRaster(path=(writeDEMFolder + demTif_name),
                           source_data = Metashape.ElevationData,
                           image_format = Metashape.ImageFormatTIFF, 
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
    
    # Delete the project file
    os.remove(project_path)
    shutil.rmtree(writeOrthoFolder + "project.files")
    
    return "GeoTIFF of DEM is saved in" + writeDEMFolder