'''
Function to undistort an image based on lens parameters (extracted from the baseModel). 
This lens correction is required for creating DEMs from the laserscanner.

Python script written by Brechtje A. Van Amstel and Eise W. NOTA (finalized JULY 2024)
'''

import cv2
import numpy as np

def undistort_laser_line(pixelCoords, imageHeight, imageWidth, lens_params):
    '''
    Parameters
    ----------
    pixelCoords : list of pixel coordinates for which we want to apply lens transformation
    imageHeight : image height of the raw laser camera photos
    imageWidth : image width of the raw laser camera photos
    lens_params : camera lens parameters

    Returns
    -------
    corrected_image : lens_params corrected

    '''
    # Retrieve camera parameters # The [0] assurs that only thje value is read
    fx = lens_params['fx'][0] # Focal length (pixels) in x-direction fx = f + b1
    fy = lens_params['fy'][0] # Focal length (pixels) f (=fy)
    cx = lens_params['cx'][0]+(0.5*imageWidth) # Principal point coordinates (pixels): cx, cy 
    cy = lens_params['cy'][0]+(0.5*imageHeight) # We do +50% height and width as this parameter is interpreted differently by Agisoft and openCV
    k1 = lens_params['k1'][0] # Radial distortion coefficients (-): k1, k2, k3, k4
    k2 = lens_params['k2'][0]
    k3 = lens_params['k3'][0]
    p1 = lens_params['p1'][0] # Tangential distortion coefficients (-): p1, p2
    p2 = lens_params['p2'][0]
    
    # The following paremeters are not used by openCV, but can be considered in Agisoft
    # Luckily for the laserCamera, they're all zero
    #k4 = lens_params['k4'] # k4 is not used
    #b1 = lens_params['b1'] # Affinity and non-orthogonality (skew coefficients) (pixels): b1, b2
    #b2 = lens_params['b2'] 
    #sk = b2                # b2 is the same as sk
    sk = 0 #b2 # Is always zero in openCV

    # Create the camera matrix
    camera_matrix = np.array([[fx, sk, cx], [0, fy, cy], [0, 0, 1]])

    # Create the distortion coefficients array
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    
    # THIS LAYOUT IS IMPORTANT FOR pixelCoords:
    #u = 1000
    #v= 1548.3125
    #v2 = int(v)
    #pixelCoords = np.array([[[u,v],[u,v]]], dtype = np.float64)
    #pixelCoords2 = np.array([[[u,v2],[u,v2]]], dtype = np.float64)
    
    # Undistort the image
    undistorted_pixels = cv2.undistortImagePoints(pixelCoords,camera_matrix,dist_coeffs)
    #undistorted_pixels2 = cv2.undistortImagePoints(pixelCoords2,camera_matrix,dist_coeffs)

    return undistorted_pixels


def undistort_image(input_file, lens_params):
    '''
    
    Parameters
    ----------
    input_file : raw image
    lens_params : camera lens parameters

    Returns
    -------
    corrected_image : lens_params corrected

    '''
    # Determine image height and width
    imageHeight = input_file.shape[0]
    imageWidth = input_file.shape[1]
    
    # Retrieve camera parameters # The [0] assurs that only thje value is read
    fx = lens_params['fx'][0] # Focal length (pixels) in x-direction fx = f + b1
    fy = lens_params['fy'][0] # Focal length (pixels) f (=fy)
    cx = lens_params['cx'][0]+(0.5*imageWidth) # Principal point coordinates (pixels): cx, cy 
    cy = lens_params['cy'][0]+(0.5*imageHeight) # We do +50% height and width as this parameter is interpreted differently by Agisoft and openCV
    k1 = lens_params['k1'][0] # Radial distortion coefficients (-): k1, k2, k3, k4
    k2 = lens_params['k2'][0]
    k3 = lens_params['k3'][0]
    p1 = lens_params['p1'][0] # Tangential distortion coefficients (-): p1, p2
    p2 = lens_params['p2'][0]
    
    # The following paremeters are not used by openCV, but can be considered in Agisoft
    # Luckily for the laserCamera, they're all zero
    #k4 = lens_params['k4'] # k4 is not used
    #b1 = lens_params['b1'] # Affinity and non-orthogonality (skew coefficients) (pixels): b1, b2
    #b2 = lens_params['b2'] 
    #sk = b2                # b2 is the same as sk
    sk = 0 # Is always zero in openCV

    # Create the camera matrix
    camera_matrix = np.array([[fx, sk, cx], [0, fy, cy], [0, 0, 1]])

    # Create the distortion coefficients array
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    
    # Undistort the image
    undistorted_image = cv2.undistort(input_file, camera_matrix, dist_coeffs)

    return undistorted_image