'''
The following script contains functions that can be used for debayering and color manipulation of Overhead Imagery of the Metronome

Written by Brechtje A. van Amstel and Eise W. NOTA (finalized JUNE 2024)
'''

import colour_demosaicing as cd

# Debayering the grayscale images
def debayer_bmp(input_file):
    # Load the 3D BMP file
    raw_image = input_file
    # Debayer to RGB
    RGB_img = cd.demosaicing_CFA_Bayer_bilinear(raw_image, "GRBG")  
    return RGB_img

def rgb_to_LAB(input_file):
    # Convert to LAB
    LAB_file = cv2.cvtColor(input_file, cv2.COLOR_RGB2Lab)   
    return LAB_file

def rgb_to_Luv(input_file):
    # Convert to Luv
    Luv_file = cv2.cvtColor(input_file, cv2.COLOR_RGB2Luv)
    return Luv_file

def rgb_to_YCrCb(input_file):
    # Convert to LAB
    LAB_file = cv2.cvtColor(input_file, cv2.COLOR_RGB2YCrCb)   
    return LAB_file

def rgb_to_HSV(input_file):
    # Convert to LAB
    LAB_file = cv2.cvtColor(input_file, cv2.COLOR_RGB2HSV)   
    return LAB_file
