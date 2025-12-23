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
