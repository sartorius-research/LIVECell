import cv2
import numpy as np

#this function is designed to adapt images acquired with other light microscopy modalities
#in order to enable inference with LIVECell trained models

#input_image = uint8 numpy array
#magnification_resample_factor = downsample factor needed to make image effective 10X
#   for 40x images, use 0.25
#   for 20x images, use 0.5
#   for 10x images, use 1

def preprocess(input_image, magnification_downsample_factor=1.0): 
    #internal variables
    #   median_radius_raw = used in the background illumination pattern estimation. 
    #       this radius should be larger than the radius of a single cell
    #   target_median = 128 -- LIVECell phase contrast images all center around a 128 intensity
    median_radius_raw = 75
    target_median = 128.0
    
    #large median filter kernel size is dependent on resize factor, and must also be odd
    median_radius = round(median_radius_raw*magnification_downsample_factor)
    if median_radius%2==0:
        median_radius=median_radius+1

    #scale so mean median image intensity is 128
    input_median = np.median(input_image)
    intensity_scale = target_median/input_median
    output_image = input_image.astype('float')*intensity_scale

    #define dimensions of downsampled image image
    dims = input_image.shape
    y = int(dims[0]*magnification_downsample_factor)
    x = int(dims[1]*magnification_downsample_factor)

    #apply resizing image to account for different magnifications
    output_image = cv2.resize(output_image, (x,y), interpolation = cv2.INTER_AREA)
    
    #clip here to regular 0-255 range to avoid any odd median filter results
    output_image[output_image > 255] = 255
    output_image[output_image < 0] = 0

    #estimate background illumination pattern using the large median filter
    background = cv2.medianBlur(output_image.astype('uint8'), median_radius)
    output_image = output_image.astype('float')/background.astype('float')*target_median

    #clipping for zernike phase halo artifacts
    output_image[output_image > 180] = 180
    output_image[output_image < 70] = 70
    output_image = output_image.astype('uint8')

    return output_image

def preprocess_fluorescence(input_image, bInvert=True, magnification_downsample_factor=1.0): 
    #invert to bring background up to 128
    img = (255-input_image)/2
    if not bInvert:
        img = 255-img
    output_image = preprocess(img, magnification_downsample_factor=magnification_downsample_factor) 
    return output_image