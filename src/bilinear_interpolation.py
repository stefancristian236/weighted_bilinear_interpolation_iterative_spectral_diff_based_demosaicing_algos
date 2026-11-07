# make use of Mihoubit et la. paper
# weighted bilinear interpolation
# we take the original image
# create a new image where every pixel is expanded into a 5x5 (orignal pixel position is maintained)
# convert 1x1 -> 5x5
# use liniar interpolation formula to find the rest of the pixels
"""

    f(x,y) = 1 / [(x2-x1)(y2-y1)] [x2 - x x - x1][f(Q11) f(Q12)   [y2 - y
                                                  f(Q21) f(Q22)]   y - y1]

    alternative

    using Mihoubi et la. 
    for each layer use an 9x9 low pass filter 
    
    start with a 9x9 unnormalize filter F
    calcualte the normalization facto cF
    H(a,b) = F / cF

"""

########################################################
########################################################
########################################################

"""
    
    libraries and functions definition
    
"""

import numpy as np
import matplotlib.pyplot as plt
import read_fits
import cube_decube




#############################################################
#############################################################
#############################################################


"""
[[ 1  2  3  4  5  4  3  2  1]
 [ 2  4  6  8 10  8  6  4  2]
 [ 3  6  9 12 15 12  9  6  3]
 [ 4  8 12 16 20 16 12  8  4]
 [ 5 10 15 20 25 20 15 10  5]    =   F
 [ 4  8 12 16 20 16 12  8  4]
 [ 3  6  9 12 15 12  9  6  3]
 [ 2  4  6  8 10  8  6  4  2]
 [ 1  2  3  4  5  4  3  2  1]]

"""


