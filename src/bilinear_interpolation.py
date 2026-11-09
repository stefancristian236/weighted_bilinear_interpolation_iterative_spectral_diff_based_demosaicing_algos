# make use of Mihoubit et la. paper
# weighted bilinear interpolation
# we take the original image
# create a new image where every pixel is expanded into a 5x5 (orignal pixel position is maintained)
# convert 1x1 -> 5x5
# use liniar interpolation formula to find the rest of the pixels
"""

    f(x,y) = 1 / [(x2-x1)(y2-y1)] [x2 - x x - x1][[f(Q11) f(Q12)]   [[y2 - y]
                                                  [f(Q21) f(Q22)]]  [y - y1]]

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
from scipy.signal import convolve2d
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


# definition of the unnormalized filter

F = np.array([
    [1, 2, 3, 4, 5, 4, 3, 2, 1],
    [2, 4, 6, 8, 10, 8, 6, 4, 2],
    [3, 6, 9, 12, 15, 12, 9, 6, 3],
    [4, 8, 12, 16, 20, 16, 12, 8, 4],
    [5, 10, 15, 20, 25, 20, 15, 10, 5],
    [4, 8, 12, 16, 20, 16, 12, 8, 4],
    [3, 6, 9, 12, 15, 12, 9, 6, 3],
    [2, 4, 6, 8, 10, 8, 6, 4, 2],
    [1, 2, 3, 4, 5, 4, 3, 2, 1]
])


# calculation fo the normalization factor cF

cF = 0

for idx0 in range(9):
    for idx1 in range(9):
        if idx0 % 5 == 0 and idx1 % 5 == 0:
            cF += F[idx0, idx1]

# define the H low pass filter

H_filter = F.astype(np.float32) / cF

B = 5 # grid size

def interpolate(
        hyp_image : np.ndarray,
        sensor_layout : np.ndarray = None
):
    H_sub, W_sub, num_bands = hyp_image.shape
    
    H = H_sub * 5
    W = W_sub * 5

    # create the demosaiced cube
    demoCube = np.zeros(
        (H, W, num_bands),
        dtype = np.float32
    )

    if sensor_layout is not None:
        _, _, wlidx = read_fits.collapse_wl(sensor_layout)
    else:
        wlidx = np.arange(num_bands)

    for idx0 in range(25):
        # row and col offset from original image
        r_offset = idx0 // B
        c_offset = idx0 % B
        band_idx = wlidx[idx0]

        # create the expanded layer 
        expandedLayer = np.zeros(
            (H, W),
            dtype = np.float32
        )

        # place the known pixels in theyr exact relative position
        expandedLayer[r_offset::B, c_offset::B] = hyp_image[:, :, band_idx]

        # apply the filter

        demoCube[:, :, band_idx] = convolve2d(
            expandedLayer,
            H_filter,
            mode = 'same',
            boundary = 'symm'
        )

    return demoCube