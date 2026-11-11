######################################################
######################################################
######################################################


"""
    spectral differense based demosaicing

    1. compute sparse channel difference
            delta(k,l) =  i(k) - iWB(k) * m(k) (Hadamard product)
            m(k) - binary mask
                    1 if MSFA(p) = k, where p is in Sk
            m(k) =
                    0, otherwise

    2. I(p, k) = IMSFA(p) (p) + ~delta(k,l)
    where ~delta(k,l) = delta(k, l) conv H

    NOTE: Hadamart product -> A[i, j] = B[i,j] * C[i, j]
"""

"""
    build the S matrix
        m is filled with 1 where there is a pixel
        Sk contains the pixels from the k channel
        
        Sk = hyp_cube - hyp_cub(k)0
        where in hyo_cub(k)0, the k layer if filled with 0
        all the other pixels are turned to 0
        only the target layer is intact
        repeat till S is filled
"""


import bilinear_interpolation
import matplotlib.pyplot as plt
from wcwidth.table_grapheme import INCB_CONSONANT


#######################################################
#######################################################
#######################################################

"""
    import libraries and packages
"""

import numpy as np
import bilinear_interpolation as WB
import read_fits
import support
import cube_decube

mars_img , sensor_layout = read_fits.read()

"""
    calculation of S matrix
        - go trough each layer
        - make the whole layer 0
        - decube the image
        - sustract all the pixels except the ones from target layer
"""



def extract_S (hyp_cube : np.ndarray) -> np.ndarray:
    h, w = mars_img.shape # extract image size

    S_matrix = np.zeros((h, w, 25)) # createa a blank S matrix is size h x w x 25

    mars_cube = cube_decube.cube(mars_img, sensor_layout) # cube the image

    for idx0 in range(25):
        # always make a fresh copy of the cube
        # we avoid sustracting multiple layers this way
        aux_cube = mars_cube.copy()

        aux_cube[:, :, idx0] = 0 # set the target layer to 0

        aux = cube_decube.decube(aux_cube, sensor_layout) # decube the image

        S_matrix[:, :, idx0] = mars_img - aux[:1088, :2048] # sustract the target pixels

    return S_matrix # return it

"""
    matrix m is 1 where there is a pixel
    it the same as S matrix 
    but instead of float values we put 1 on where a pixel is
"""

def extract_m(arrS : np.ndarray) -> np.ndarray:

    # create a blank m matrix and extract S matrix size
    m = np.zeros(arrS.shape)
    h, w, bands = arrS.shape

    # iterate trough each layer
    # the iterata trough the image
    # if a value is different than 0 => 1
    for idx0 in range(bands):
        for idx1 in range(h):
            for idx2 in range(w):
                if arrS[idx1, idx2, idx0] != 0:
                    m[idx1, idx2, idx0] = 1
    # NOTE: numpy can be used to speed up the process
    # NOTE: refactor to make use of vectorization for faster compute time
    return m

"""
    compute the sparse difference
"""


def sparse_diff(hyp_cube : np.ndarray) -> np.ndarray:

    _S = extract_S(cube_decube.cube(mars_img, sensor_layout))
    _m = extract_m(_S)

    hyp_cube_WB = WB.interpolate(cube_decube.cube(mars_img, sensor_layout), sensor_layout)

    prod = np.multiply(hyp_cube_WB[:1088, :2048, :], _m)
    prod_red = support.reduce_intwerp(prod, cube_decube.cube(mars_img, sensor_layout))
    S_red = support.reduce_intwerp(_S, cube_decube.cube(mars_img, sensor_layout))

    delta = S_red - prod_red

    I_crown = cube_decube.cube(mars_img, sensor_layout) + delta

    return bilinear_interpolation.interpolate(I_crown, sensor_layout)


def run_pipeline():
    im_sd = sparse_diff(cube_decube.cube(mars_img, sensor_layout))
    img = cube_decube.decube(im_sd, sensor_layout)

    return img

if __name__ == '__main__':
    run_pipeline()