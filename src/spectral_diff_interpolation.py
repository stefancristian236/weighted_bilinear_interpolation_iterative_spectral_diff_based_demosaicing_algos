######################################################
######################################################
######################################################


"""
    spectral differense based demosaicing

    1. compute sparse channel difference
            delta(k,l) =  i(k) - dot_prod(iWB(k), m(k))
            m(k) - binary mask
                    1 if MSFA(p) = k, where p is in Sk
            m(k) =
                    0, otherwise



    2. I(p, k) = IMSFA(p) (p) + ~delta(k,l)
    where ~delta(k,l) = delta(k, l) conv H
"""
import bilinear_interpolation
import matplotlib.pyplot as plt
from wcwidth.table_grapheme import INCB_CONSONANT

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

def extract_S (hyp_cube : np.ndarray) -> np.ndarray:
    h, w = mars_img.shape

    S_matrix = np.zeros((h, w, 25))

    mars_cube = cube_decube.cube(mars_img, sensor_layout)

    for idx0 in range(25):
        aux_cube = mars_cube.copy()
        aux_cube[:, :, idx0] = 0
        aux = cube_decube.decube(aux_cube, sensor_layout)
        S_matrix[:, :, idx0] = mars_img - aux[:1088, :2048]

    return S_matrix

def extract_m(arrS : np.ndarray) -> np.ndarray:

    m = np.zeros(arrS.shape)

    for idx0 in range(arrS.shape[0]):
        for idx1 in range(arrS.shape[1]):
            for idx2 in range(arrS.shape[2]):
                if arrS[idx0, idx1, idx2] > 0:
                    m[idx0, idx1, idx2] = 1

    return m


def sparse_diff(hyp_cube : np.ndarray) -> np.ndarray:

    _S = extract_S(cube_decube.cube(mars_img, sensor_layout))
    _m = extract_m(_S)

    hyp_cube_WB = WB.interpolate(cube_decube.cube(mars_img, sensor_layout), sensor_layout)

    prod = hyp_cube_WB[:1088, :2048, :] * _m
    print(prod.shape)
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