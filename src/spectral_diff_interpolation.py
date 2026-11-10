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

_ , sensor_layout = read_fits.read()

def sparse_diff(hyp_cube : np.ndarray) -> np.ndarray:

    # extract required hyper cubes
    hyp_cube_WB = WB.interpolate(hyp_cube, sensor_layout)
    hyp_cube_WB = support.reduce_intwerp(
        hyp_cube,
        hyp_cube_WB
    )

    