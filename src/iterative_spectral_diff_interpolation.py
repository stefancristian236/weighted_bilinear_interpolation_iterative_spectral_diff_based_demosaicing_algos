############################################################################################
############################################################################################
############################################################################################

"""
    Iterative Spectral Difference(ISD)

    compiute the correlation between channels
    calculate the number of iterations as N[k][l]
    find a proper sigma hyperparameter to calculate the number of iterations

    Spectrial diff is the initial iteration
"""

# define libraries and functions

import numpy as np
import read_fits


# calculation of the N[k][l] iterations
def calculate_iterations(hyp_cube : np.ndarray, sigma : np.float16) -> np.ndarray:
    _, wl_layout = read_fits.read()
    _, sorted_wl, _ = read_fits.collapse_wl(wl_layout)
    # extract the hyperspectral cube size
    h, w, bands = hyp_cube.shape

    # iterate and calculate an iteration matrix
    # calcualte the iteartion matrix only for diff wavelengths
    # N is of size 25x25
    N_maxtrix = np.zeros((bands, bands))

    # for simplicty and blueprint sake calculate everything
    # NOTE: it not neceasrry to calculate very difference
    # as |band[k] - band[l]| = |band[l] - band[k]|
    for idx0 in range(bands):
        for idx1 in range(bands):
            wl_diff = np.abs(sorted_wl[idx0] - sorted_wl[idx1])
            wl_frac = (wl_diff - 100) / (20 * sigma)
            wl_exp = np.exp(-wl_frac)
            N_maxtrix[idx0, idx1] = np.ceil(wl_exp)

    return N_maxtrix