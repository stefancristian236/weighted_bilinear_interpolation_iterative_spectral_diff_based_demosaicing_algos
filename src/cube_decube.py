###################################################################
###################################################################
###################################################################


"""
    libraries and functions definition
"""

import numpy as np
from read_fits import collapse_wl as cw

###################################################################
###################################################################
###################################################################




""""
    pack the 2D hyperspectral image into a 3D cube(h // 5, w // 5, 25)
    return the cube as an ndarray

    the function takes as input :
        - the hyperspectral image
        - the MSFA layout
"""

def cube(
        hyperspectral_image : np.ndarray,
        sensor_layout : np.ndarray,
        fill_value: float = 0.0
):
    # extract the MSFA filter featured arrays

    _, _, wlidx = cw(sensor_layout)


    # define the cube parameters

    x_side, y_side = hyperspectral_image.shape
    cube_size = 25
    x_out = int(np.ceil(x_side / 5) * 5)
    y_out = int(np.ceil(y_side / 5) * 5)


    # start building the empy cube

    padded_data = np.pad(
        hyperspectral_image,
        ((0, x_out - x_side), (0, y_out - y_side)),
        mode = "constant",
        constant_values = fill_value,
    )    

    mx, my = x_out // 5, y_out // 5
    packed_cube = np.zeros((mx, my, cube_size), dtype = float)


    # slice the original map according to MSFA grid

    for idx0 in range(5):
        for idx1 in range(5):
            flat_idx = idx0 * 5 + idx1
            slice = wlidx[flat_idx]
            subgrid = padded_data[idx0::5, idx1::5]
            packed_cube[:, :, slice] = subgrid

    # crop the macropixel

    mx_valid = int(np.ceil(x_side / 5))
    my_valid = int(np.ceil(y_side / 5))

    return packed_cube[:mx_valid, :my_valid, :]

    


################################################################
################################################################
################################################################





"""
    unpack the 3D cube into a 2D(h x w) image
"""


def decube (
        cubed_image : np.ndarray,
        sensor_layout : np.ndarray
):

    # extract cube shape
    mx, my, _ = cubed_image.shape


    raw_wl, sorted_wl, wlidx = cw(sensor_layout)
    flat_wl = raw_wl.flatten()
    # find the matchign distance based on distance

    match_dist = np.abs(sorted_wl[:, None] - flat_wl[None, :])
    band_pos = np.argmin(match_dist, axis = 0)
    band_pos = band_pos.reshape(5, 5)

    # resize to the original size

    size_x, size_y = mx * 5, my * 5
    recon_image = np.zeros((size_x, size_y), dtype = float)

    # reconstruct the image
    for idx0 in range(5):
        for idx1 in range(5):
            band_idx = band_pos[idx0, idx1]
            recon_image[idx0::5, idx1::5] = cubed_image[:, :, band_idx]

    return recon_image