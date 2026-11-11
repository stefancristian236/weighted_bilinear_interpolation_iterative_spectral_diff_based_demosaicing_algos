import numpy as np

"""
    NOTE: the file needs a better name
"""


"""
    to test out the interpolation algorithms
    take 5 x 5 pixels and mediate them
"""


def reduce_interp(
        hyp_cube_interp : np.ndarray,
        hyp_cube_original : np.ndarray
):

    # extract everu cube sides
    # calculate the new size
    H, W, _ = hyp_cube_interp.shape
    h, w, _ = hyp_cube_original.shape

    pad_h = (h * 5) - H
    pad_w = (w * 5) - W

    padded = np.pad(
        hyp_cube_interp,
        pad_width= ((0, pad_h), (0, pad_w), (0, 0)),
        mode = 'constant',
        constant_values=np.nan
    )


    H_pad, W_pad, C = padded.shape
    new_h, new_w = H_pad // 5, W_pad // 5
    reduced_cube = np.empty((new_h, new_w, C), dtype=padded.dtype)

    # iterate trough each layer and apply median
    for idx in range(C):
        layer_blocks = padded[:, :, idx].reshape(new_h, 5, new_w, 5)
        reduced_cube[:, :, idx] = np.nanmedian(layer_blocks, axis=(1, 3))

    # return the cube
    return reduced_cube