import numpy as np



def reduce_intwerp(
        hyp_cube_interp : np.ndarray,
        hyp_cube_original : np.ndarray
):

    H, W, B = hyp_cube_interp.shape
    h, w = hyp_cube_original.shape[:2]

    pad_h = (h * 5) - h
    pad_w = (w * 5) - w

    padded = np.pad(
        hyp_cube_interp,
        pad_width= ((0, pad_h), (0, pad_w), (0, 0)),
        mode = 'constant',
        constant_values=np.nan
    )

    for idx in range(25):
        padded[:, :, idx] = hyp_cube_interp[:, :, idx].reshape(218, 5, 410, 5)

    return np.nanmedian(padded, axis = (1, 3))