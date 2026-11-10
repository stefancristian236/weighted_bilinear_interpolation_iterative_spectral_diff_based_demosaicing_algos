###################################################################
###################################################################
###################################################################


"""
    libraries and functions definition
"""




from astropy.io import fits
import numpy as np


###################################################################
###################################################################
###################################################################


"""
    define the path to the source data

"""

mars = '../assets/HS-L0-FE-025D-05-01_1C.fits'
sensor = '../assets/CRS-HERA-SW101-HyperScout-H-detector-per-pixel-central-wavelength-map.fits'

###################################################################
###################################################################
###################################################################



# define the function which opens and extracts the wavelengths and reflectance values

def read(image : np.ndarray,
         layout : np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # open the image
    with fits.open(image) as hdul:
        mars_data = hdul[0].data

    # open the wl sensor layout
    with fits.open(layout) as hdul:
        layout_data = hdul[0].data

    # return the reflenctance and wavelengths as numpy arrays
    return mars_data, layout_data



# collapse the 2048 x 1088 wl array into a size 25 array by median values

def collapse_wl(wl_layout : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # get dimensions
    h, w = wl_layout[:2].shape

    # make dimensions divisible by 5
    h_trim = h - (h % 5)
    w_trim = w - (w % 5)

    # remove the imcompleted edges
    layout_trim = wl_layout[:h_trim, :w_trim]

    # reshape to 5 x 5 structure
    # n rows x 5 rows x m columns x 5 columns tensor
    reshaped = layout_trim.reshape(h_trim // 5, 5, w_trim // 5, 5)

    # collapse corresponding positions across all tiles into one representative 5 x 5 layout
    raw_wl = np.median(reshaped, axis = (0, 2))

    wl_flatten = raw_wl.flatten() # flatten the sensor matrix
    sorted_wl = np.sort(wl_flatten) # sort the wavelenghts
    wlidx = np.argsort(np.argsort(wl_flatten)) # extract the sorted wavelengths

    return raw_wl, sorted_wl, wlidx