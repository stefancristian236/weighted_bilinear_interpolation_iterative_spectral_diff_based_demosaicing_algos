# make use of Mihoubit et la. paper
# weighted bilinear interpolation
# I* = I conv H, where I is a band(2D array, a sllice of the cube) and H a low pass filter
# H can be modeled like a 2D FIR filter so we need to test out multiple Window functions
# 1st try a Box Kernel - uniform filter, but it migh smooth or wash out
# 2nd try a Gaussian Kernel

