import numpy as np


ROWS = 181
COLS = 217

FILE = open('data.rawb',mode='r')

print("... Load input image")
img = np.fromfile(FILE, dtype = np.uint8, count = ROWS * COLS)
print("Dimension of the old image array: ", img.ndim)
print("Size of the old image array: ", img.size)


img.shape = (img.size // COLS, COLS)
print("New dimension of the array:", img.ndim)
print("----------------------------------------------------")
print(" The 2D array of the original image is: \n", img)
print("----------------------------------------------------")
print("The shape of the original image array is: ", img.shape)

# Save the output image
print("... Save the output image")
img.astype('int8').tofile('NewImage.raw')
print("... File successfully saved")
# Closing the file

FILE.close()
