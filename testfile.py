import scipy.stats as stats
import numpy as np

# Create an array of pixel values
array = np.array([[10, 20, 30], [40, 50, 60], [None, 80, 90]])

# Find the index of the pixel value that we do not want to change
index = np.where(array == 50)[0][0]

# Create a mask array
mask = np.zeros_like(array)
mask[index, index] = 1

# Normalize the values in the array, excluding the pixel value
normalized_array = stats.zscore(array, axis=None, nan_policy='omit')

# Apply the inverse of the mask array to the normalized array
restored_array = normalized_array * mask + array * (1 - mask)

# Print the normalized array with the excluded pixel value
print(normalized_array)
