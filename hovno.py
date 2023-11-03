import numpy as np

def unique_numbers_2d_numpy(input_2d_array):
    unique_numbers = np.unique(input_2d_array)
    return unique_numbers

# Example usage:
original_2d_array = np.array([[1, 2, 2], [3, 4, 4], [5, 6, 5]])
unique_result = unique_numbers_2d_numpy(original_2d_array)
print(unique_result)
