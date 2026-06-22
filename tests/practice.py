
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method

	arr = np.array(a)
	original_shape = arr.shape
	element_num = original_shape[0] * original_shape[1]

	if (new_shape[0]*new_shape[1]) == element_num:
		# print(new_shape)
		return arr.reshape(new_shape)

	return []

print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (4, 2)))

# np.reshape()