import numpy as np

arr = np.zeros((3,6,6,34))
arr2 = arr[:, [2,3,4,5], : , :]
arr3 = arr2[:, :, [2,3,4,5] , :]
print(arr3.shape)