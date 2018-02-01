import numpy as np 

a = np.array([1, 2, 3])
b = np.array([4, 3, 2])
print a.shape
c =  np.outer( a, b)
print c
