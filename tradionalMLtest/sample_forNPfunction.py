import numpy as np

a=np.array([1,2,2])
b=np.ones((3,3))
print(b@a)
print('?')
D = (np.diag(np.ones(7-1),1) - np.diag(np.ones(7),0))
print(D)