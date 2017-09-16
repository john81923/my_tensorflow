import numpy as np
from scipy import stats

batch_size = 100
data_num = 5011

aa = np.zeros((224,224,3  ))

bb = []
for i in range(10):
    bb.append(aa)
bb = np.asarray(bb, dtype='int')

aa = np.array([1,2.1,3,4,5])
aa =stats.threshold(aa, threshmin=2,  newval=0)
aa =stats.threshold(aa, threshmax=2,  newval=1)
print aa

