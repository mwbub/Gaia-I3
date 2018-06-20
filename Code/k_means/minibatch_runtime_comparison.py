from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import time

print(time.gmtime(0))
t0 = time.time()
t_minibatch = time.time() - t0
print(t_minibatch)