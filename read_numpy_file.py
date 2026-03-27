import numpy as np
import matplotlib.pyplot as plt

arr = np.load("data/tiles/Badetumnar/tiles/tile_0010_0029.npy")

print(arr.shape)
print(arr.dtype)

plt.imshow(arr)
plt.show()