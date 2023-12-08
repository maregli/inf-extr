import numpy as np
import time

print("Numpy version: ", np.__version__)

a = np.array([1, 2, 3])
print("a: ", a)

# Count to 50 with 1 second delay
for i in range(50):
    print(i)
    time.sleep(1)