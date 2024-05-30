from skimage import segmentation, color, io, filters, measure, morphology
import numpy as np
from matplotlib import pyplot as plt
import os
import random

list_point=[]
for i in range(100):
    list_point.append(random.randint(0,100))
list_point.sort()

plt.figure()
plt.plot([i for i in range(len(list_point))], list_point)
plt.show()
