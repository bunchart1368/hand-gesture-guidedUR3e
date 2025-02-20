# test_math3d.py
import math3d as m3d
import numpy as np
import math
import re


x = [-0.1710,  0.3119,  0.5579,   -1.4006, -0.6485,  0.7004]
# x = [1.2,1,1,1,1,1]
y = [0.1,0.1,0.1,0.1,0.1,0.1]
x = m3d.Transform(x)
y = m3d.Transform(y)
z = x * y
print(z)

