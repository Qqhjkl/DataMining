import numpy as np
import math as m

x = [3.2,3.6,3,6,2.5,1.1]
a = np.var(x,ddof=1)
b = np.average(x)
print(a)
print(b)
y = 1.1
val = m.pi
e = (y-b)**2 / 2*a
p = m.exp(-e) / (2*val*a)**0.5
print(p)
