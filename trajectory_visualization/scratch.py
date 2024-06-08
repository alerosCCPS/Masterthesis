from decimal import Decimal
import numpy as np

toDecimal= lambda x: Decimal(x)

p = np.array([[0,0],[0.5,0],[1.5,1],[1.5,3]])
p = np.vectorize(toDecimal)(p)
t = Decimal(Decimal(0.1))
rtn = p[0] * Decimal(1.0) * t ** 0 * (1 - t) ** 3 + p[1] * Decimal(3.0) * t ** 1 * (1 - t) ** 2 + \
              p[2] * Decimal(3.0) * t ** 2 * (1 - t) ** 1 + p[3] * Decimal(1.0) * t ** 3 * (1 - t) ** 0

print(0**0)