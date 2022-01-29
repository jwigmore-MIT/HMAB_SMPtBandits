import numpy as np
import matplotlib.pyplot as plt

from SMPyBandits.Distribution import NormalGamma
from SMPyBandits.Distribution import Normal


NG = NormalGamma(1,1,1,1)

N = Normal(0,1)

fig, ax = plt.subplots()
ax = N.plot(ax)
fig.show()




