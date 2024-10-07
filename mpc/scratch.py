import matplotlib.pyplot as plt
import numpy as np
r2Deg = lambda x: 180*x/np.pi
d2R = lambda x: x*np.pi/180
L = 0.25
k1,k2 = 2.25,4
delta_list = np.arange(-26,26,2)
kappa_theory = np.tan(d2R(delta_list))/L
kappa_exp = k1*np.tanh(d2R(delta_list)*k2)

plt.plot(delta_list, kappa_theory, label="theory")
plt.plot(delta_list, kappa_exp, label="exp")
# plt.plot(r2Deg(delta_theory), r2Deg(delta_trans), label="trans")
plt.xlabel("Delta")
plt.ylabel("Curvature")
plt.title("Theoretical and Experimental Results")
plt.legend()
plt.savefig("steering_theory.jpg")
plt.show()

# x = np.arange(0,2.2,0.1)
# plt.plot(x, np.tanh(x), label="1")
# plt.plot(x, np.arctan(x))
# plt.plot(x, x)
# plt.legend()
# plt.show()