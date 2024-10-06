import matplotlib.pyplot as plt
import numpy as np
r2Deg = lambda x: 180*x/np.pi
L = 0.25
k1,k2 = 2.2,5
kappa_list = np.arange(0,2.2,0.1)
kappa_list[-1] = 2.08
delta_theory = np.arctan(L*kappa_list)
delta_trans = np.arctanh(np.tan(delta_theory)/(k1*L))/k2

delta_exp = np.arctanh(kappa_list/k1)/k2
# plt.plot([i for i in range(len(kappa_list))], r2Deg(delta_theory), label="theory")
# plt.plot([i for i in range(len(kappa_list))], r2Deg(delta_exp), label="exp")
# plt.plot(r2Deg(delta_theory), r2Deg(delta_trans), label="trans")
# plt.xticks([i for i in range(len(kappa_list))],[round(k,2) for k in kappa_list])
# plt.legend()
# plt.show()

x = np.arange(0,2.2,0.1)
plt.plot(x, np.tanh(x), label="1")
plt.plot(x, np.arctan(x))
plt.plot(x, x)
plt.legend()
plt.show()