from qiskit import quantum_info
import QUINE
import matplotlib.pyplot as plt
import scienceplots
import scipy.stats
import scipy.special
import numpy as np

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":11})  
plt.style.use('science')


num_layers = 25

EPR = np.zeros(2**(3))
EPR[0] = EPR[-1] = 1
EPR /= np.linalg.norm(EPR)

GHZ = np.zeros(2**(4))
GHZ[0] = GHZ[-1] = 1
GHZ /= np.linalg.norm(GHZ)

time_array, A_cost_array, A_array = QUINE.quine_2(1, num_layers, EPR, 3, 150, [0])
time_array, B_cost_array, B_array = QUINE.quine_2(1, num_layers, EPR, 3, 150, [0])

time_array, BR_cost_array, BR_array = QUINE.quine_2(2, num_layers, GHZ, 4, 150, [1, 3])
time_array, AR_cost_array, AR_array = QUINE.quine_2(2, num_layers, GHZ, 4, 150, [0, 3])

c1, c2, c3, c4 = 0.1, 0.2, 0.3, 0.4

lb_cost_array = [c1 * A_cost_array[i] + c3 * B_cost_array[i] + c4 * (BR_cost_array[i] - AR_cost_array[i]) for i in range(len(time_array))]
lb_array = [c1 * A_array[i] + c3 * B_array[i] + c4 * (BR_array[i] - AR_array[i]) for i in range(len(time_array))]

plt.rc('axes', labelsize=13) 
plt.rc('legend', fontsize=13) 
plt.plot(time_array, lb_cost_array, color='b', label='Estimation')
plt.plot(time_array, lb_array, linestyle='--', color='r', label='Exact bound')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.savefig('Narrow_lb_6.png', dpi=200)
# plt.show()