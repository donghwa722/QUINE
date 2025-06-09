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

c = np.random.normal(-1, 1, (4))
c /= np.linalg.norm(c)

r = np.zeros(4)
r[0] = c[0] * c[0]
r[1] = c[1] * c[1]
r[2] = c[2] * c[2]
for i in range(4):
    r[3] += - c[i] * c[i] * np.log2(c[i] * c[i])

print(c)

state = np.zeros((2**9))
state[0] = c[0] / np.sqrt(2)
state[1 + 1 * (2**6)] = c[0] / np.sqrt(2)
state[2 + 1 * (2**6)] = c[0] / np.sqrt(2)
state[1 + 1 * (2**6)] = c[0] / np.sqrt(2)
state[1 + 1 * (2**6)] = c[0] / np.sqrt(2)
state[1 + 1 * (2**6)] = c[0] / np.sqrt(2)
state[1 + 1 * (2**6)] = c[0] / np.sqrt(2)


lb_cost_array = [r[0] * A_cost_array[i] + r[1] * B_cost_array[i] + r[3] * (BR_cost_array[i] - AR_cost_array[i]) for i in range(len(time_array))]
lb_array = [r[0] * A_array[i] + r[1] * B_array[i] + r[3] * (BR_array[i] - AR_array[i]) for i in range(len(time_array))]

plt.rc('axes', labelsize=13) 
plt.rc('legend', fontsize=13) 
plt.plot(time_array, lb_cost_array, color='b', label='Estimation')
plt.plot(time_array, lb_array, linestyle='--', color='r', label='Exact bound')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.savefig('Narrow_lb_6.png', dpi=200)
# plt.show()