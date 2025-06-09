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

A_qubits = 1
B_qubits = 1
R_qubits = 2

k = 2

P = np.random.normal(-1, 1, (2**(A_qubits), 2**(B_qubits)))
for _ in range(2**(A_qubits)):
    for __ in range(2**(B_qubits)):
        if _ < k and __ < k and _ > __:
            P[_][__]=P[__][_]
        if (_ >= k and __ < k) or (_ < k and __ >= k):
            P[_][__]=0



num_qubits = A_qubits + B_qubits + R_qubits + A_qubits + B_qubits
state = np.zeros(2**(num_qubits))
print(state.size)

for i in range(2**(A_qubits)):
    for j in range(2**(B_qubits)):
        if(i < k and j < k):
            state[i + j * (2**(A_qubits)) + (i + j * (2**(A_qubits))) * (2**(A_qubits + B_qubits))] = P[i][j]
            print(i + j * (2**(A_qubits)) + (i + j * (2**(A_qubits))) * (2**(A_qubits + B_qubits)), i, j, P[i][j])
        if(i >= k and j >= k):
            state[2**(A_qubits) - 1 + (2**(B_qubits) - 1) * 2**(A_qubits) + (i + j * (2**(A_qubits))) * (2**(A_qubits + B_qubits)) + i * (2**(A_qubits + B_qubits + R_qubits)) + j * (2**(A_qubits + B_qubits + R_qubits + A_qubits))] = P[i][j]

state /= np.linalg.norm(state)

print(state)

num_layers = 25

index_AR = [i for i in range(A_qubits)] + [i for i in range(A_qubits + B_qubits, A_qubits + B_qubits + R_qubits)]

time_array, AR_cost_array, AR_array = QUINE.quine_2(A_qubits + R_qubits, num_layers, state, num_qubits, 500, index_AR)
time_array, A_cost_array, A_array = QUINE.quine_2(A_qubits, num_layers, state, num_qubits, 500, range(A_qubits))

ub_cost_array = [AR_cost_array[i]-A_cost_array[i] for i in range(len(time_array))]
ub_array = [AR_array[i]-A_array[i] for i in range(len(time_array))]

plt.rc('axes', labelsize=13) 
plt.rc('legend', fontsize=13) 
plt.plot(time_array, ub_cost_array, color='b', label='Estimation')
plt.plot(time_array, ub_array, linestyle='--', color='r', label='Exact bound')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.savefig('Narrow_ub_4.png', dpi=200)
# plt.show()