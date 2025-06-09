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

A_qubits = 2
B_qubits = 2
R_qubits = 2

k = 2

num_layers = 25

P = np.random.normal(-1, 1, (2**(A_qubits), 2**(B_qubits), 2**(R_qubits)))
for a in range(2**(A_qubits)):
    for b in range(2**(B_qubits)):
        for c in range(2**(R_qubits)):
            if a < k and b < k and a > b:
                P[a][b][c]=P[b][a][c]
            if (a >= k and b < k) or (a < k and b >= k):
                P[a][b][c]=0

num_qubits = A_qubits + B_qubits + R_qubits
state = np.zeros(2**(num_qubits))

for a in range(2**(A_qubits)):
    for b in range(2**(B_qubits)):
        for c in range(2**(R_qubits)):
            state[a + b * (2**A_qubits) + c * (2**(A_qubits + B_qubits))] = P[a][b][c]

state /= np.linalg.norm(state)

time_array, AB_cost_array, AB_array = QUINE.quine_2(A_qubits + B_qubits, num_layers, state, num_qubits, 500, range(A_qubits + B_qubits))

num_qubits = A_qubits + B_qubits + R_qubits + A_qubits + B_qubits
st_state = np.zeros(2**(num_qubits))
print(st_state.size)

for a in range(2**(A_qubits)):
    for b in range(2**(B_qubits)):
        for c in range(2**(R_qubits)):
            if(a < k and b < k):
                st_state[a + b * (2**A_qubits) + c * (2**(A_qubits + B_qubits))] = P[a][b][c]
                print(a + b * (2**A_qubits) + c * (2**(A_qubits + B_qubits)), a, b, c, P[a][b][c])
            if(a >= k and b >= k):
                st_state[(2**A_qubits - 1) + (2**B_qubits - 1) * (2**A_qubits) + c * (2**(A_qubits + B_qubits)) + a * (2**(A_qubits + B_qubits + R_qubits)) + b * ((2**(A_qubits + B_qubits + R_qubits + A_qubits)))] = P[a][b][c]

#print(state)

st_state /= np.linalg.norm(st_state)

print(st_state)

index_AR = [i for i in range(A_qubits)] + [i for i in range(A_qubits + B_qubits, A_qubits + B_qubits + R_qubits)]

time_array, A_cost_array, A_array = QUINE.quine_2(A_qubits, num_layers, st_state, num_qubits, 500, range(A_qubits))
time_array, AR_cost_array, AR_array = QUINE.quine_2(A_qubits + R_qubits, num_layers, st_state, num_qubits, 500, index_AR)

n_ub_cost_array = [abs(AR_cost_array[i]-A_cost_array[i]) for i in range(len(time_array))]
n_ub_array = [AR_array[i]-A_array[i] for i in range(len(time_array))]

b_ub_cost_array = [AB_cost_array[i] for i in range(len(time_array))]
b_ub_array = [AB_array[i] for i in range(len(time_array))]

plt.figure(figsize=(8,4))
plt.rc('axes', labelsize=13) 
plt.rc('legend', fontsize=13) 
plt.plot(time_array, n_ub_cost_array, color='b', label='Estimation')
plt.plot(time_array, n_ub_array, linestyle='--', color='r', label='Tight bound')
plt.plot(time_array, b_ub_cost_array, color='g', label='Estimation')
plt.plot(time_array, b_ub_array, linestyle='--', color='#E69F00', label='Loose bound')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.savefig('Narrow_and_Broad_ub.png', dpi=200)
# plt.show()