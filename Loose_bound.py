from qiskit import quantum_info
import QUINE
import matplotlib.pyplot as plt
import scienceplots
import scipy.stats
import scipy.special
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":11})  
plt.style.use('science')

A_qubits = 4
B_qubits = 4

num_qubits = A_qubits + B_qubits

#select target density matrix
U_qubits = 10
U = quantum_info.random_unitary(2**(U_qubits)).data

num_layers = 25

is_ub = False

if is_ub:
    time_array, cost_array, array = QUINE.quine(num_qubits, num_layers, U, U_qubits, 400, range(num_qubits), 1.0)

else:
    time_array_1, A_cost_array, A_entropy_array = QUINE.quine(A_qubits, num_layers, U, U_qubits, 250, range(0, A_qubits), 1.0)
    
    U_qubits = 10
    U = quantum_info.random_unitary(2**(U_qubits)).data

    time_array_2, B_cost_array, B_entropy_array = QUINE.quine(B_qubits, num_layers, U, U_qubits, 250, range(A_qubits, num_qubits), 1.0)

    time_array = time_array_1

    cost_array = [abs(B_cost_array[i]-A_cost_array[i]) for i in range(len(time_array))]
    array = [abs(B_entropy_array[i]-A_entropy_array[i]) for i in range(len(time_array))]
    
df = pd.DataFrame({
    'Iteration' : time_array,
    'Estimation' : cost_array,
    'Exact' : array
})

df.to_csv('data/loose_lb_8.csv')

#plt.figure(figsize=(8, 4))

plt.rc('axes', labelsize=13) 
plt.rc('legend', fontsize=13) 

plt.plot(time_array, cost_array, color='b', label='Estimation', linewidth = 2)
plt.plot(time_array, array, linestyle='--', color='r', label='Exact bound', linewidth = 2)


plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.show()