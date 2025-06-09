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

A_qubits = 3
B_qubits = 3

num_qubits = A_qubits + B_qubits

#select target density matrix
U_qubits = 8
U = quantum_info.random_unitary(2**(U_qubits)).data

num_layers = 25

is_ub = False
is_poster = True

if is_ub:
    time_array, ub_cost_array, ub_array = QUINE.quine(num_qubits, num_layers, U, U_qubits, 500, range(num_qubits))

else:
    time_array_1, A_cost_array, A_entropy_array = QUINE.quine(A_qubits, num_layers, U, U_qubits, 250, range(0, A_qubits))
    time_array_2, B_cost_array, B_entropy_array = QUINE.quine(B_qubits, num_layers, U, U_qubits, 250, range(A_qubits, num_qubits))

    time_array = time_array_1 + time_array_2
    print(time_array)

    lb_cost_array = [abs(B_cost_array[i]-A_cost_array[i]) for i in range(len(time_array))]
    lb_array = [abs(B_entropy_array[i]-A_entropy_array[i]) for i in range(len(time_array))]

plt.rc('axes', labelsize=13) 
plt.rc('legend', fontsize=13) 

if is_ub:
    if is_poster:
        plt.plot(time_array, ub_cost_array, color='b', label='Estimation')
        plt.plot(time_array, ub_array, linestyle='--', color='r', label='Exact bound')
    else:
        plt.plot(time_array, ub_cost_array, color='b', label='Estimation ($S(AB)$)')
        plt.plot(time_array, ub_array, linestyle='--', color='r', label='Exact bound ($S(AB)$)')

#plt.plot(time_array, B_cost_array, color='gray', label='Estimation ($S(B)$)')
#plt.plot(time_array, B_entropy_array, linestyle='--', color='gray', label='Exact ($S(B)$)')
#plt.plot(time_array, A_cost_array, color='gray', label='Estimation ($S(A)$)')
#plt.plot(time_array, A_entropy_array, linestyle='--', color='gray', label='Exact ($S(A)$)')
else:
    if is_poster:
        plt.plot(time_array, lb_cost_array, color='b', label='Estimation')
        plt.plot(time_array, lb_array, linestyle='--', color='r', label='Exact bound')

    else:
        plt.plot(time_array, lb_cost_array, color='b', label='Estimation ($|S(B)-S(A)|$)')
        plt.plot(time_array, lb_array, linestyle='--', color='r', label='Exact bound ($|S(B)-S(A)|$)')
    
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
if is_ub:
    plt.savefig('Borad_ub_6_pos.png', dpi=200)
else:
    plt.savefig('Borad_lb_6_pos.png', dpi=200)
# plt.show()