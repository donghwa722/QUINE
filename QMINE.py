import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from qiskit import quantum_info
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import csv
np.random.seed(40)

# number of qubits in the circuit
nr_qubits = 8
r_qubits = 5
# number of layers in the circuit
nr_layers = 25
# unitary size
u_size = 10

U = quantum_info.random_unitary(2**(u_size), 42).data

def init():
    qml.QubitUnitary(U, wires=[i for i in range(u_size)])

# a layer of the circuit ansatz
def layer(params, j):
    for i in range(nr_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)
        qml.CNOT(wires=[i, (i+1)%nr_qubits])

dev = qml.device("default.qubit", wires=u_size)

@qml.qnode(dev)
def density_matrix():
    init()
    return qml.density_matrix(wires=[i for i in range(nr_qubits)])

@qml.qnode(dev)
def circuit_entropy():
    init()
    return qml.vn_entropy(wires=[i for i in range(nr_qubits)])

@qml.qnode(dev, interface="torch")
def circuit(params, A, layers):
    init()

    for j in range(layers):
        layer(params, j)

    return qml.expval(qml.Hermitian(A, wires=[i for i in range(nr_qubits)]))


# rank of the density matrix
rank = matrix_rank(np.array(density_matrix()))
print(rank)

observables = Variable(torch.zeros([rank, 2**nr_qubits, 2**nr_qubits], dtype=torch.complex128), requires_grad=False)
for i in range(rank):
    observables[i] = torch.tensor([[(int)(i==j and j==k) for k in range(2**nr_qubits)] for j in range(2**nr_qubits)])

# randomly initialize parameters from a normal distribution
qnn_params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
qnn_params = Variable(torch.tensor(qnn_params), requires_grad=True)

observable_params = np.random.normal(0, nr_qubits, (rank))
observable_params = Variable(torch.tensor(observable_params), requires_grad=True)


def cost_fn(r, qnn_p, observable_p, layers):
    cost = 0
    for i in range(r):
        c = circuit(qnn_p, observables[i], layers)
        p = observable_p[i]
        cost += torch.mul(c, torch.abs(p))
    log_cost = 2**nr_qubits-r
    for i in range(r):
        p = observable_p[i]
        log_cost += torch.exp(torch.abs(p))
    return - cost + torch.log(log_cost)

entropy = circuit_entropy()

optimizer = torch.optim.Adam([qnn_params, observable_params], lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# number of steps in the optimization routine
steps = 500

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(rank, qnn_params, observable_params, nr_layers)
print("Entropy {:.7f}, Cost after 0 steps is {:.7f}".format(entropy, best_cost))

time_array = []
cost_array = []
entropy_array = []

#f = open('32-layer10-layer20-layer30.csv', 'w')
#writer = csv.writer(f)
# optimization begins
for n in range(steps):
    optimizer.zero_grad()
    loss = cost_fn(rank, qnn_params, observable_params, nr_layers)
    loss.backward()
    optimizer.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Entropy {:.7f}, Cost after {} steps is {:.7f}, error rate is {:.3f}%".format(entropy, n + 1, loss, (loss.detach().numpy()-entropy)/entropy*100))
        #writer.writerow([loss.detach().numpy()])
        time_array.append(n+1)
        cost_array.append(loss.detach().numpy())
        entropy_array.append(entropy)

print((best_cost.detach().numpy()-entropy)/entropy*100)
'''
plt.plot(time_array, pcost_array, 'r', label='rank 9 estimation')
plt.plot(time_array, cost_array, 'g', label='rank 8 estimation')
plt.plot(time_array, mcost_array, 'b', label='rank 7 estimation')
plt.plot(time_array, entropy_array, 'k', label='entropy')
plt.xlabel('Iteration', labelpad=15)
plt.ylabel('Entropy', labelpad=20)
# plt.show()
plt.savefig('32-rank7-rank8-rank9.png')
'''
'''
x   0.331 0.337
x^2 0.141 0.174
x^3 0.000
x^4 0.000
10  0.000
'''
