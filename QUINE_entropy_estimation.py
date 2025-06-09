from qiskit import quantum_info
import pennylane as qml
import numpy as np
import torch

num_qubits = 5

#select target density matrix
rho_rank = 4
rho = quantum_info.random_density_matrix(2**(num_qubits), rho_rank).data

num_layers = 20

def layer(params, j):
    for i in range(num_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)
        qml.CNOT(wires=[i, (i+1)%num_qubits])

dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev, interface='torch')
def circuit(i, U_params, rho):
    qml.BasisState(state=i, wires=range(num_qubits))
    
    for j in range(num_layers):
        layer(U_params, j)
    
    return qml.expval(qml.Hermitian(rho, wires=range(num_qubits)))

@qml.qnode(dev)
def exact_entropy(rho):
    qml.QubitDensityMatrix(rho, wires=range(num_qubits))
    return qml.vn_entropy(wires=range(num_qubits))

c = 1.0

def cost_f(T_params, U_params, rho):
    cost = 0
    for i in range(rho_rank):
        cost += - c * T_params[i] * circuit(i, U_params, rho)
    tmp_cost = 2**(num_qubits) - rho_rank
    for i in range(rho_rank):
        tmp_cost += torch.exp(torch.mul(c, torch.abs(T_params[i])))
    return cost + torch.log(tmp_cost)

T_params = torch.tensor(np.random.normal(0, 1, (rho_rank)), requires_grad=True)
U_params = torch.tensor(np.random.normal(0, np.pi, (num_qubits, num_layers, 3)), requires_grad=True)

optimizer = torch.optim.Adam([T_params, U_params], lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

steps = 500

ex_entropy = exact_entropy(rho)
print("Exact entropy is {:.7f}".format(ex_entropy))

best_cost = cost_f(T_params, U_params, rho)
print("Step 0: cost is {:.7f}".format(best_cost))

for n in range(steps):
    optimizer.zero_grad()
    loss = cost_f(T_params, U_params, rho)
    loss.backward()
    optimizer.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
    
    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Step {}: cost is {:.7f}, error rate is {:.3f}%".format(n + 1, loss, (loss.detach().numpy() - ex_entropy) / ex_entropy * 100))
        
