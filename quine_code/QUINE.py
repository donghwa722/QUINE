from qiskit import quantum_info
from numpy.linalg import matrix_rank
import pennylane as qml
import numpy as np
import torch

def quine(num_qubits, num_layers, U, U_qubits, steps, wires):

    def layer(params, j):
        for i in range(num_qubits):
            qml.RX(params[i, j, 0], wires=wires[i])
            qml.RY(params[i, j, 1], wires=wires[i])
            qml.RZ(params[i, j, 2], wires=wires[i])
            qml.CNOT(wires=[wires[i], wires[(i+1)%num_qubits]])

    dev = qml.device('default.qubit', wires=U_qubits)

    @qml.qnode(dev)
    def rho():
        qml.QubitUnitary(U, wires=range(U_qubits))
        return qml.density_matrix(wires=wires)

    rho_rank = matrix_rank(np.array(rho()))
    print(rho_rank)

    observables = torch.zeros([rho_rank, 2**num_qubits, 2**num_qubits], dtype=torch.complex128, requires_grad=False)
    for i in range(rho_rank):   
        observables[i] = torch.tensor([[(int)(i==j and j==k) for k in range(2**num_qubits)] for j in range(2**num_qubits)])

    @qml.qnode(dev, interface='torch')
    def circuit(i, U_params):
        qml.QubitUnitary(U, wires=range(U_qubits))
        
        for j in range(num_layers):
            layer(U_params, j)
        
        return qml.expval(qml.Hermitian(observables[i], wires=wires))

    @qml.qnode(dev)
    def exact_entropy():
        qml.QubitUnitary(U, wires=range(U_qubits))
        return qml.vn_entropy(wires=wires)

    c = 1.0

    def cost_f(T_params, U_params):
        cost = 0
        for i in range(rho_rank):
            cost += torch.mul(torch.mul(- c, T_params[i]), circuit(i, U_params))
        tmp_cost = 2**(num_qubits) - rho_rank
        for i in range(rho_rank):
            tmp_cost += torch.exp(torch.mul(c, torch.abs(T_params[i])))
        return cost + torch.log(tmp_cost)

    T_params = torch.tensor(np.random.normal(0, 2, (rho_rank)), requires_grad=True)
    U_params = torch.tensor(np.random.normal(0, np.pi, (num_qubits, num_layers, 3)), requires_grad=True)

    optimizer = torch.optim.Adam([T_params, U_params], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    ex_entropy = exact_entropy()
    print("Exact entropy is {:.7f}".format(ex_entropy))

    best_cost = cost_f(T_params, U_params)
    print("Step 0: cost is {:.7f}".format(best_cost))

    time_array = []
    cost_array = []
    entropy_array = []

    for n in range(steps):
        optimizer.zero_grad()
        loss = cost_f(T_params, U_params)
        loss.backward()
        optimizer.step()

        # keeps track of best parameters
        if loss < best_cost:
            best_cost = loss
        
        # Keep track of progress every 10 steps
        if n % 5 == 4 or n == steps - 1:
            print("Step {}: cost is {:.7f}, error rate is {:.3f}%".format(n + 1, loss, (loss.detach().numpy() - ex_entropy) / ex_entropy * 100))
            time_array.append(n+1)
            cost_array.append(loss.detach().numpy())
            entropy_array.append(ex_entropy)

    return time_array, cost_array, entropy_array


def quine_2(num_qubits, num_layers, psi, psi_qubits, steps, wires):

    def layer(params, j):
        for i in range(num_qubits):
            qml.RX(params[i, j, 0], wires=wires[i])
            qml.RY(params[i, j, 1], wires=wires[i])
            qml.RZ(params[i, j, 2], wires=wires[i])
            if(num_qubits > 1): qml.CNOT(wires=[wires[i], wires[(i+1)%num_qubits]])

    dev = qml.device('default.qubit', wires=psi_qubits)

    @qml.qnode(dev)
    def rho():
        qml.StatePrep(psi, wires=range(psi_qubits))
        return qml.density_matrix(wires=wires)

    rho_rank = matrix_rank(np.array(rho()))
    print(rho_rank)

    observables = torch.zeros([rho_rank, 2**num_qubits, 2**num_qubits], dtype=torch.complex128, requires_grad=False)
    for i in range(rho_rank):   
        observables[i] = torch.tensor([[(int)(i==j and j==k) for k in range(2**num_qubits)] for j in range(2**num_qubits)])

    @qml.qnode(dev, interface='torch')
    def circuit(i, U_params):
        qml.StatePrep(psi, wires=range(psi_qubits))
        
        for j in range(num_layers):
            layer(U_params, j)
        
        return qml.expval(qml.Hermitian(observables[i], wires=wires))

    @qml.qnode(dev)
    def exact_entropy():
        qml.StatePrep(psi, wires=range(psi_qubits))
        return qml.vn_entropy(wires=wires)

    c = 1.0

    def cost_f(T_params, U_params):
        cost = 0
        for i in range(rho_rank):
            cost += torch.mul(torch.mul(- c, T_params[i]), circuit(i, U_params))
        tmp_cost = 2**(num_qubits) - rho_rank
        for i in range(rho_rank):
            tmp_cost += torch.exp(torch.mul(c, torch.abs(T_params[i])))
        return cost + torch.log(tmp_cost)

    T_params = torch.tensor(np.random.normal(0, 2, (rho_rank)), requires_grad=True)
    U_params = torch.tensor(np.random.normal(0, np.pi, (num_qubits, num_layers, 3)), requires_grad=True)

    optimizer = torch.optim.Adam([T_params, U_params], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    ex_entropy = exact_entropy()
    print("Exact entropy is {:.7f}".format(ex_entropy))

    best_cost = cost_f(T_params, U_params)
    print("Step 0: cost is {:.7f}".format(best_cost))

    time_array = []
    cost_array = []
    entropy_array = []

    for n in range(steps):
        optimizer.zero_grad()
        loss = cost_f(T_params, U_params)
        loss.backward()
        optimizer.step()

        # keeps track of best parameters
        if loss < best_cost:
            best_cost = loss
        
        # Keep track of progress every 10 steps
        if n % 5 == 4 or n == steps - 1:
            print("Step {}: cost is {:.7f}, error rate is {:.3f}%".format(n + 1, loss, (loss.detach().numpy() - ex_entropy) / ex_entropy * 100))
            time_array.append(n+1)
            cost_array.append(loss.detach().numpy())
            entropy_array.append(ex_entropy)

    return time_array, cost_array, entropy_array