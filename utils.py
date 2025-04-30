import itertools
import numpy as np
from scipy.linalg import sqrtm, expm
import qiskit
from qiskit_aer import Aer
Aer_simulator = Aer.get_backend('statevector_simulator')

def kron_list(op_list): # not currently used
    ret = op_list[0]
    for x in op_list[1:]:
        ret = np.kron(ret, x)
    return ret


def parse_qecc_str(qecc_str):
    """
    parse the string of qecc_str and return a dictionary
    """
    assert (qecc_str[:2]=='((') and ('))[' in qecc_str) and (qecc_str[-1]==']')
    tmp0 = qecc_str[2:-1]
    tmp1,tmp2 = tmp0.split('))[',1)
    tmp3 = tmp1.split(',',2)
    num_qubit = int(tmp3[0])
    num_logical_dim = int(tmp3[1])
    if '=' in tmp3[2]:
        # ((6,2,de(2)=4))[8]
        weight_z = float(tmp3[2].split('(',1)[1].split(')',1)[0])
        distance = int(tmp3[2].split('=',1)[1])
    else:
        weight_z = None
        distance = int(tmp3[2])
    num_layer = int(tmp2)
    ret = dict(num_qubit=num_qubit, num_logical_dim=num_logical_dim,
            weight_z=weight_z, weight=distance, num_layer=num_layer)
    return ret


def make_all_to_all_connection(num_qubit):
    ret = np.array([(x, y) for x in range(num_qubit) for y in range(x + 1, num_qubit)])
    return ret


def Debbie_amplitude_damping_noise(num_qubit, t, gamma): # this is the function used in the current version
    """
    num_qubit: int
    gamma: float

    return ret: np.ndarray
    """
    A0 = np.array([[1,0], [0, np.sqrt(1-gamma)]])
    A1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    e_Kraus_list = [A0, A1]

    ret = []
    t_local_kraus=[]
    for error_weight in range(1,t+1):
        for c in itertools.combinations(np.arange(num_qubit), error_weight):
            c=list(c)
            c_aux = [-1]+c

            for i in range(1,len(c_aux)):
                temp = np.eye(1)
                for j in range(c_aux[i]-c_aux[i-1]-1):
                    temp = np.kron(temp, A0)
                temp = np.kron(temp, A1)
            for i in range(c_aux[len(c_aux)-1]+1,num_qubit):
                temp = np.kron(temp, A0)
            t_local_kraus.append(temp)
    temp = np.eye(1)
    for i in range(num_qubit):
        temp = np.kron(temp, A0)
    t_local_kraus.append(temp)
    for i in range(len(t_local_kraus)):
        for j in range(len(t_local_kraus)):
            ret.append(np.conj(t_local_kraus[i]).T @ t_local_kraus[j])
            
    return ret


def run_circuit(num_qubit, num_layer, params, code_ind, connection_list):
    """
    num_qubit: int
    num_layer: int
    params: np.ndarray
    code_ind: int, this is the decimal representation of qubits state
    connection_list: np.ndarray [change it to all to all]

    return ret: np.ndarray
    """

    qr = qiskit.QuantumRegister(num_qubit)
    circ = qiskit.QuantumCircuit(qr)

    tmp0 = [x for x,y in enumerate(bin(code_ind)[-1:1:-1]) if y=='1'] # convert the decimal representation of qubits state to binary
    for x in tmp0:
        circ.x(x)

    params_s = params[:2 * num_qubit * (num_layer + 1)].reshape([num_layer + 1, num_qubit, 2]) 
    params_d = params[2 * num_qubit * (num_layer + 1):].reshape([num_layer, len(connection_list)])
    for ind0 in range(num_layer):
        for ind1 in range(num_qubit):
            circ.rx(params_s[ind0,ind1,0], qr[ind1])
            circ.rz(params_s[ind0,ind1,1], qr[ind1])
        for ind1 in range(len(connection_list)):
            circ.rzz(params_d[ind0,ind1], qr[connection_list[ind1,0]], qr[connection_list[ind1,1]])
    for ind1 in range(num_qubit):
        circ.rx(params_s[-1,ind1,0], qr[ind1])
        circ.rz(params_s[-1,ind1,1], qr[ind1])
    circ = qiskit.transpile(circ, Aer_simulator)
    ret = Aer_simulator.run(circ).result().data(0)["statevector"]
    return ret


def loss_function_wrapper(num_qubit, num_logical_dim, error_list, num_layer, connection_list, loss_type='L2', sample_ratio=None, seed=None):

    assert loss_type in {'L1', 'L2'}
    K = num_logical_dim
    np_rng = np.random.default_rng(seed)
    def hf_loss(params):
        code = [run_circuit(num_qubit, num_layer, params, x, connection_list) for x in range(K)]
        if sample_ratio is None:
            sub_error_list = error_list
        else:
            N0 = len(error_list)
            tmp0 = np_rng.choice(N0, min(N0, int(N0*sample_ratio)), replace=False, shuffle=False)
            sub_error_list = [error_list[x] for x in tmp0]
        loss = 0
        if loss_type=='L2':
            for e in sub_error_list:
                loss += sum([np.abs(np.vdot(code[i], e.dot(code[j])))**2 for i in range(K-1) for j in range(i+1,K)])
    
                tmp0 = np.array([np.vdot(x, e.dot(x)) for x in code])
                loss += np.sum(np.abs(tmp0 - tmp0.mean())**2)/2
        else:
            for e in sub_error_list:
                loss += sum([np.abs(np.vdot(code[i], e.dot(code[j]))) for i in range(K-1) for j in range(i+1, K)])
  
                tmp0 = np.array([np.vdot(x, e.dot(x)) for x in code])
                loss += np.sum(np.abs(tmp0 - np.mean(tmp0)))/2
        return loss
    num_parameter = 2 * num_qubit * (num_layer + 1) + num_layer * len(connection_list)
    return hf_loss, num_parameter


def loss_function_for_codestate_list(codestate_list, error_list, loss_type='L2'):

    assert loss_type in {'L1', 'L2'}
    K = len(codestate_list)
    code = codestate_list

    loss = 0
    if loss_type=='L2':
        for e in error_list:
            loss += sum([np.abs(np.vdot(code[i], e.dot(code[j])))**2 for i in range(K-1) for j in range(i+1, K)])
            tmp0 = np.array([np.vdot(x, e.dot(x)) for x in code])
            loss += np.sum(np.abs(tmp0 - tmp0.mean())**2)/2
    else:
        for e in error_list:
            loss += sum([np.abs(np.vdot(code[i], e.dot(code[j]))) for i in range(K-1) for j in range(i+1, K)])
            tmp0 = np.array([np.vdot(x, e.dot(x)) for x in code])
            loss += np.sum(np.abs(tmp0 - np.mean(tmp0)))/2
    return loss


def Pauli_tomography_density_matrix(logical_index, num_qubit, num_logical_dim, params, num_layer, connection_list):
    K = num_logical_dim
    code = [run_circuit(num_qubit, num_layer, params, x, connection_list) for x in range(K)]
    logical_state = code[logical_index]
    Pauli_weight = np.zeros(4**num_qubit)
    recon_rho = np.zeros((2**num_qubit,2**num_qubit), dtype=complex)

    for p in range(4**num_qubit):
        Pauli=np.eye(1)
        for site in range(num_qubit):
            Pauli_type = p // 4**(num_qubit - site - 1) % 4
            if Pauli_type == 1:
                Pauli = np.kron(Pauli, np.array([[0,1],[1,0]]))
            elif Pauli_type == 2:
                Pauli = np.kron(Pauli, np.array([[0,-1j],[1j,0]]))
            elif Pauli_type == 3:
                Pauli = np.kron(Pauli, np.array([[1,0],[0,-1]]))
            else:
                Pauli = np.kron(Pauli, np.eye(2))
        Pauli_weight[p]= np.vdot(logical_state, Pauli.dot(logical_state))
        recon_rho += (Pauli_weight[p] / (2 ** num_qubit)) * Pauli

    return recon_rho