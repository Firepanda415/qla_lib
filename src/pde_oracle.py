## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng

import numpy
import scipy 
import qiskit
from utils_synth import *




###----------------------------------------- Spartially Discretized Hamiltonian Simulation -----------------------------------------####



def ham_wj(circuit:qiskit.QuantumCircuit, gammatau:float, lam:float, start_qubit:int, end_qubit:int):
    '''
    Circuit for matrix W_j, see Fig. 1 in [1], Eq. (38), (37), (35)
    end_qubit is j here and j is from 1 to n in the paper [1]
    Note that P(-lam) @ H() @ Z() @ H() @ P(lam) = np.exp(1j*lam)*|0><1|+np.exp(-1j*lam)*|1><0| = w_1, (as indexed in the paper)

    [1]  Hamiltonian simulation for hyperbolic partial differential equations by scalable quantum circuits https://arxiv.org/pdf/2402.18398
    '''
    if start_qubit > end_qubit:
        raise ValueError("The start qubit should be less than the end qubit, but", start_qubit, end_qubit, "were given")
    if start_qubit == end_qubit:
        circuit.p(lam, end_qubit) ## P(-lam) dagger is just P(lam)
        circuit.h(end_qubit)
        circuit.rz(2*gammatau, end_qubit)
        circuit.h(end_qubit)
        circuit.p(-lam, end_qubit)
    else:
        control_list = list(range(start_qubit, end_qubit)) ## control qubits
        ## U_j (-lam)
        for i in control_list: ## end_qubit is control
            circuit.cx(end_qubit, i)
        circuit.p(lam, end_qubit) ## P(-lam) dagger is just P(lam)
        circuit.h(end_qubit)
        ## CZ
        mc_rot(circuit, 'Z', 2*gammatau, control_list, end_qubit)
        ## U_j (lam)^dagger
        circuit.h(end_qubit)
        circuit.p(-lam, end_qubit)
        for i in control_list[::-1]: ## end_qubit is control
            circuit.cx(end_qubit, i)


def expmiht_approx(n_terms:int, time_tau:float, gamma:float, lam:int, order:int=2, verbose:int=0):
    '''
    Hamiltonian simulation circuit in [1]. See Fig. 1, Eq. (39), (33)
    NOTE: the paper use little endian, as stated in the paragraph below (36)

    Hermitian matrix H = gamma * sum_{j=1}^{n} (e^{i*lam} s^-_j + e^{-i*lam} s^+_j)
    gamma in R is a scale parameter, lam in R is a phase parameter
    s^+_j = I^{⊗(n-j)} ⊗ sigma_01 ⊗ sigma_10^{⊗(j-1)}
    s^-_j = I^{⊗(n-j)} ⊗ sigma_10 ⊗ sigma_01^{⊗(j-1)}
    sigma_01 = |0><1|, sigma_10 = |1><0|

    [1]  Hamiltonian simulation for hyperbolic partial differential equations by scalable quantum circuits https://arxiv.org/pdf/2402.18398
    '''

    # num_qubits = int(numpy.log2(ham_mat.shape[0]))
    num_qubits = n_terms
    ##
    full_circuit = qiskit.QuantumCircuit(num_qubits)
    if order == 1:
        for i in range(num_qubits-1):
            ham_wj(full_circuit, gamma*time_tau, lam, 0, i )
    elif order == 2:
        ham_wj(full_circuit, -0.5*gamma*time_tau, lam, 0, 0 )
        for i in range(num_qubits-1):
            ham_wj(full_circuit, gamma*time_tau, lam, 0, i )
        ham_wj(full_circuit, 0.5*gamma*time_tau, lam, 0, 0 )

    ##
    if verbose > 0:
        if order == 1:
            error_upperbound = 0.5*(gamma**2)*(time_tau**2)*(num_qubits-1)
        elif order == 2:
            error_upperbound = (1.0/6.0)*(gamma**3)*(time_tau**3)*(2*num_qubits-3)
        print("  Ham. Sim. Error Upper Bound=", error_upperbound)
    return full_circuit








####----------------------------------------- Tests -----------------------------------------####

if __name__ == "__main__":
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    from scipy.stats import unitary_group


    test_pdeham = False
    ##########################################################################################################
    if test_pdeham:

        ## for Hamilon construction

        def tensor_power(mat:numpy.ndarray, power:int):
            if power == 0:
                return 1
            if power == 1:
                return mat
            return numpy.kron(mat, tensor_power(mat, power-1))

        def hams_sju(sign:str, j:int, n:int): ## little endian
            KET0 = numpy.array([[1], [0]])
            KET1 = numpy.array([[0], [1]])
            K10B = KET1 @ KET0.T
            K01B = KET0 @ KET1.T
            I = numpy.eye(2)
            Inj = tensor_power(I, n-j-1)
            if sign == '+':
                tmp1 = tensor_power(K01B, j)
                tmp = numpy.kron(K10B, tmp1)
            elif sign == '-':
                tmp1 = tensor_power(K10B, j)
                tmp = numpy.kron(K01B, tmp1)

            return numpy.kron(Inj, tmp)

        def hams_fullmat(gamma:float, lam:float, n:int):
            ham_mat = numpy.zeros((2**n,2**n), dtype=complex)
            for j in range(n):
                ham_mat += numpy.exp(1j*lam)*hams_sju('-', j, n) + numpy.exp(-1j*lam)*hams_sju('+', j, n)
            ham_mat *= gamma
            return ham_mat
        
        lam = 0.5
        gamma = 0.1
        n = 2 ## 2^n is number of spatial discretization points
        time_tau = 1
        ham = hams_fullmat(gamma, lam, n)

        scipy_sol = scipy.linalg.expm(-1j * time_tau * ham)
        my_sol_circ = expmiht_approx(n, time_tau, gamma, lam, order=2, verbose=1)
        my_sol = Operator(my_sol_circ).data

        error = numpy.linalg.norm( scipy_sol - Operator(my_sol).data )
        print("  >>Hamiltonian Evolution Error", error)
