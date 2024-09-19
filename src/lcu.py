## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng


import qiskit
import numpy
# import scipy



def nearest_num_qubit(x):
    return int(numpy.ceil(numpy.log2(x)))


def qiskit_to_normal_order(qiskit_matrix):
    num_qubits = int(numpy.log2(qiskit_matrix.shape[0]))
    bin_str = '{0:0'+str(num_qubits)+'b}'
    new_matrix = numpy.zeros(qiskit_matrix.shape, dtype=complex)
    for i in range(qiskit_matrix.shape[0]):
        for j in range(qiskit_matrix.shape[1]):
            normal_i = int(bin_str.format(i)[::-1],2)
            normal_j = int(bin_str.format(j)[::-1],2)
            new_matrix[normal_i,normal_j] = qiskit_matrix[i,j]
    return new_matrix

## Use QR decomposition to make the matrix unitary
## O(n^3) cost for orthogonalization
def state_prep_qr(coeff_array: list) -> numpy.array:
    ## See (7.58) in https://arxiv.org/pdf/2201.08309
    ## LCU Oracle for PREPARE for T = \sum_i=0^{K-1} a_i U_i
    ## V|0000...0> = 1/\sqrt(||a||_1) \sum_i=0^{K-1} \sqrt(|a_i|)|i>
    ## where ||a||_1 = \sum_i |a_i|, a_i>0 (WLOG by absorbing the phase into U_i)
    ##                        \sqrt(a_0)    *   *  ...  *
    ## V = 1/\sqrt(||a||_1)   \sqrt(a_1)    *   *  ...  *
    ##                          ....        *   *  ...  *
    ##                        \sqrt(a_{K-1} *   *  ...  *
    ## Return
    ##   V = Q, where Q is unitary that prepare V|0000...0> = 1/\sqrt(||a||_1) \sum_i=0^{K-1} \sqrt(|a_i|)|i>
    for c in coeff_array:
        if c < 0:
            raise ValueError("All coefficients should be positive, but we have", c)

    l1norm = numpy.linalg.norm(coeff_array, ord=1)
    coeff_array_normedsqrt = numpy.sqrt(numpy.abs(coeff_array)/l1norm)
    num_terms = len(coeff_array)
    num_qubits = nearest_num_qubit(num_terms)
    v = numpy.identity(2**num_qubits, dtype=float)
    for row in range(num_terms):
        v[row,0] =  coeff_array_normedsqrt[row]
    ## orthgonoalize v to make it unitary
    q,r = numpy.linalg.qr(v, mode='complete')
    if abs(q[0,0] -coeff_array_normedsqrt[0]) > 1e-12:
        q = -q
    if numpy.linalg.norm(q[:,0] - v[:,0]) > 1e-12:
        raise ValueError("QR decomposition failed to make V a unitary matrix", "we get", q[:,0], "but suppose to get", v[:,0])
    return q



def prep_oracle(coeff_array: list, method='qr') -> numpy.array:
    if method.lower() == 'qr':
        return state_prep_qr(coeff_array)
    else:
        raise ValueError("Unknown method", method, "for preparing the oracle")


def select_oracle(unitary_array: list[numpy.ndarray]) -> qiskit.QuantumCircuit:
    ## See (7.55) in https://arxiv.org/pdf/2201.08309
    num_terms = len(unitary_array)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = int(numpy.log2(unitary_array[0].shape[0]))
    bin_string = '{0:0'+str(num_qubits_control)+'b}'
    ##
    select_circ = qiskit.QuantumCircuit(num_qubits_control+num_qubits_op)
    for i in range(num_terms):
        ibin = bin_string.format(i)[::-1] ## NOTE: Qiskit uses reverse order
        control_u = qiskit.quantum_info.Operator(unitary_array[i]).to_instruction().control(num_qubits_control)
        ## For 0-control
        for q in range(len(ibin)):
            qbit = ibin[q]
            if qbit == '0':
                select_circ.x(q)
        ## Apply the controlled-U gate
        select_circ.append( control_u, list(range(num_qubits_control+num_qubits_op)) )
        ## UNDO the X gate for 0-control
        for q in range(len(ibin)):
            qbit = ibin[q]
            if qbit == '0':
                select_circ.x(q)
    ##
    select_circ.name = 'SELECT'
    return select_circ

def lcu_generator(coeff_array:list, unitary_array: list[numpy.ndarray]) -> qiskit.QuantumCircuit:
    prep_mat = prep_oracle(coeff_array)
    select_circ = select_oracle(unitary_array)
    num_terms = len(unitary_array)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = int(numpy.log2(unitary_array[0].shape[0]))
    print("  LCHS-Oracle: num_qubits_control=", num_qubits_control, "num_qubits_op=", num_qubits_op)
    ##
    lcu_circ = qiskit.QuantumCircuit(num_qubits_control+num_qubits_op)
    ## Apply the preparation oracle
    lcu_circ.unitary(prep_mat, list(range(num_qubits_control)), label='PREP')
    ## Apply the selection oracle
    lcu_circ.append(select_circ, list(range(num_qubits_control+num_qubits_op)))
    ## Apply the preparation oracle dagger
    lcu_circ.unitary(prep_mat.conj().T, list(range(num_qubits_control)), label='PREP_DAG')
    ##
    lcu_circ.reverse_bits() ## stupid qiskit measurement
    return lcu_circ








## Test
if __name__ == "__main__":
    test_coefs = numpy.array([1,2,3,4])
    test_coefs_normed = test_coefs/numpy.linalg.norm(test_coefs, ord=1)
    test_unitaries = [numpy.array([[1,0],[0,1]]), numpy.array([[0,1],[1,0]]), numpy.array([[1,0],[0,-1]]), numpy.array([[0,1j],[1j,0]])]

    # test_coefs = numpy.array([1,2])
    # test_coefs_normed = test_coefs/numpy.linalg.norm(test_coefs, ord=1)
    # test_unitaries = [numpy.array([[1,0],[0,1]]), numpy.array([[0,1],[1,0]])]

    ## -------------------
    if len(test_coefs) != len(test_unitaries):
        raise ValueError("The number of coefficients and unitaries should be the same, but we have", len(test_coefs), "coefficients and", len(test_unitaries), "unitaries")
    ##
    correct_answer = numpy.zeros(test_unitaries[0].shape, dtype=complex)
    for i in range(len(test_coefs_normed)):
        correct_answer += test_coefs_normed[i]*test_unitaries[i]

    ## -------------------
    PREP = prep_oracle(test_coefs)
    SELE = select_oracle(test_unitaries)
    LCU = lcu_generator(test_coefs, test_unitaries)
    circ_op = qiskit_to_normal_order(qiskit.quantum_info.Operator(LCU).data)
    lcu_sol = circ_op[:test_unitaries[0].shape[0],:test_unitaries[0].shape[1]]
    ##
    print("Correct answer:", correct_answer)
    print("\nLCU Implementation:", lcu_sol)
    print("\nError:", numpy.linalg.norm(correct_answer - lcu_sol, ord=2))






