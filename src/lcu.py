## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng


import qiskit
import numpy
# import qiskit.quantum_info as qi
from oracle_synth import *
from utils_synth import *
from utils_synth import _ACCEPTED_GATES






## Use QR decomposition to make the matrix unitary
## O(n^3) cost for orthogonalization
def state_prep_qr(coeff_array: list) -> numpy.array:
    '''
    See (7.58) in https://arxiv.org/pdf/2201.08309
    LCU Oracle for PREPARE for T = sum_i=0^{K-1} a_i U_i
    V|0000...0> = 1/sqrt(||a||_1) sum_i=0^{K-1} sqrt(|a_i|)|i>
    where ||a||_1 = sum_i |a_i|, a_i>0 (WLOG by absorbing the phase into U_i)
                           sqrt(a_0)    *   *  ...  *
    V = 1/sqrt(||a||_1)    sqrt(a_1)    *   *  ...  *
                             ....        *   *  ...  *
                           sqrt(a_{K-1} *   *  ...  *
    Return
      V = Q, where Q is unitary that prepare V|0000...0> = 1/sqrt(||a||_1) sum_i=0^{K-1} sqrt(|a_i|)|i>
    '''

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



def prep_oracle(coeff_array: list, qiskit_api:bool=False) -> numpy.array:
    '''
    LCU Oracle for PREPARE for T = sum_i=0^{K-1} a_i U_i
    Synthesis unitary V such that
    V|0000...0> = 1/sqrt(||a||_1) sum_i=0^{K-1} sqrt(|a_i|)|i> 
    '''
    ## Make the length of coeff_array to be 2^n
    num_terms = len(coeff_array)
    num_qubits = nearest_num_qubit(num_terms)
    l1norm = numpy.linalg.norm(coeff_array, ord=1)
    coeff_array_normedsqrt = numpy.sqrt(numpy.abs(coeff_array)/l1norm)
    full_coeffs = [0]*(2**num_qubits)
    full_coeffs[:num_terms] = coeff_array_normedsqrt
    ##
    if qiskit_api:
        qis_prep_isometry = qiskit.circuit.library.StatePreparation(full_coeffs)
        qis_prep_isometry.name = "PREP"
        return qis_prep_isometry
    else:
        ucr_circuit = qiskit.QuantumCircuit(num_qubits)
        stateprep_ucr(full_coeffs, ucr_circuit)
        ucr_circuit = ucr_circuit.reverse_bits() ## stateprep_ucr not follow qiskit rule in my implementation
        ucr_circuit.name = "PREP"
        return ucr_circuit.to_gate()
    


def select_oracle(unitary_array: list[numpy.ndarray], qiskit_api:bool=False, debug:bool=False) -> qiskit.QuantumCircuit:
    ## See (7.55) in https://arxiv.org/pdf/2201.08309
    num_terms = len(unitary_array)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = int(numpy.log2(unitary_array[0].shape[0]))
    bin_string = '{0:0'+str(num_qubits_control)+'b}'
    ##
    select_circ = qiskit.QuantumCircuit(num_qubits_control+num_qubits_op)
    for i in range(num_terms):
        ibin = bin_string.format(i)[::-1] ## NOTE: Qiskit uses reverse order
        if qiskit_api:
            # u_gate = qiskit.transpile(qiskit.circuit.library.UnitaryGate(unitary_array[i]).definition, basis_gates=['rz', 'ry', 'rx','x', 'z','p','cx','cz'], optimization_level=1)
            # control_u = u_gate.control(num_qubits_control)
            control_u = qiskit.circuit.library.UnitaryGate(unitary_array[i]).control(num_qubits_control)
        else:
            # qsd_circuit = qiskit.QuantumCircuit(num_qubits_op)
            # qsd_circuit.append(qiskit.circuit.library.UnitaryGate(unitary_array[i]), range(num_qubits_op))
            qsd_circuit = synthu_qsd(unitary_array[i])
            qsd_circuit = qiskit.transpile(qsd_circuit, basis_gates=_ACCEPTED_GATES, optimization_level=1)
            qsd_circuit.name = "U"+str(i)
            control_u = selected_controlled_circuit(qsd_circuit, num_qubits_control, 
                           to_known_basis=False, reverse_bits=True)   ## NOTE:synthu_qsd not follow qiskit rule in my implementation
        if debug:
            my_u = synthu_qsd(unitary_array[i])
            my_u = qiskit.transpile(my_u, basis_gates=_ACCEPTED_GATES, optimization_level=0)
            qis_u = qiskit.circuit.library.UnitaryGate(unitary_array[i])
            qis_u = qiskit.transpile(qis_u.definition, basis_gates=["p", "x", "z", "rx", "ry", "rz", "cx"], optimization_level=0)
            my_transu = qiskit.transpile(my_u, basis_gates=['cx','u'], optimization_level=2)
            qis_transu = qiskit.transpile(qis_u, basis_gates=['cx','u'], optimization_level=2)
            ##
            my_cu = selected_controlled_circuit(my_u, num_qubits_control, 
                           to_known_basis=False, reverse_bits=False)
            qis_cu = qis_u.to_gate().control(num_qubits_control)
            my_transcu = qiskit.transpile(my_cu, basis_gates=['cx','u'], optimization_level=2)
            qis_transcu = qiskit.transpile(qis_cu.definition, basis_gates=['cx','u'], optimization_level=2)
            ##
            print(" --DEBUG Select Oracle--")
            print(unitary_array[i])
            print("   U Gate Count (Qis vs. Mine):", opc_str(qis_transu.count_ops()), opc_str(my_transu.count_ops()))
            print("  CU Gate Count (Qis vs. Mine):", opc_str(qis_transcu.count_ops()), opc_str(my_transcu.count_ops()))
            print()


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

def lcu_generator(coeff_array:list, unitary_array: list[numpy.ndarray], initial_state_circ=None,verbose:int=0, qiskit_api:bool=False, debug:bool=False) -> qiskit.QuantumCircuit:
    '''
    NOTE: Check example usage for big endian
    NOTE: initial_state_circ in big endian (not qiskit one anyway)
    Example usage in big endian:
            rng = numpy.random.Generator(numpy.random.PCG64(726348874394184524479665820111))
            scipy_uni = scipy.stats.unitary_group
            scipy_uni.random_state = rng
            ##
            test_coefs =  rng.random(2**n)
            test_coefs_normed = test_coefs/numpy.linalg.norm(test_coefs, ord=1)
            test_unitaries = [scipy_uni.rvs(2**n) for _ in range(2**n)]
            ##
            correct_answer = numpy.zeros(test_unitaries[0].shape, dtype=complex)
            for i in range(len(test_coefs_normed)):
                correct_answer += test_coefs_normed[i]*test_unitaries[i]
            ##
            LCU = lcu_generator(test_coefs, test_unitaries, verbose=1, qiskit_api=qiskit_api)
            circ_mat = qiskit.quantum_info.Operator(LCU).data
            lcu_sol = qiskit_normal_order_switch(circ_mat[:test_unitaries[0].shape[0],:test_unitaries[0].shape[1]]) 
            ## need the endianness switch for each coordinates on submatrix
            ## Only in this case, numpy.linalg.norm(correct_answer - lcu_sol, ord=2) gives no error
    '''
    ##
    ## Absorb the phase into the unitaries
    def vec_mag_angles(complex_vector:numpy.ndarray):
        norm_vector = numpy.array(complex_vector)
        for i in range(len(complex_vector)):
            entry_norm = numpy.abs(complex_vector[i])
            if entry_norm > 1e-12:
                norm_vector[i] = complex_vector[i]
            else:
                norm_vector[i] = 0
        return numpy.abs(complex_vector), numpy.angle(norm_vector)
    coef_abs, coef_phase = vec_mag_angles(coeff_array)
    absorbed_unitaries = [numpy.exp(1j*coef_phase[i])*unitary_array[i] for i in range(len(unitary_array))]
    ##
    num_terms = len(absorbed_unitaries)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = int(numpy.log2(absorbed_unitaries[0].shape[0]))
    if verbose > 0:
        print("  LCU-Oracle: num_qubits_control=", num_qubits_control, "num_qubits_op=", num_qubits_op)
    ##
    prep_circ = prep_oracle(coef_abs, qiskit_api=qiskit_api)
    select_circ = select_oracle(absorbed_unitaries, qiskit_api=qiskit_api, debug=debug)
    ##
    lcu_circ = qiskit.QuantumCircuit(num_qubits_control+num_qubits_op)
    if initial_state_circ:
        lcu_circ.append(initial_state_circ, list(range(num_qubits_control+num_qubits_op))[num_qubits_control:])
    ## Apply the preparation oracle
    lcu_circ.append(prep_circ, list(range(num_qubits_control)))
    ## Apply the selection oracle
    lcu_circ.append(select_circ, list(range(num_qubits_control+num_qubits_op)))
    ## Apply the preparation oracle dagger
    lcu_circ.append(prep_circ.inverse(), list(range(num_qubits_control)))
    return lcu_circ.reverse_bits(), coef_abs, absorbed_unitaries, numpy.sum(coef_abs)





## Test
if __name__ == "__main__":
    import scipy
    rng = numpy.random.Generator(numpy.random.PCG64(726348874394184524479665820111))
    scipy_uni = scipy.stats.unitary_group
    scipy_uni.random_state = rng

    print("="*50)

    fixed_nterms = 17
    optimization_level = 2
    for n in range(2,4):
        print("\n\n\n")
        print("-"*50)
        print(f"LCU {fixed_nterms} terms Test case: Random {n}-qubit unitaries")
        for _ in range(3):
            print("-"*10)
            test_coefs =  (rng.random( fixed_nterms ) - 0.5) + 1j*(rng.random( fixed_nterms ) - 0.5) ## test case for less than 2^n coefficients
            test_coefs_normed = test_coefs/numpy.linalg.norm(test_coefs, ord=1)
            print("  Normalized Coefficients:", test_coefs_normed)
            test_unitaries = [scipy_uni.rvs( 2**n ) for _ in range( fixed_nterms )]
            ## -------------------
            if len(test_coefs) != len(test_unitaries):
                raise ValueError("The number of coefficients and unitaries should be the same, but we have", len(test_coefs), "coefficients and", len(test_unitaries), "unitaries")
            ##
            correct_answer = numpy.zeros(test_unitaries[0].shape, dtype=complex)
            for i in range(len(test_coefs_normed)):
                correct_answer += test_coefs_normed[i]*test_unitaries[i]
            ## -------------------
            LCU = lcu_generator(test_coefs, test_unitaries, verbose=1, qiskit_api=False)
            LCU_qis = lcu_generator(test_coefs, test_unitaries, verbose=1, qiskit_api=True)
            circ_mat = qiskit.quantum_info.Operator(LCU).data
            lcu_sol = qiskit_normal_order_switch(circ_mat[:test_unitaries[0].shape[0],:test_unitaries[0].shape[1]]) ## need the endianness switch for each coordinates on submatrix
            
            LCU_trans = qiskit.transpile(LCU, basis_gates=['cx', 'u'], optimization_level=optimization_level)
            LCU_qis_trans = qiskit.transpile(LCU_qis, basis_gates=['cx', 'u'], optimization_level=optimization_level)
            LCU_trans_gates = dict(LCU_trans.count_ops())
            LCU_qis_trans_gates = dict(LCU_qis_trans.count_ops())
            ##
            
            # print("  Correct answer:\n", correct_answer)
            # print("\n  LCU Implementation:\n", lcu_sol)
            print(f"\n  U Gates (Qis vs. Mine) {LCU_qis_trans_gates['u']} vs. {LCU_trans_gates['u']}")
            print(f"  CX Gates (Qis vs. Mine) {LCU_qis_trans_gates['cx']} vs. {LCU_trans_gates['cx']}")
            print(f"  Depth (Qis vs. Mine) {LCU_qis_trans.depth()} vs. {LCU_trans.depth()}")
            print(f"\n  >>>>>>Error: {numpy.linalg.norm(correct_answer - lcu_sol, ord=2)}<<<<<<\n")
            assert(numpy.linalg.norm(correct_answer - lcu_sol, ord=2) < 1e-8)



    for n in range(2,5):
        print("\n\n\n")
        print("-"*50)
        print(f"LCU 2^n terms Test case: Random {n}-qubit unitaries")
        for _ in range(3):
            print("-"*10)
            test_coefs =  (rng.random( (2**n) ) - 0.5) + 1j*(rng.random( (2**n) ) - 0.5)
            test_coefs_normed = test_coefs/numpy.linalg.norm(test_coefs, ord=1)
            print("  Normalized Coefficients:", test_coefs_normed)
            test_unitaries = [scipy_uni.rvs( 2**n ) for _ in range((2**n))]
            ## -------------------
            if len(test_coefs) != len(test_unitaries):
                raise ValueError("The number of coefficients and unitaries should be the same, but we have", len(test_coefs), "coefficients and", len(test_unitaries), "unitaries")
            ##
            correct_answer = numpy.zeros(test_unitaries[0].shape, dtype=complex)
            for i in range(len(test_coefs_normed)):
                correct_answer += test_coefs_normed[i]*test_unitaries[i]
            ## -------------------
            LCU = lcu_generator(test_coefs, test_unitaries, verbose=1, qiskit_api=False)
            LCU_qis = lcu_generator(test_coefs, test_unitaries, verbose=1, qiskit_api=True)
            circ_mat = qiskit.quantum_info.Operator(LCU).data
            lcu_sol = qiskit_normal_order_switch(circ_mat[:test_unitaries[0].shape[0],:test_unitaries[0].shape[1]]) ## need the endianness switch for each coordinates on submatrix
            
            LCU_trans = qiskit.transpile(LCU, basis_gates=['cx', 'u'], optimization_level=optimization_level)
            LCU_qis_trans = qiskit.transpile(LCU_qis, basis_gates=['cx', 'u'], optimization_level=optimization_level)
            LCU_trans_gates = dict(LCU_trans.count_ops())
            LCU_qis_trans_gates = dict(LCU_qis_trans.count_ops())
            ##
            
            # print("  Correct answer:\n", correct_answer)
            # print("\n  LCU Implementation:\n", lcu_sol)
            print(f"\n  U Gates (Qis vs. Mine) {LCU_qis_trans_gates['u']} vs. {LCU_trans_gates['u']}")
            print(f"  CX Gates (Qis vs. Mine) {LCU_qis_trans_gates['cx']} vs. {LCU_trans_gates['cx']}")
            print(f"  Depth (Qis vs. Mine) {LCU_qis_trans.depth()} vs. {LCU_trans.depth()}")
            print(f"\n  >>>>>>Error: {numpy.linalg.norm(correct_answer - lcu_sol, ord=2)}<<<<<<\n")
            assert(numpy.linalg.norm(correct_answer - lcu_sol, ord=2) < 1e-8)







