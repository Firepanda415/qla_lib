## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng

import numpy
import scipy ## for cosine-sine decomposition for Quantum Shannon Decomposition
import qiskit
from utils_synth import *

ATOL = 1e-12

####----------------------------------------- Unitary matrix Synthesis -----------------------------------------####




def second_decomp(block_u1:numpy.ndarray, block_u2:numpy.ndarray, enable_debug:bool=True) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray):
    """
    Accoding to Eq. (16) in [1]
    Cosine-Sine decompsotion gives 
    U = [ A_1     ] [C    -S ] [A_2     ]
        [     B_1 ] [S     C ] [    B_2 ]
    where A_1, B_1, A_2, B_2 are unitary matrices in equal shapes
    Then, this function construct the 2nd decompsotion for each left and right block diagonal matrices
    [ U_1    ] = [V   ] [D          ] [W   ]
    [     U_2]   [   V] [   D^dagger] [   W]
    the constructon is like the following  => U_1 = VDW, U_2 = VD^dagger W
    We cancel out W terms and get U_1 U_2^dagger = V D^2 V^dagger
    Then we diagonalize U_1 U_2^dagger to obtain V (eigenvector matrix) and D^2 (eigenvalue matrix)
    Then D = sqrt(D^2), W = D V^dagger U_2 -> this is the equation in paper, but W = D^{-1} V^dagger U_1 is correct
    Note that [D          ] is a R_z multiplexer
              [   D^dagger]
    [1] Synthesis of quantum-logic circuits 10.1109/TCAD.2005.855930  or https://arxiv.org/pdf/quant-ph/0406176  (Seems like arxiv version has more details)
    """
    if block_u1.shape[0] != block_u1.shape[1] or block_u2.shape[0] != block_u2.shape[1]:
        raise ValueError('Input matrices must be square, but', block_u1.shape, block_u2.shape, 'were given')
    if block_u1.shape[0] != block_u2.shape[0]:
        raise ValueError('Input matrices must have the same size, but', block_u1.shape[0], block_u2.shape[0], 'were given')
    
    from qiskit.quantum_info.operators.predicates import is_hermitian_matrix ## 
    if is_hermitian_matrix(block_u1.dot( block_u2.T.conj() )):
        bu_evals, bu_v = scipy.linalg.eig(block_u1.dot( block_u2.T.conj() ) )
    else:
        bu_evals, bu_v = scipy.linalg.schur(block_u1.dot( block_u2.T.conj() ), output="complex" )
        bu_evals = bu_evals.diagonal()

    bu_d_inv = numpy.diag( 1/numpy.sqrt(bu_evals) )
    bu_w = bu_d_inv @ bu_v.T.conj() @ block_u1

    if enable_debug:
        bu_d = numpy.diag( numpy.sqrt(bu_evals) )
        zeroes = numpy.zeros_like(block_u1)
        prod_mat = numpy.array([[bu_v, zeroes], [zeroes, bu_v]]) @ np.array([[bu_d, zeroes], [zeroes, bu_d.conj().T]]) @ np.array([[bu_w, zeroes],[zeroes, bu_w]])
        ans = numpy.array([[block_u1, zeroes], [zeroes, block_u2]])
        print("2nd decomp error", numpy.linalg.norm(prod_mat - ans))  

    return bu_v, numpy.sqrt(bu_evals), bu_w



def synthu_qsd(unitary:numpy.ndarray, circuit:qiskit.QuantumCircuit, bottom_control:list[int], top_target:int, cz_opt:bool=True, debug:bool=False):
    """
    Use uniformly controlled rotations to synthesize a unitary matrix
    Based on Quantum Shannon Decomposition (QSD) in [1]
    
    QSD decomposition gives U = U CS Vh, 
    2nd decomposition gives U = U_V U_D U_W   and   Vh = V_V V_D V_W
    So the Circuit order: |phi> [v1h v2h] [C S] [u1 u2]
                       -> |phi> (V_W V_D V_V) CS (U_W U_D U_V)
    V_D and U_D are diagonal matrices, so we can use multiplexer R_z gates
    CS is multiplexer R_y gates
    Each V_W, V_V, U_W, U_V are unitaries, so do the decompsoition recursively
    
    As discussed in [1], the number of CNOT gates highly depends on the number l: recurively apply QSD until l-qubit operators,
    for l=1, the number of CNOT gates is 0.75 4^n  - 1.5 2^n
    for l=2, the number of CNOT gates is 9/16 4^n - 1.5 2^n
    for l=2 with optimization, the number is 23/48 4^n - 1.5 2^n + 4/3 (See Appendix A and B in [1])
       - The Appeneix A (A1) optimization is essentially multiply the last CZ gate into U2 to save a two-qubit gate
       i.e,  [u_11  u_12  0     0   ][1 0 0 0 ]     [u_11  u_12   0      0   ]
             [u_13  u_14  0     0   ][0 1 0 0 ]  =  [u_13  u_14   0      0   ]
             [0     0     u_21  u_22][0 0 1 0 ]     [0     0      u_21  -u_22]
             [0     0     u_23  u_24][0 0 0 -1]     [0     0      u_23  -u_24]
       so we just set right half of u2 be negative of itself
       This reduces (4^(n-l) - 1)/3 CNOT gates, where l is usually 2, so 1/48 4^n - 1/3 (bring down to 26/48 4^n - 1.5 2^n + 1/3)
       - The Appendix B (A2) use a custimized decomposition on the bottom level 2-qubit gates, and absorb extras to the neighbor gates
         the two-qubit operators, see https://github.com/Qiskit/qiskit/blob/97f4f6dfff4a1dd93d74a32b5fecd13382164fd3/qiskit/synthesis/unitary/qsd.py#L252
         This only has 2 CNOT, saving 1 CNOT from default 3-CNOT 2-qubit gate decomposition
         Thus, it reduces 4^(n-2) -1 CNOT gates

    the theortical lower bound is 1/4 (4^n - 3n - 1) for the exact synthesis, about 1/2 of this method
    
    [1] Synthesis of quantum-logic circuits 10.1109/TCAD.2005.855930 or https://arxiv.org/pdf/quant-ph/0406176  (Seems like arxiv version has more details)
    [2] Smaller two-qubit circuits for quantum communication and computation  10.1109/DATE.2004.1269020
    NOTE: For Multiplexer
                         0 ----|R_y|--
                       n-1 -/---ctrl-
    bottom_control is [1,2,3,...,n-1], top_target is 0
    NOTE: I don't follow the qiskit convention for the endianess
    """

    (u1,u2), thetas_cs, (v1h, v2h) = scipy.linalg.cossin(unitary, p=unitary.shape[0]//2, q=unitary.shape[1]//2, separate=True)
    thetas_cs = list(thetas_cs*2) ##  WARNING: based on Qiskit implementation on Rz, the angles are multiplied by 2 for Multiplexer
    ## Later after I realize Qiskit use QSD instead of isometry, in their code they also multiple angles by 2 for Cosine-Sine 
    ## and multiple -2 (or conj()*2) for Z rotation in U and V
    ## as shown in https://github.com/Qiskit/qiskit/blob/97f4f6dfff4a1dd93d74a32b5fecd13382164fd3/qiskit/synthesis/unitary/qsd.py#L210
    ## and https://github.com/Qiskit/qiskit/blob/97f4f6dfff4a1dd93d74a32b5fecd13382164fd3/qiskit/synthesis/unitary/qsd.py#L141C34-L141C40

    if debug:
        u1err = numpy.linalg.norm(u_v @ numpy.diag(u_dd) @ u_w - u1)
        u2err = numpy.linalg.norm(u_v @ numpy.diag(u_dd).conj() @ u_w - u2)
        if u1err > ATOL:
            raise ValueError('Invalid 2nd decomposition for U, the error is', u1err)
        if u2err > ATOL:
            raise ValueError('Invalid 2nd decomposition for U, the error is', u2err)
    
    ## Recursively synthesize the unitaries
    if len(bottom_control) == 0: ## general single qubit gate, l=1
        circuit.unitary(unitary, top_target) ## gives (3/4)*4**n - 1.5*(2**n) CNOT gates
        return
    if len(bottom_control) == 1: ## general two-qubit gate, l=2
        circuit.unitary(unitary, list(bottom_control)+[top_target]) ## gives (9/16)*4**n - 1.5*(2**n) CNOT gates without CZ optimization
        return
    
    ## v
    v_v, v_dd, v_w = second_decomp(v1h, v2h, enable_debug=debug)
    v_zangle = list(numpy.angle(v_dd)* (-2)) ## R_z(lambda) = exp(-i lambda Z/2)
    synthu_qsd(v_w, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)
    multiplexer_pauli(circuit, v_zangle, bottom_control, top_target, axis='Z') ## V_D
    synthu_qsd(v_v, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)

    # CS
    if debug:
        circuit.barrier()
    if cz_opt: ## for l=2 with optimization in Appendix A1,  bring down CNOT number to 26/48 4^n - 1.5 2^n + 1/3 (See Appendix A1 in [1])
        ## calling this function not the wrapper make the last CX gate left out
        muxry_cz(circuit, thetas_cs, bottom_control, top_target, angle_convert_flag=True) ## CS
        ##    i.e,  [u_11  u_12  0     0   ][1 0 0 0 ]     [u_11  u_12   0      0   ]
        ##          [u_13  u_14  0     0   ][0 1 0 0 ]  =  [u_13  u_14   0      0   ]
        ##          [0     0     u_21  u_22][0 0 1 0 ]     [0     0      u_21  -u_22]
        ##          [0     0     u_23  u_24][0 0 0 -1]     [0     0      u_23  -u_24]
        u2[:, len(thetas_cs)//2:] = -u2[:, len(thetas_cs)//2:] ## multiply the last CZ into U2 to save a two-qubit gate
    else:  ## l=2, the number of CNOT gates is 9/16 4^n - 1.5^n
        multiplexer_pauli(circuit, thetas_cs, bottom_control, top_target, axis='Y') ## CS
    if debug:
        circuit.barrier()

    ## u
    u_v, u_dd, u_w = second_decomp(u1, u2, enable_debug=debug)
    u_zangle = list(numpy.angle(u_dd)* (-2)) ## R_z(lambda) = exp(-i lambda Z/2)
    synthu_qsd(u_w, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)
    multiplexer_pauli(circuit, u_zangle, bottom_control, top_target, axis='Z') ## U_D
    synthu_qsd(u_v, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)

    # print(f">>>>>>>>>   Depth 1  <<<<<<<<<<<<")





## Multiplexer R_y, speicialize CX- > CZ, left the last two-qubit gate out
def muxry_cz(circuit:qiskit.QuantumCircuit, 
                     angles:list[float], controls:list[int], target:int,
                     angle_convert_flag:bool=False):
    """
    See multiplexer_rot() in ultils_synth.py for details
    With optimization in Appendix A in [1]
    The last CZ in the right-most circuit should be ignored and absorbed in to next multiplexer
    ## [1] Synthesis of quantum-logic circuits 10.1109/TCAD.2005.855930 or https://arxiv.org/pdf/quant-ph/0406176  (Seems like arxiv version has more details)
    """
    angles = numpy.array(angles)
    num_controls = len(controls)

    if len(angles) != 2**(num_controls):
        raise ValueError(f"The number of angles should be 2^{len(controls)}")
    if num_controls == 0:
        rot_helper(circuit, angles[0], target, "")
        return
    ## see Eq. (5) in [2]
    if angle_convert_flag:
        thetas = uc_angles(angles)
    else:
        thetas = angles
    ## Resursive construction
    if num_controls == 1:
        ##
        if abs(thetas[0]) > ATOL:
            circuit.ry(thetas[0], target)
        circuit.cz(controls[0], target)
        if abs(thetas[1]) > ATOL:
            circuit.ry(thetas[1], target)
    else:
        muxry_cz(circuit, thetas[:len(thetas)//2], controls[1:], target, angle_convert_flag=False)
        circuit.cz(controls[0], target)
        muxry_cz(circuit, thetas[len(thetas)//2:], controls[1:], target, angle_convert_flag=False)






####----------------------------------------- State Preparation -----------------------------------------####


def vec_mag_angles(complex_vector:numpy.ndarray):
    norm_vector = numpy.array(complex_vector)
    for i in range(len(complex_vector)):
        entry_norm = numpy.abs(complex_vector[i])
        if entry_norm > ATOL:
            norm_vector[i] = complex_vector[i]

        else:
            norm_vector[i] = 0
    return numpy.abs(complex_vector), numpy.angle(norm_vector)

def alphaz_angle(vec_omega, j,k):
    '''
    Eq. (4), for j = 1, 2, ..., 2**(n-k), k = 1,2, .., n NOTE: code index from 1
    [1] Transformation of quantum states using uniformly controlled rotations http://arxiv.org/abs/quant-ph/0407010
    '''
    angle_sum = 0
    for l in range(1, 2**(k-1)+1):
        ind1 = (2*j-1)*2**(k-1)+l
        ind2 = (2*j-2)*2**(k-1)+l
        angle_sum += vec_omega[ind1-1] - vec_omega[ind2-1]
    return angle_sum/(2**(k-1))

def alphaz_arr(vec_omega, k):
    num_qubits = int(numpy.log2(len(vec_omega)))
    res = []
    for j in range(1, 2**(num_qubits-k)+1):
        res.append( alphaz_angle(vec_omega, j,k) )
    return res


def alphay_angle(vec_amag, j, k):
    # NOTE: code index from 1
    # Eq. (8) in [1] 
    # [1] Transformation of quantum states using uniformly controlled rotations http://arxiv.org/abs/quant-ph/0407010
    tmp1_sum = 0
    tmp2_sum = 0
    for l in range(1, 2**(k-1)+1):
        ind1 = (2*j-1)*2**(k-1)+l
        tmp1_sum += vec_amag[ind1-1]**2
    for l in range(1, 2**(k)+1):
        ind2 = (j-1)*2**k+l
        tmp2_sum += vec_amag[ind2-1]**2

    if numpy.abs(tmp2_sum) < ATOL:
        return 0
    return 2* numpy.arcsin( numpy.sqrt(tmp1_sum)/numpy.sqrt(tmp2_sum) )


def alphay_arr(vec_amag, k):
    num_qubits = int(numpy.log2(len(vec_amag)))
    res = []
    for j in range(1, 2**(num_qubits-k)+1):
        res.append( alphay_angle(vec_amag, j, k) )
    return res


def aj2(vec_amag, j):
    return numpy.sqrt( vec_amag[2*j-1-1]**2 + vec_amag[2*j-1]**2 )


def stateprep_ucr(init_state:numpy.ndarray, circuit:qiskit.QuantumCircuit, debug:bool=False):
    '''
    State preparation using uniformly controlled rotations 
    Based on the algorithm in Section III [1]
    Note that Section III discuss the construction of the U such that U|a> = |0>, then Fig. 3 shows the circuit for |a> -> |0> -> |b>
    which is unnecessary for in our case
    We only need the circuit in left half of Fig. 2 (until and include single R^n_y)
    Then the state preparation circuit is just applying all operators in the inverse order
    NOTE: as shown in Eq. (7), there is a global phase remains, so we need to adjust the global phase at the beginning
    NOTE: I don't follow the qiskit convention for the endianess

    [1] Transformation of quantum states using uniformly controlled rotations http://arxiv.org/abs/quant-ph/0407010
    '''

    num_qubits = int(numpy.log2(len(init_state)))
    psi_mag, psi_angles = vec_mag_angles(init_state)

    ## Circuit
    global_phase_gate(circuit, numpy.sum(psi_angles)/(2**num_qubits), 0)

    # for j in range(1, num_qubits+1):
    #     yangles = alphay_arr(psi_mag, num_qubits-j+1)
    #     multiplexer_pauli(circuit,list(numpy.array(yangles)), list(range(j-1)), j-1, axis='Y')
    # for j in range(1, num_qubits+1):
    #     zangles = alphaz_arr(psi_angles, num_qubits-j+1)
    #     multiplexer_pauli(circuit,list(numpy.array(zangles)), list(range(j-1)), j-1, axis='Z')

    ## Put Y and Z together, easier to cancel out the CX (by transipiler)
    for j in range(1, num_qubits+1):
        ##
        yangles = alphay_arr(psi_mag, num_qubits-j+1)
        zangles = alphaz_arr(psi_angles, num_qubits-j+1)
        if debug:
            print()
            print("j", j)
            print(" - yangles" , yangles)
            print(" - zangles" , zangles)
        if debug:
            circuit.barrier()
        multiplexer_pauli(circuit,yangles, list(range(j-1)), j-1, axis='Y')
        if debug:
            circuit.barrier()
        ## If anles are all real posive, only CX gates are used and they can be cancelled out
        if numpy.linalg.norm(zangles, ord=1) > ATOL:
            multiplexer_pauli(circuit,zangles, list(range(j-1)), j-1, axis='Z')







####----------------------------------------- Tests -----------------------------------------####

if __name__ == "__main__":
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    from scipy.stats import unitary_group


    rng = numpy.random.default_rng(429096209556973234794512152190348779897183667923694427)

    test_state_prep = True #False 
    test_unitary_synth = True

    ##########################################################################################################
    print("\n\n\n\n\n")
    print("="*100)

    if test_state_prep:
        for n in range(1,8):
            for _ in range(3):
                print("\n\n\n")
                print("-"*50)
                print(f"Complex State Prep Test case: Random {n}-qubit")
                psi_real = numpy.array(rng.random(2**n) - 0.5)
                psi_imag = numpy.array(rng.random(2**n) - 0.5)
                psi = psi_real + 1j*psi_imag
                psi = psi / numpy.linalg.norm(psi, ord=2)
                ##
                print("  - State to Prepare", psi)

                ## Standard Answer from Qiskit, using isometry
                print("  \nQiskit State Preparation (isometry, column by column decomposition)")
                qiscirc = QuantumCircuit(n)
                # qiscirc.initialize(psi)
                qis_prep_isometry = qiskit.circuit.library.StatePreparation(psi)
                qiscirc.append(qis_prep_isometry, list(range(n)))

                qiscirc_trans = transpile(qiscirc, basis_gates=['rz','ry','rx', 'cx'], optimization_level=0)
                qiscirc_trans_opt = transpile(qiscirc, basis_gates=['rz','ry','rx', 'cx'], optimization_level=2)
                qis_op_dict = dict(qiscirc_trans.count_ops())
                qis_op_dict_opt = dict(qiscirc_trans_opt.count_ops()) 
                print("    - Theoretical SP upper bound (Schmidt Decomposition): ", 23/24 * (2**n)) ## https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.032318
                print("    - Theoretical SP lower bound (Schmidt Decomposition): ", 12/24 * 2**n) ## https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.032318
                print("    - Qiskit Initialize Circuit Op Count", dict(qiscirc_trans.count_ops()) )
                print("    - Qiskit Initialize Optimized Circuit Op Count", dict(qiscirc_trans_opt.count_ops()) )


                ## My uniform controlled rotation implementation
                print("  \nUCR State Preparation")
                my_ucr_circuit = QuantumCircuit(n)
                stateprep_ucr(psi, my_ucr_circuit)
                my_ucr_circuit = my_ucr_circuit.reverse_bits() ## Hi, not gonna follow qiskit rule in my implementation

                my_ucr_circuit_trans = transpile(my_ucr_circuit, basis_gates=['rz','ry','rx', 'cx'], optimization_level=0)
                my_ucr_circuit_trans_opt = transpile(my_ucr_circuit, basis_gates=['rz','ry','rx', 'cx'], optimization_level=2)
                ucr_op_dict = dict(my_ucr_circuit_trans.count_ops())
                ucr_op_dict_opt = dict(my_ucr_circuit_trans_opt.count_ops())
                print("    - Theoretical UCR lower bound: ", 2*(2**n) - 2*n) ## See https://arxiv.org/pdf/quant-ph/0406176 from  https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/circuits/3_summary_of_quantum_operations.ipynb
                print("    - UCR Direct Op Count", dict(my_ucr_circuit.count_ops()) )
                print("    - UCR Transpile Op Count", dict(my_ucr_circuit_trans.count_ops()) )
                print("    - UCR Optimized Op Count", dict(my_ucr_circuit_trans_opt.count_ops()) )
                print("    - UCR State Prep error", numpy.linalg.norm(qiskit.quantum_info.Statevector(my_ucr_circuit).data-psi) )


                
                print(f"\n>>>>UCR State Prep error = {numpy.linalg.norm(qiskit.quantum_info.Statevector(my_ucr_circuit).data-psi)}<<<<")
                print(f">>>>Qiskit State Prep error = {numpy.linalg.norm(qiskit.quantum_info.Statevector(qiscirc).data-psi)}<<<<")
                if n>1:
                    print(f">>>>Depth Summary: Qiskit={qiscirc_trans.depth()}, Qiskit_opt={qiscirc_trans_opt.depth()}, UCR={my_ucr_circuit_trans.depth()}, UC_opt={my_ucr_circuit_trans.depth()}<<<<")
                    print(f">>>>CX Summary: Qiskit={qis_op_dict['cx']}, UCR={ucr_op_dict['cx']}, UC_opt={ucr_op_dict_opt['cx']}<<<<")
                    print(f">>>>Total Gates Summary: Qiskit={numpy.sum( list(qis_op_dict.values()) )}, Qiskit_opt={numpy.sum(list(qis_op_dict_opt.values()))}, UCR={numpy.sum( list(ucr_op_dict.values()) )},UCR={numpy.sum( list(ucr_op_dict_opt.values()) )}")
                assert(numpy.linalg.norm(qiskit.quantum_info.Statevector(my_ucr_circuit).data-psi) < 1e-12)


    ##########################################################################################################
    print("\n\n\n\n\n")
    print("="*100)

    if test_unitary_synth:
        for n in range(1,8):
            for _ in range(3):
                print("\n\n\n")
                print("-"*50)
                print(f"Unitary Synthesis Test case: Random {n}-qubit")
                ## Create the state preparation U to synthesize
                U = unitary_group.rvs(2**n)
                # print("  - Unitary to Prepare\n", U)

                ## Standard Answer from Qiskit, using isometry
                print("  \nQiskit Unitary Synthesis (actually also QSD)")
                qiscirc = QuantumCircuit(n)
                qiscirc.unitary(U, list(range(n)))

                qiscirc_mat = Operator(qiscirc).data

                qiscirc_trans = transpile(qiscirc, basis_gates=['rz','ry','rx', 'cx'], optimization_level=0)
                qiscirc_trans_opt = transpile(qiscirc, basis_gates=['rz','ry','rx', 'cx'], optimization_level=2)
                print("    - Qiskit Unitary Circuit Op Count", dict(qiscirc_trans.count_ops()) )
                print("    - Qiskit Unitary Optimized Circuit Op Count", dict(qiscirc_trans_opt.count_ops()) )
                print("    - Qiskit Unitary error", numpy.linalg.norm(qiscirc_mat - U) )


                ## My QSD implementation
                print("  \nQSD Unitary Synthesis")
                my_circuit = QuantumCircuit(n)
                synthu_qsd(U, my_circuit, list(range(n))[1:], 0, cz_opt=True)
                my_circuit = my_circuit.reverse_bits() ## Hi, not gonna follow qiskit rule in my implementation

                my_circ_mat = Operator(my_circuit).data

                my_circuit_trans = transpile(my_circuit, basis_gates=['rz','ry','rx', 'cx'], optimization_level=0)
                my_circuit_trans_opt = transpile(my_circuit, basis_gates=['rz','ry','rx', 'cx'], optimization_level=2)

                print("    - Theoretical CX lower bound: ", 0.25*(4**n-3*n-1))
                print("    - QSD l=1 lower bound", (3/4)*4**n - 1.5*(2**n))
                print("    - QSD l=2 lower bound", (9/16)*4**n - 1.5*(2**n))  
                print("    - QSD l=2 A1opt lower bound", numpy.ceil((26/48)*4**n - 1.5*(2**n) + 1/3) )
                print("    - QSD l=2 A1A2opt lower bound", (23/48)*4**n - 1.5*(2**n) + 4/3)

                # print("    - QSD Direct Op Count", dict(my_circuit.count_ops()) )
                print("    - QSD Transpile Op Count", dict(my_circuit_trans.count_ops()) )
                print("    - QSD Optimized Op Count", dict(my_circuit_trans_opt.count_ops()) )
                print("    - QSD Unitary error", numpy.linalg.norm(my_circ_mat - U) )

                
                print(f"\n>>>>QSD Unitary error = {numpy.linalg.norm(my_circ_mat - U)}<<<<")
                print(f">>>>Qiskit Unitary error = {numpy.linalg.norm(qiscirc_mat - U)}<<<<")
                if n>1:
                    print(f">>>>Depth Summary: Qiskit={qiscirc_trans.depth()}, Qiskit_opt={qiscirc_trans_opt.depth()}, QAS={my_circuit_trans.depth()}, QSD_opt={my_circuit_trans_opt.depth()}<<<<")
                    print(f">>>>CX Summary: Qiskit={dict(qiscirc_trans.count_ops())['cx']}, QSD={dict(my_circuit_trans.count_ops())['cx']}, QSD_opt={dict(my_circuit_trans_opt.count_ops())['cx']}<<<<")
                    print(f">>>>Total Gates Summary: Qiskit={numpy.sum(list(dict(qiscirc_trans.count_ops()).values()))}, Qiskit_opt={numpy.sum(list(dict(qiscirc_trans_opt.count_ops()).values()))}, QSD={numpy.sum(list(dict(my_circuit_trans.count_ops()).values()))},QSD={numpy.sum(list(dict(my_circuit_trans_opt.count_ops()).values() ))}")
                assert(numpy.linalg.norm(my_circ_mat - U) < 1e-12)



