## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng



import numpy
import qiskit

## TODO: fix ccccx()
## TODO: verify the implementation in  mc_x()


# # I: 00 -> 0
# # X: 01 -> 1
# # Z: 10 -> 2
# # Y: 11 -> 3
# PAULI_I2C = ['I', 'X', 'Z', 'Y']

# #   I  X  Z  Y
# # I 1  1  1  1
# # X 1  1 -i  i
# # Z 1  i  1 -i
# # Y 1 -i  i  1
# PHASE_TABLE = {
#     (0, 0): 1,
#     (0, 1): 1,
#     (0, 2): 1,
#     (0, 3): 1,
    
#     (1, 0): 1,
#     (1, 1): 1,
#     (1, 2): -1j,
#     (1, 3): 1j,
    
#     (2, 0): 1,
#     (2, 1): 1j,
#     (2, 2): 1,
#     (2, 3): -1j,
    
#     (3, 0): 1,
#     (3, 1): -1j,
#     (3, 2): 1j,
#     (3, 3): 1
# }

# def pauli_binary(p:str) -> int:
#     if p == 'I':
#         return 0
#     elif p == 'X':
#         return 1
#     elif p == 'Z':
#         return 2
#     elif p == 'Y':
#         return 3
#     else:
#         raise ValueError(f"Invalid Pauli operator: {p}")

# def pauli_mult(p1:str, p2:str):
#     """
#     Multiply two Pauli operators represented by their binary codes.

#     Parameters:
#     p1 (str): Pauli operator 1, X,Y,Z or I
#     p2 (str): Pauli operator 2, X,Y,Z or I

#     Returns:
#     tuple: (phase factor, resulting Pauli binary code)
#     """
#     p1_binary=pauli_binary[p1.upper()]
#     p2_binary=pauli_binary[p2.upper()]
#     phase = PHASE_TABLE(p1_binary, p2_binary)
#     if phase is None:
#         raise ValueError(f"Invalid Pauli binary codes: {p1_binary}, {p2_binary}")
    
#     # Determine the resulting Pauli operator
#     # I * any = any
#     if p1_binary == 0:
#         result_binary = p2_binary
#     elif p2_binary == 0:
#         result_binary = p1_binary
#     elif p1_binary == p2_binary:
#         result_binary = 0  # X*X=I, Y*Y=I, Z*Z=I
#     else:
#         # For X*Y= iZ, Y*Z= iX, Z*X= iY and their inverses
#         # The resulting Pauli is determined by XORing the binary codes
#         result_binary = p1_binary ^ p2_binary
#     return (phase, PAULI_I2C[result_binary])


def nearest_num_qubit(x):
    return int(numpy.ceil(numpy.log2(x)))


def qiskit_normal_order_switch(qiskit_matrix):
    ## qiskit matrix to normal matrix or verse versa
    num_qubits = int(numpy.log2(qiskit_matrix.shape[0]))
    bin_str = '{0:0'+str(num_qubits)+'b}'
    new_matrix = numpy.zeros(qiskit_matrix.shape, dtype=complex)
    for i in range(qiskit_matrix.shape[0]):
        for j in range(qiskit_matrix.shape[1]):
            normal_i = int(bin_str.format(i)[::-1],2)
            normal_j = int(bin_str.format(j)[::-1],2)
            new_matrix[normal_i,normal_j] = qiskit_matrix[i,j]
    return new_matrix

def qiskit_normal_order_switch_vec(qiskit_vector):
    ## qiskit matrix to normal matrix or verse versa
    num_qubits = int(numpy.log2(len(qiskit_vector)))
    bin_str = '{0:0'+str(num_qubits)+'b}'
    new_vector = numpy.zeros(qiskit_vector.shape, dtype=complex)
    for i in range(len(qiskit_vector)):
        normal_i = int(bin_str.format(i)[::-1],2)
        new_vector[normal_i] = qiskit_vector[i]
    return new_vector




#-------------------------------- Linear Algebra --------------------------------#
def mat_egh2su(mat):
    """
    Compute the global phase TO MAKE A MATRIX HAS DETERMINANT 1.
    We want to find an alpha such that 
        1 = det(alpha * mat) = alpha^n * det(mat) where n is the dimension of the matrix.
    
    Parameters:
    mat (numpy.ndarray): A square matrix.
    
    Returns:
    complex: The global phase of the matrix.
    """
    # mat_det = detm(mat.tolist())
    mat_det = numpy.linalg.det(mat)
    if type(mat) == numpy.ndarray:
        mat_dim = mat.shape[0]
    else:
        mat_dim = len(mat)
    return mat_det ** (-1/mat_dim)

def mat_normdet1(mat: numpy.ndarray) -> numpy.ndarray:
    """
    Normalize the matrix to have determinant 1.
    
    Parameters:
    mat (numpy.ndarray): A square matrix.
    
    Returns:
    numpy.ndarray: The normalized matrix.
    """
    expphase = mat_egh2su(mat)
    return numpy.array(mat) * expphase



## From https://github.com/mpham26uchicago/laughing-umbrella/blob/main/background/Full%20Two%20Qubit%20KAK%20Implementation.ipynb
def decompose_one_qubit_product(Umat: numpy.ndarray):
    """
    Decompose a 4x4 unitary matrix to two 2x2 unitary matrices.
    Args:
        U (numpy.ndarray): input 4x4 unitary matrix to decompose.
    Returns:
        exp_phase (float): exp(1j*global phase).
        U1 (numpy.ndarray): decomposed unitary matrix U1.
        U2 (numpy.ndarray): decomposed unitary matrix U2.
    """
    i, j = numpy.unravel_index(numpy.argmax(Umat, axis=None), Umat.shape)

    def u1_set(i):
        return (1, 3) if i % 2 else (0, 2)
    def u2_set(i):
        return (0, 1) if i < 2 else (2, 3)

    u1 = Umat[numpy.ix_(u1_set(i), u1_set(j))]
    u2 = Umat[numpy.ix_(u2_set(i), u2_set(j))]

    u1 = mat_normdet1(u1)
    u2 = mat_normdet1(u2)

    exp_phase = Umat[i, j] / (u1[i // 2, j // 2] * u2[i % 2, j % 2])

    return exp_phase, u1, u2

#--------------------------------   




#-------------------------------- Append the controlled version of a qiskit.QuantumCircuit --------------------------------#

def selected_controlled_circuit(son:qiskit.QuantumCircuit, num_controls:int, to_known_basis:bool=True, reverse_bits:bool=False):
    '''
    TODO: implement controlled H so CZ is accepted as CZ_01 =  H_1 CX_01 H_1
    Notice how qiskit add (controlld) global phase in [1]
    [1] https://github.com/Qiskit/qiskit/blob/90e92a46643c72a21c5852299243213907453c21/qiskit/circuit/add_control.py#L215
    NOTE: with reverse_bits=True, make sure the input circuit is transpiled in the accpeted operations or set to_known_basis=True
    '''
    controlled_qubits = list(range(num_controls))
    accepted_ops = ['rz', 'ry', 'rx','x','p', 'cx']
    if to_known_basis:
        dispclined_son = qiskit.transpile(son, basis_gates=accepted_ops, optimization_level=1)
    else:
        dispclined_son = son
    ##
    total_qubits = son.num_qubits+num_controls
    max_target = son.num_qubits-1
    mother = qiskit.QuantumCircuit(total_qubits)
    for instrct in dispclined_son.data:
        # print("  -DEBUG Instruction:", instrct)
        operator = instrct.operation
        action_qubits = instrct.qubits
        if len(action_qubits) == 1:
            if reverse_bits:
                target = (max_target-action_qubits[0]._index)+num_controls
            else:
                target = action_qubits[0]._index+num_controls
            params = operator.params ## rotation gates
        elif len(action_qubits) == 2: ## only CX currently
            if reverse_bits:
                control = (max_target-action_qubits[0]._index)+num_controls
                target = (max_target-action_qubits[1]._index)+num_controls ## 1st is control, 2nd is target
            else:
                control = action_qubits[0]._index+num_controls
                target = action_qubits[1]._index+num_controls ## 1st is control, 2nd is target
        else:
            raise ValueError(f"Invalid number of qubits in the instruction, get {len(action_qubits)}")

        if operator.name == 'rz':
            mc_rot(mother, 'Z', params[0], controlled_qubits, target)
        elif operator.name == 'ry':
            mc_rot(mother, 'Y', params[0], controlled_qubits, target)
        elif operator.name == 'rx':
            mc_rot(mother, 'X', params[0], controlled_qubits, target)
        elif operator.name == 'cx':
            mc_x(mother, controlled_qubits+[control], target)
        elif operator.name == 'p':
            mc_p(mother, params[0], controlled_qubits, target)
        elif operator.name == 'x':
            mc_x(mother, controlled_qubits, target)
        else:
            raise ValueError(f"Invalid operation {operator.name}")
    # add global phase
    mc_p(mother, son.global_phase, controlled_qubits[:-1], controlled_qubits[-1])
    return mother



    










## ------------------- Different Gates ------------------- ##


def global_phase_gate(circuit:qiskit.QuantumCircuit, phase:float, target:int, no_gate:bool=False):
    """
    Apply a global phase gate to a quantum circuit (phi in exp(i phi))
    Ph(theta) = [[ exp(i theta)   0           ] = P(theta)XP(theta)X
                 [ 0              exp(i theta)]]
    NOTE  X = [[0 1]  and P(theta) = [[1 0           ]    in Qiskit definition 
               [1 0]]                 [0 exp(i theta)]]
    NOTE: usually you don't want a physical gate on this, just add the new global phase to the existing global phase
    """
    if no_gate:
        circuit.global_phase += phase ## do not create phase yet
    else:
        circuit.x(target)
        circuit.p(phase, target)
        circuit.x(target)
        circuit.p(phase, target)


#-------------------------------- Code for multiplexer_rot --------------------------------#

## Binary operations
def binary_reflected_gray_code(m:int) -> int:
    """
    Generate binary reflected Gray code for an integer m.
    For methods related to Quantum Shannon Decomposition
    """
    return m ^ (m >> 1)

##
def binary_inner_product(a: int, b: int) -> int:
    """
    Compute the dot-wise inner product of the binary representations
    of two decimal integers. a and b are decimal integers.
    
    For methods related to Quantum Shannon Decomposition
    """
    # Perform bitwise AND
    and_result = a & b
    # Count the number of set bits (1s) in the result
    count = 0
    while and_result:
        count += and_result & 1
        and_result >>= 1
    
    return count





## Helper functions for multiplexer_rot
def rot_helper(circuit:qiskit.QuantumCircuit, 
               angle:float, target_qubit:int, axis:str):
    """
    Helper function for apply 1-qubit rotation gate
    """
    if abs(angle) <1e-12:
        return
    if axis.capitalize() == 'X':
        circuit.rx(angle, target_qubit)
    elif axis.capitalize() == 'Y':
        circuit.ry(angle, target_qubit)
    elif axis.capitalize() == 'Z':
        circuit.rz(angle, target_qubit)

## Helper functions for multiplexer_rot
def rot_cx_helper(circuit:qiskit.QuantumCircuit,control_qubit:int, target_qubit:int, axis:str):
    """
    For R_x gate, need extra R_y rotation to change the basis
    See https://github.com/Qiskit/qiskit/blob/cb486e6a312dccfcbb4d88e8f21d93455d1ddf82/qiskit/circuit/library/generalized_gates/uc_pauli_rot.py#L121
    """
    if axis.capitalize() == 'X':
        circuit.ry(numpy.pi/2, target_qubit)
    circuit.cx(control_qubit, target_qubit)
    if axis.capitalize() == 'X':
        circuit.ry(-numpy.pi/2, target_qubit)

## angle computation for uniformly controlled rotation gates
def uc_angles(angles:list[float]) -> list[float]:
    """
    Given a list of angles, return the angles for uniformly controlled rotation gates
    See Eq. (5) in [1] and the following discussion of the inverse of the coeeficient matrix
    [1] Transformation of quantum states using uniformly controlled rotations 10.1103/PhysRevLett.93.130502  (Published version is preferred)
    """
    ## since we only care the inverse of the coefficient matrix, compute the transpose directly
    coeff_mat_T = numpy.zeros((len(angles), len(angles)), dtype=int) ## transpose of the coefficient matrix
    for i in range(len(angles)):
        for j in range(len(angles)):
            gj = binary_reflected_gray_code(j) ## g_{j-1} in Eq. (5) in [1], the index in paper starts from 1
            coeff_mat_T[j,i] = (-1)**binary_inner_product(i, gj)  ## fill the transpose, not the matrix itself
    return coeff_mat_T.dot( angles/len(angles) ) ## 2^-k is len(angles)

## Multiplexer for rotation gates
def multiplexer_rot(circuit:qiskit.QuantumCircuit, 
                     angles:list[float], controls:list[int], target:int, axis:str,
                     angle_convert_flag:bool=False):
    """
    Multiplexor gate for rotation gates
    For 2-qubit multiplexor, see Theorem 4 in [2]
    For 2+-qubit multiplexor, see Fig 2 in [1] and Theorem 8 in [2]
    For angles, see Eq. (5) in [1]
    Should be the same as the implementation in Qiskit [3]
    NOTE: THE LAST CNOT IS LEFT OUT, THE CORRECT ONE IS CALLED IN THE WRAPPER FUNCTION multiplexer_pauli()
    [1] Quantum Circuits for General Multiqubit Gates 10.1103/PhysRevLett.93.130502  (Published version is preferred)
    [2] Synthesis of Quantum Logic Circuits https://arxiv.org/abs/quant-ph/0406176
    [3] https://github.com/Qiskit/qiskit/blob/cb486e6a312dccfcbb4d88e8f21d93455d1ddf82/qiskit/circuit/library/generalized_gates/uc_pauli_rot.py#L32-L164

    Note that, the circuit in [1] is equivalent to the circuit in Fig. 2 in [2] that has less number of CX gates
    I prefer [1] for the implementation since the angle calculation is more clear
    """
    angles = numpy.array(angles)
    num_controls = len(controls)

    if len(angles) != 2**(num_controls):
        raise ValueError(f"The number of angles should be 2^{len(controls)}")
    if num_controls == 0:
        rot_helper(circuit, angles[0], target, axis)
        return
    ## see Eq. (5) in [2]
    if angle_convert_flag:
        thetas = uc_angles(angles)
    else:
        thetas = angles
    ## Resursive construction
    if num_controls == 1:
        ##
        rot_helper(circuit, thetas[0], target, axis)
        rot_cx_helper(circuit, controls[0], target, axis)
        rot_helper(circuit, thetas[1], target, axis)
    else:
        multiplexer_rot(circuit, thetas[:len(thetas)//2], controls[1:], target, axis, angle_convert_flag=False)
        rot_cx_helper(circuit, controls[0], target, axis)
        multiplexer_rot(circuit, thetas[len(thetas)//2:], controls[1:], target, axis, angle_convert_flag=False)



## wrapper for multiplexer_rot
def multiplexer_pauli(circuit:qiskit.QuantumCircuit, 
                     angles:list[float], controls:list[int], target:int, axis:str):
    """
    Multiplexor gate for Pauli rotations
    """
    multiplexer_rot(circuit, angles.copy(), controls, target, axis, angle_convert_flag=True)
    if len(controls) > 0:
        rot_cx_helper(circuit, controls[0], target, axis)




#-------------------------------- Code for Multiple-controlled RX, RY, RZ  --------------------------------#

def mc_p(circuit:qiskit.QuantumCircuit, angle:float, controls:list[int], target:int):
    """
    Multiple-controlled phase gate, qiskit says it uses multiple-controlled SU(2) implementation in [1]
    
    TODO: verifiy the implementation in [2]
    TODO: implement the better controlled version of P gate
    [1] Decomposition of Multi-controlled Special Unitary Single-Qubit Gates http://arxiv.org/abs/2302.06377
    [2] https://github.com/Qiskit/qiskit/blob/f5c005c773f5125325cd38ed3b62014f98479d51/qiskit/circuit/library/standard_gates/p.py#L362
    """
    if numpy.abs(angle) < 1e-12:
        return
    if len(controls) == 0:
        circuit.p(angle, target)
    elif len(controls) == 1:
        circuit.cp(angle, controls[0], target)
    else:
        mcp_circ = qiskit.QuantumCircuit(len(controls)+1)
        itr_controls = list(range(len(controls)))
        itr_target = mcp_circ.num_qubits-1
        for i in range(len(itr_controls)):
            mc_rot(mcp_circ, 'Z', angle/(2**i), itr_controls, itr_target)
            itr_target = itr_controls.pop()
        mcp_circ.p(angle/(2**len(controls)), itr_target)
        circuit.append(mcp_circ, controls + [target])



def ccx(circuit:qiskit.QuantumCircuit, controls:list[int], target:int):
    '''
    Special case for 2-conntrolled X gate (Toffoli gate)
    [1] Quantum computation and quantum information by Nielsen and Chuang 978-1-107-00217-3
    [2] On the CNOT-cost of TOFFOLI gates https://arxiv.org/pdf/0803.2316
    '''
    if len(controls) != 2:
        raise ValueError(f"The number of controls should be 2 but {len(controls)} is given")
    
    circuit.h(target)
    circuit.cx(controls[1], target)
    circuit.tdg(target)
    circuit.cx(controls[0], target)
    circuit.t(target)
    circuit.cx(controls[1], target)
    circuit.tdg(target)
    circuit.cx(controls[0], target)

    circuit.t(controls[1])
    circuit.t(target)
    circuit.h(target)
    circuit.cx(controls[0], controls[1])
    circuit.t(controls[0])
    circuit.tdg(controls[1])
    circuit.cx(controls[0], controls[1])
    

def cccx(circuit:qiskit.QuantumCircuit, controls:list[int], target:int):
    '''
    Special case for 3-conntrolled X gate
    https://github.com/Qiskit/qiskit/blob/90e92a46643c72a21c5852299243213907453c21/qiskit/synthesis/multi_controlled/mcx_synthesis.py#L300
    '''
    if len(controls) != 3:
        raise ValueError(f"The number of controls should be 3 but {len(controls)} is given")
    circuit.h(target)
    for qu in [controls[0], controls[1], controls[2], target]:
        circuit.p(numpy.pi / 8, qu)
    circuit.cx(controls[0], controls[1])
    circuit.p(-numpy.pi / 8, controls[1])
    circuit.cx(controls[0], controls[1])
    circuit.cx(controls[1], controls[2])
    circuit.p(-numpy.pi / 8, controls[2])
    circuit.cx(controls[0], controls[2])
    circuit.p(numpy.pi / 8, controls[2])
    circuit.cx(controls[1], controls[2])
    circuit.p(-numpy.pi / 8, controls[2])
    circuit.cx(controls[0], controls[2])
    circuit.cx(controls[2], target)
    circuit.p(-numpy.pi / 8, target)
    circuit.cx(controls[1], target)
    circuit.p(numpy.pi / 8, target)
    circuit.cx(controls[2], target)
    circuit.p(-numpy.pi / 8, target)
    circuit.cx(controls[0], target)
    circuit.p(numpy.pi / 8, target)
    circuit.cx(controls[2], target)
    circuit.p(-numpy.pi / 8, target)
    circuit.cx(controls[1], target)
    circuit.p(numpy.pi / 8, target)
    circuit.cx(controls[2], target)
    circuit.p(-numpy.pi / 8, target)
    circuit.cx(controls[0], target)
    circuit.h(target)

def ccccx(circuit:qiskit.QuantumCircuit, controls:list[int], target:int):
    '''
    Special case for 4-conntrolled X gate   
    TODO: implement the better version
    '''
    if len(controls) != 4:
        raise ValueError(f"The number of controls should be 4 but {len(controls)} is given")
    from qiskit.circuit.library.standard_gates import C4XGate
    place_holder = C4XGate().definition
    circuit.append(place_holder, controls + [target])



def mc_x(circuit:qiskit.QuantumCircuit, controls:list[int], target:int):
    """
    Multiple-controlled X, see Fig 7 in [1]

    As mentioned in [2], the qiskit actually implement multiple-controlled X gate as multiple-controlled P gate, 
    with H gate before and after target qubit, where multiple-controlled P gate uses multiple-controlled SU(2) implementation in [1]
    TODO: verifiy the implementation in [2]

    [1] Decomposition of Multi-controlled Special Unitary Single-Qubit Gates http://arxiv.org/abs/2302.06377
    [2] https://github.com/Qiskit/qiskit/blob/90e92a46643c72a21c5852299243213907453c21/qiskit/synthesis/multi_controlled/mcx_synthesis.py#L289
    """
    if len(controls) == 0:
        circuit.x(target)
    elif len(controls) == 1:
        circuit.cx(controls[0], target)
    elif len(controls) == 2:
        ccx(circuit, controls, target)
    elif len(controls) == 3:
        cccx(circuit, controls, target)
    elif len(controls) == 4:
        ccccx(circuit, controls, target)
    else:
        circuit.h(target)
        mc_p(circuit, numpy.pi, controls, target)
        circuit.h(target)

def mc_rot(circuit:qiskit.QuantumCircuit, axis:str, angle:float, controls:list[int], target:int):
    """
    Multiple-controlled rotation gate

    Eq. (16) in [1]
    Rx(theta) = H (Rz(theta/4) X Rz(-theta/4) X)^2 H   ## Typo in [1], the sign here is correct, or maybe different definition of RZ?
    Ry(theta) = (Ry(theta/4) X Ry(-theta/4) X)^2
    Rz(theta) = ( Rz(theta/4) X Rz(-theta/4) X )^2

    [1] Decomposition of Multi-controlled Special Unitary Single-Qubit Gates http://arxiv.org/abs/2302.06377
    """
    axis = axis.upper()
    if axis not in ['X', 'Y', 'Z']:
        raise ValueError("Invalid axis, must be X, Y, or Z")
    if len(controls) == 0:
        if axis == 'X':
            circuit.rx(angle, target)
        elif axis == 'Y':
            circuit.ry(angle, target)
        elif axis == 'Z':
            circuit.rz(angle, target)
        return
    if len(controls) == 1:
        if axis == 'X':
            circuit.crx(angle, controls[0], target)
        elif axis == 'Y':
            circuit.cry(angle, controls[0], target)
        elif axis == 'Z': ## https://github.com/Qiskit/qiskit/blob/90e92a46643c72a21c5852299243213907453c21/qiskit/circuit/library/standard_gates/equivalence_library.py#L585
            circuit.crz(angle, controls[0], target)
        return

    ##
    num_controls = len(controls) ## k in Fig 7
    k1 = int(numpy.ceil(0.5*num_controls)) ## k1 in Fig 7, k1 = ceil(k/2)
    k2 = num_controls - k1 ## k2 in Fig 7, k2 = floor(k/2)
    k1_controls = controls[:k1]
    k2_controls = controls[k1:]

    if axis == 'X': ## Note that 
        circuit.h(target)
        ##
        mc_x(circuit, k1_controls, target)
        circuit.rz(-0.25*angle, target)
        mc_x(circuit, k2_controls, target)
        circuit.rz(0.25*angle, target)
        ##
        mc_x(circuit, k1_controls, target)
        circuit.rz(-0.25*angle, target)
        mc_x(circuit, k2_controls, target)
        circuit.rz(0.25*angle, target)
        ##
        circuit.h(target)
    elif axis == 'Y':
        ##
        mc_x(circuit, k1_controls, target)
        circuit.ry(-0.25*angle, target)
        mc_x(circuit, k2_controls, target)
        circuit.ry(0.25*angle, target)
        ##
        mc_x(circuit, k1_controls, target)
        circuit.ry(-0.25*angle, target)
        mc_x(circuit, k2_controls, target)
        circuit.ry(0.25*angle, target)
    elif axis == 'Z':
        ##
        mc_x(circuit, k1_controls, target)
        circuit.rz(-0.25*angle, target)
        mc_x(circuit, k2_controls, target)
        circuit.rz(0.25*angle, target)
        ##
        mc_x(circuit, k1_controls, target)
        circuit.rz(-0.25*angle, target)
        mc_x(circuit, k2_controls, target)
        circuit.rz(0.25*angle, target)

## wrapper
def mc_rx(circuit:qiskit.QuantumCircuit, angle:float, controls:list[int], target:int):
    mc_rot(circuit, 'X', angle, controls, target)

def mc_ry(circuit:qiskit.QuantumCircuit, angle:float, controls:list[int], target:int):
    mc_rot(circuit, 'Y', angle, controls, target)

def mc_rz(circuit:qiskit.QuantumCircuit, angle:float, controls:list[int], target:int):
    mc_rot(circuit, 'Z', angle, controls, target)




## Tests
if __name__ == "__main__":
    ## Test for multiplexer_pauli
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UCPauliRotGate

    test_mcp = False
    test_mcx = True
    
    test_multiplexer_pauli = False
    
    test_mc_rot = False
    test_custom_controlled = False
    test_custom_controlled_qsdrev = False

    ##================== Tests for individual gates ==================##
    if test_mcp:
        print('='*20, "Test for Multiple-Controlled P", '='*20)
        seed = 71
        rng = numpy.random.default_rng(seed)
        def test_case(num_qubits, reps=5):
            print("\nNumber of Qubits:", num_qubits)
            for _ in range(reps):
                angle = rng.random(1)[0]*2*numpy.pi
                test_circ = qiskit.QuantumCircuit(num_qubits)
                mc_p(test_circ, angle, list(range(num_qubits-1)), num_qubits-1)

                qis_test_circ = qiskit.circuit.library.standard_gates.MCPhaseGate(angle, num_qubits-1).definition

                error = numpy.linalg.norm(qiskit.quantum_info.Operator(test_circ).data - qiskit.quantum_info.Operator(qis_test_circ).data)
                print("  - Error", error, "  angle=", angle)
                assert(error < 1e-8)

        for num_qubits in [2,3,4,5,6]:
            test_case(num_qubits)
            print()

    if test_mcx:
        print('='*20, "Test for Multiple-Controlled X", '='*20)
        def test_case(num_qubits, reps=5):
            print("\nNumber of Qubits:", num_qubits)
            for _ in range(reps):
                test_circ = qiskit.QuantumCircuit(num_qubits)
                mc_x(test_circ, list(range(num_qubits-1)), num_qubits-1)
                my_op_counts = dict(qiskit.transpile(test_circ, basis_gates=['cx','u'], optimization_level=0).count_ops())
                # test_circ = test_circ.reverse_bits()

                test_circ_qis = qiskit.QuantumCircuit(1)
                test_circ_qis.x(0)
                qis_test_circ = test_circ_qis.control(num_qubits-1)
                qis_op_counts = dict(qiskit.transpile(qis_test_circ, basis_gates=['cx','u'], optimization_level=0).count_ops())

                error = numpy.linalg.norm(qiskit.quantum_info.Operator(test_circ).data - qiskit.quantum_info.Operator(qis_test_circ).data)
                print("  - Error", error, "Qiskit Ops", qis_op_counts, "My Ops", my_op_counts)
                assert(error < 1e-8)

        for num_qubits in [2,3,4,5,6,7,8]:
            test_case(num_qubits)
            print()


    if test_multiplexer_pauli:
        print('='*20, "Test for Multiplerxer Paulis", '='*20)
        def test_case(axis, num_qubits, seed):
            rng = numpy.random.default_rng(seed)
            dim = 2**(num_qubits-1)
            angles = list(rng.random(dim)*2*numpy.pi)

            print("Number of Qubits:", num_qubits, "Axis:", axis, "Seed:", seed)
            print("Angles:", angles)

            ## my implementation
            circ = qiskit.QuantumCircuit(num_qubits)
            multiplexer_pauli(circ, angles, list(range(num_qubits-1)), num_qubits-1, axis)
            circ = circ.reverse_bits() ## not gonna follow qiskit rule in my implementation

            ## qiskit implementation
            circ_qiskit = qiskit.QuantumCircuit(num_qubits)
            circ_qiskit.append(UCPauliRotGate(angles, axis), list(range(num_qubits)))

            error = numpy.linalg.norm(Operator(circ).data - Operator(circ_qiskit).data)
            print("Error: ", error)
            assert(error < 1e-8)
                
        print("\nTest case 1")
        test_case('X', 1, 7)

        print("\nTest case 2")
        test_case('Y', 1, 13)

        print("\nTest case 3")
        test_case('Z', 2, 29)

        print("\nTest case 4")
        test_case('X', 3, 103)

        print("\nTest case 5")
        test_case('Z', 4, 211)

    ##================== Tests for combined gates ==================##
    if test_mc_rot:
        print('='*20, "Test for Multiple-Controlled RX, RY, RZ", '='*20)
        seed = 7
        rng = numpy.random.default_rng(seed)
        def test_case(axis, num_qubits):
            angle = rng.random(1)[0]*2*numpy.pi

            my_circ = qiskit.QuantumCircuit(num_qubits)
            mc_rot(my_circ, axis, angle, list(range(num_qubits-1)), num_qubits-1)

            my_op = qiskit.quantum_info.Operator(my_circ).data
            my_op_counts = dict(qiskit.transpile(my_circ, basis_gates=['cx','u'], optimization_level=1).count_ops())

            qis_circ = qiskit.QuantumCircuit(num_qubits)
            if axis == 'X':
                qis_circ.append( qiskit.circuit.library.RXGate(angle).control(num_qubits-1), range(num_qubits) )
            elif axis == 'Y':
                qis_circ.append( qiskit.circuit.library.RYGate(angle).control(num_qubits-1), range(num_qubits) )
            elif axis == 'Z':
                qis_circ.append( qiskit.circuit.library.RZGate(angle).control(num_qubits-1), range(num_qubits) )
            qis_op = qiskit.quantum_info.Operator(qis_circ).data

            qis_op_count = dict(qiskit.transpile(qis_circ, basis_gates=['cx','u'], optimization_level=1).count_ops())
            error = numpy.linalg.norm( my_op - qis_op )
            print("  - Error", error, "  angle=", angle)
            print("  - CX Count (Qiskit vs. Mine) - CX ", qis_op_count['cx'],'vs.', my_op_counts['cx'], "  - U", qis_op_count['u'],'vs.', my_op_counts['u'])
            assert(error < 1e-8)

        
        for axis in ['X', 'Y', 'Z']:
            for num_qubits in [2,3,4,5,6,7]:
                print("\nAxis:", axis, "Number of Qubits:", num_qubits)
                for _ in range(8):
                    test_case(axis, num_qubits)
                print()




    ##================== Tests for advanced function ==================##
    if test_custom_controlled:
        print('='*20, "Test for Custommed Controlled Circuit", '='*20)
        seed = 71
        rng = numpy.random.default_rng(seed)
        def test_case(num_target, num_controls, reps=5):
            print("\n#Target:", num_target, "#Control:", num_controls)
            for _ in range(reps):
                rc_seed = rng.integers(1000000)
                random_circuit = qiskit.circuit.random.random_circuit(num_target, num_target*5, max_operands=2, seed=rc_seed)
                trans_circ = qiskit.transpile(random_circuit, basis_gates=['cx','ry','rz','rx'], optimization_level=0)
                ##
                my_cc = selected_controlled_circuit(trans_circ, num_controls)
                my_cop = qiskit.quantum_info.Operator(my_cc).data
                ##
                qis_cc = trans_circ.control(num_controls)
                qis_cop =  qiskit.quantum_info.Operator(qis_cc ).data
                ##
                error = numpy.linalg.norm( my_cop - qis_cop )  
                print("  - Error", error)
                assert(error < 1e-8)

        for num_target in [1,2,3,4,5]:
            for num_controls in [1,2,3]:
                test_case(num_target, num_controls)
                print()
            
            
    if test_custom_controlled_qsdrev:
        print('='*20, "Test for Custommed Controlled QSD", '='*20)
        from scipy.stats import unitary_group
        from oracle_synth import synthu_qsd
        seed = 71
        rng = numpy.random.default_rng(seed)
        def test_case(num_target, num_controls, reps=5):
            print("\n#Target:", num_target, "#Control:", num_controls)
            for _ in range(reps):
                U = unitary_group.rvs(2**num_target)
                print("Eigenvals:", numpy.linalg.eigvals(U))
                random_circuit = synthu_qsd(U, cz_opt=True)
                random_circuit = qiskit.transpile(random_circuit, basis_gates=['cx','ry','rz','rx'], optimization_level=1)
                ##
                my_cc = selected_controlled_circuit(random_circuit, num_controls, to_known_basis=False, reverse_bits=True)
                my_cop = qiskit.quantum_info.Operator(my_cc).data
                ##
                qis_cc = random_circuit.decompose().reverse_bits().control(num_controls)
                qis_cop =  qiskit.quantum_info.Operator(qis_cc ).data
                ##
                error = numpy.linalg.norm( my_cop - qis_cop )  
                print("  - Error", error)
                assert(error < 1e-8)

        for num_target in [1,2,3,4,5]:
            for num_controls in [1,2,3]:
                test_case(num_target, num_controls)
                print()




