## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng



import numpy
import qiskit


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







#--------------------------------  
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

# #--------------------------------   
# def get_minor(matrix, i, j):
#     """
#     Get the minor of the matrix excluding the i-th row and j-th column.
    
#     Parameters:
#     matrix (list of lists): The original matrix.
#     i (int): Row to exclude (0-based index).
#     j (int): Column to exclude (0-based index).
    
#     Returns:
#     list of lists: The minor matrix.
#     """
#     return [row[:j] + row[j+1:] for idx, row in enumerate(matrix) if idx != i]

# def detm4(matrix):
#     """
#     Calculate the determinant of a 4x4 matrix manually using Laplace expansion.
    
#     Parameters:
#     matrix (list of lists): A 4x4 matrix.
    
#     Returns:
#     complex or float: Determinant of the matrix.
#     """
#     # Validate matrix dimensions
#     if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
#         raise ValueError("The matrix must be 4x4.")
    
#     det = 0
#     for col in range(4):
#         # Calculate minor for element (0, col)
#         minor = get_minor(matrix, 0, col)
        
#         # Recursive call for 3x3 minor
#         det_minor = detm3(minor)
        
#         # Cofactor sign
#         sign = (-1) ** (col)
        
#         # Add to determinant
#         det += sign * matrix[0][col] * det_minor
#     return det

# def detm3(matrix):
#     """
#     Calculate the determinant of a 3x3 matrix using Laplace expansion.
    
#     Parameters:
#     matrix (list of lists): A 3x3 matrix.
    
#     Returns:
#     complex or float: Determinant of the matrix.
#     """
#     # Base case for 2x2 matrix
#     if len(matrix) == 2 and len(matrix[0]) == 2:
#         return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    
#     det = 0
#     for col in range(3):
#         minor = get_minor(matrix, 0, col)
#         det_minor = detm2(minor)
#         sign = (-1) ** (col)
#         det += sign * matrix[0][col] * det_minor
#     return det

# def detm2(matrix):
#     """
#     Calculate the determinant of a 2x2 matrix.
    
#     Parameters:
#     matrix (list of lists): A 2x2 matrix.
    
#     Returns:
#     complex or float: Determinant of the matrix.
#     """
#     return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

# def detm(mat):
#     """
#     Calculate the determinant of a square matrix using Laplace expansion.
    
#     Parameters:
#     matrix (list of lists): A square matrix.
    
#     Returns:
#     complex or float: Determinant of the matrix.
#     """
#     if type(mat) == numpy.ndarray:
#         matrix = mat.tolist()
#     else:
#         matrix = mat

#     # Base case for 2x2 matrix
#     if len(matrix) == 2 and len(matrix[0]) == 2:
#         return detm2(matrix)
    
#     # Base case for 3x3 matrix
#     if len(matrix) == 3 and len(matrix[0]) == 3:
#         return detm3(matrix)
    
#     # Base case for 4x4 matrix
#     if len(matrix) == 4 and len(matrix[0]) == 4:
#         return detm4(matrix)
    
#     # Recursive case for larger matrices
#     det = 0
#     for col in range(len(matrix)):
#         minor = get_minor(matrix, 0, col)
#         det_minor = detm(minor)
#         sign = (-1) ** (col)
#         det += sign * matrix[0][col] * det_minor
#     return det
# #--------------------------------   




## From https://github.com/mpham26uchicago/laughing-umbrella/blob/main/background/Full%20Two%20Qubit%20KAK%20Implementation.ipynb
def decompose_one_qubit_product(Umat: numpy.ndarray):
    """
    Decompose a 4x4 unitary matrix to two 2x2 unitary matrices.
    Args:
        U (np.ndarray): input 4x4 unitary matrix to decompose.
    Returns:
        exp_phase (float): exp(1j*global phase).
        U1 (np.ndarray): decomposed unitary matrix U1.
        U2 (np.ndarray): decomposed unitary matrix U2.
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












#--------------------------------   

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

## -------------------


def global_phase_gate(circuit:qiskit.QuantumCircuit, phase:float, target:int):
    """
    Apply a global phase gate to a quantum circuit
    Ph(theta) = [[ exp(i theta)   0           ] = P(theta)XP(theta)X
                 [ 0              exp(i theta)]]
    NOTE  X = [[0 1]  and P(theta) = [[1 0           ]    in Qiskit definition 
               [1 0]]                 [0 exp(i theta)]]
    """
    circuit.x(target)
    circuit.p(phase, target)
    circuit.x(target)
    circuit.p(phase, target)

# def mat1q_det(uni_mat):
#     return uni_mat[0,0] * uni_mat[1,1] - uni_mat[0,1] * uni_mat[1,0]


## Looks like some bug in the function, some times ry need -theta1, some times +theta1
# def su1q_gate(circuit:qiskit.QuantumCircuit,unitary:numpy.ndarray, target:int):
#     """
#     Apply a single qubit unitary gate to a quantum circuit, ZYZ decomposition
#     U = exp(i alpha) R_z(theta_2) R_y(theta_1) R_z(theta_0) where exp(i alpha) is the global phase global_phase_gate(alpha)
#     See Section 4.1 in https://threeplusone.com/pubs/on_gates.pdf
#     """
#     m,n = unitary.shape
#     if m != 2 or n != 2:
#         raise ValueError("The input matrix should be 2x2, but", unitary.shape, "is given")
    

#     alpha = 0.5*numpy.arctan2(mat1q_det(unitary).imag, mat1q_det(unitary).real)
#     # alpha = 0.5*numpy.arctan2(mat1q_det(U).real, mat1q_det(U).imag)
#     V = numpy.exp(-1j*alpha) * unitary
#     if numpy.abs(mat1q_det(V) - 1) > 1e-12:
#         raise ValueError('Invalid global phase, the determinant is', mat1q_det(V))
#     print(V)
#     theta1 = 2*numpy.arccos(numpy.abs(V[0,0])) if numpy.abs(V[0,0]) >= numpy.abs(V[0,1]) else 2*numpy.arcsin(numpy.abs(V[0,1]))

#     if numpy.abs(numpy.cos(0.5*theta1)) < 1e-12:
#         tmp1 = 0
#     else:
#         tmp1 = 2 * numpy.arctan2( (V[1,1]/numpy.cos(0.5*theta1)).imag , (V[1,1]/numpy.cos(0.5*theta1)).real )
#     if numpy.abs(numpy.sin(0.5*theta1)) < 1e-12:
#         tmp2 = 0
#     else:
#         tmp2 = 2 * numpy.arctan2( (V[1,0]/numpy.sin(0.5*theta1)).imag , (V[1,0]/numpy.sin(0.5*theta1)).real )
#     theta0 = 0.5*(tmp1 + tmp2)
#     theta2 = 0.5*(tmp1 - tmp2)

#     circuit.rz(theta0, 0)
#     circuit.ry(theta1, 0)
#     circuit.rz(theta2, 0)
#     global_phase_gate(circuit, alpha, 0)






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



# def mux_qiswrapper(angles:list[float], controls:list[int], target:int, axis:str):
#     tmp_circuit = qiskit.QuantumCircuit(len(controls)+1)
#     multiplexer_pauli(tmp_circuit, angles, controls, target, axis)
#     tmp_circuit.reverse_bits()
#     return tmp_circuit









## Tests
if __name__ == "__main__":

    ## 




    ## Test for multiplexer_pauli
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UCPauliRotGate

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
        
        print("Error: ", numpy.linalg.norm(Operator(circ).data - Operator(circ_qiskit).data))
              
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
