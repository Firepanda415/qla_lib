## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng

import numpy
import scipy 
import qiskit
from utils_synth import *

## TODO: Improve state preparation, see improved implementation in comments in stateprep_ucry.py
## TODO: Implement Appendix A2 optimization, see comments in synthu_qsd ## This is More Important
## TODO Weyl Chamber https://journals.aps.org/pra/pdf/10.1103/PhysRevA.63.062309

_ATOL = 1e-12
_ROUND = 12 ## round precision
_ATOL_MAT = 1e-10

####----------------------------------------- 1-qubit Unitary matrix Synthesis -----------------------------------------####

def synthu_1q(circuit:qiskit.QuantumCircuit, matrix:numpy.ndarray, qubit:int, debug:bool=False):
    '''
    U(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda),     where Ry(theta) = exp(-i theta Y/2), Rz(phi) = exp(-i phi Z/2) 
                          = [exp(-i(phi+lambda)/2)cos(theta/2)     -exp(-i(phi-lambda)/2) sin(theta/2)]
                            [exp(i(phi-lambda)/2) sin(theta/2)    exp(i(phi+lambda)/2) cos(theta/2)]
    [1] Defination see Eq. (2) https://arxiv.org/pdf/1707.03429
    [2] Solve for those angles, See Section 4.1 in https://threeplusone.com/pubs/on_gates.pdf
    '''
    if matrix.shape[0] != 2 or matrix.shape[1] != 2:
        raise ValueError("The matrix should be 2x2, but", matrix.shape, "were given")
    
    mat_expphase = mat_egh2su(matrix)
    matrix_su = matrix*mat_expphase
    ##
    theta = 2*numpy.arccos(numpy.abs(matrix_su[0,0])) if numpy.abs(matrix_su[0,0]) > numpy.abs(matrix_su[0,1]) else 2*numpy.arcsin( numpy.abs(matrix_su[0,1]))
    ##
    tmp1 = matrix_su[1,1]/numpy.cos(theta/2)
    RHS1 = 2*numpy.arctan2(tmp1.imag, tmp1.real)

    tmp2 = matrix_su[1,0]/numpy.sin(theta/2)
    RHS2 = 2*numpy.arctan2(tmp2.imag, tmp2.real)

    phi = 0.5*(RHS1+RHS2)
    lamb = RHS1 - phi
    if debug:
        print("DEBUG-angles(theta, phi, lambda)", theta, phi, lamb)
    #
    if debug:
        q1_circ = qiskit.QuantumCircuit(1)
        q1_circ.name = "1q U"
        global_phase_gate(q1_circ, numpy.angle(numpy.conj(mat_expphase)), 0, no_gate=True)
        q1_circ.rz(lamb, 0)
        q1_circ.ry(theta, 0)
        q1_circ.rz(phi, 0)
        ## Verify the decomposition
        if not numpy.allclose(qiskit.quantum_info.Operator(q1_circ).data, matrix):
            raise ValueError("Error on 1-qubit circuit:", numpy.linalg.norm( qiskit.quantum_info.Operator(q1_circ).data-matrix))
    ##
    global_phase_gate(circuit, numpy.angle(numpy.conj(mat_expphase)), qubit, no_gate=True)
    circuit.rz(lamb, qubit)
    circuit.ry(theta, qubit)
    circuit.rz(phi, qubit)




####----------------------------------------- 2-qubit Unitary matrix Synthesis -----------------------------------------####

def kak_decomp(matrix:numpy.ndarray, debug:bool=False):
    '''
    KAK decomposition (Cartan decomposition) for a 2-qubit unitary matrix
     1. Unitary V to Special Unitary U: V = e^{i phi} U => U = e^{i psi} V where psi = -phi
     2. Use magic matrix B to unconjugation U' = B^† U B
     3. Given a special unitary U, U = K W = K (K_2^† A K_2) = K_1 A K_2, where K_1 = KK_2^† and K_2 are local unitaries (tensor product of 2-by-2 unitaries) 
        and A is a diagonal matrix. Cartan involution function  Theta is used, such that 
            Theta(A) = A if A is like K matrix, and  Theta(A) = A^† if A is like W matrix
        and for 2-qubit unitary matrix U,  Theta(U) = U^* (conjugate), so it is clear that  Theta(U^†) = U^T
        Note 
            U^TU =  Theta(U^†) U =  Theta( W^† K^†) K W =  Theta( W^†)  Theta(K^†) K W = W K^† K W = W^2
        i.e., W =  sqrt{U^TU}
        We need diagonalization of W^2, which is diagonalization of U^TU,
            W^2 = U^TU = PDP^† => W = P D^{1/2} P^†
        P is also in special unitary as W share the same eigenvectors with W^2
        NOTE: as the result of square root, if det(P)=-1, we multiplying the 1st column of P by -1, also the 1st eigenvalue (1st entry of D) by -1
     4. So U = KW = K_1 A K_2 = K P D^{1/2} P^† indicate
            K_2 =  P^†   
            A = D^{1/2}   <- NOTE: det(A) = 1
            K_1 = KP = U K_2^† A^†
        This gives U' = K_1 A K_2 
    5. So U = B U' B^†
            = B K_1 A K_2  B^†
            = (B K_1 A B^†) (B A B^†) (B K_2 B^†)
       And L:=(B K_1 A B^†)  in SU(2)  ⊗ SU(2), R:=(B K_2 B^†)  in SU(2)  ⊗ SU(2)
       i.e, L = L_1 ⊗ L_2, R = R_1  ⊗ R_2
       So we seprate L1, L2, R1, R2 from L and R
    6. NOTE exp(i (a_0 IZ + a_1 ZI + a_2 ZZ)) = A, and A has determinant 1, so the diagonal elements of A are in the form of exp(i t_i) such that
             a_0+a_1_a_2 = t_0
            -a_0+a_1_a_2 = t_1
             a_0-a_1_a_2 = t_2
            -a_0-a_1+a_2 = -t_0-t_1-t_2
        we know t_i by finding the angle of diagonals of A, then we can solve for a_i's using the inverse of the coefficient matrix
            [1  0  1]
        1/2*[1  1  0]
            [0 -1 -1] 
    7. But we want to find B A B^†, given
         exp(i (c_0 XX + c_1 YY + c_2 ZZ) ) =  B A B^†
       we find c_0 = a_1, c_1 = -a_0, c_2 = a_2,
    8. Finally, we have U = exp(i s) (L_1  ⊗ L_2) exp(i (c_0 XX + c_1 YY + c_2 ZZ) ) (R_1  ⊗ R_2) for some global phase s
    9. Since XX, YY, ZZ commute with each other, we have exp(i (c_0 XX)). exp(i/2 (c_1 YY)). exp(i/2 (c_2 ZZ))
    Notations:
    Magic matrix B = [1 0  0  i]
        1/sqrt(2) *  [0 i  1  0]
                     [0 i -1  0]
                     [1 0  0 -i]
    [1] Section 3.2 for the whole process https://arxiv.org/pdf/0806.4015
    [2] Some implementation https://github.com/Qiskit/qiskit/blob/stable/0.45/qiskit/quantum_info/synthesis/two_qubit_decompose.py
    [3] Some implementation https://github.com/mpham26uchicago/laughing-umbrella/blob/main/background/Full%20Two%20Qubit%20KAK%20Implementation.ipynb
    [4] Figure 6 for the circuit https://arxiv.org/pdf/quant-ph/0308006
    [5] Figure 2 for the different circuits https://arxiv.org/pdf/2105.01063
    [6] TODO Weyl Chamber https://journals.aps.org/pra/pdf/10.1103/PhysRevA.63.062309
    '''
    # Magic basis transformation
    MAG = (1/numpy.sqrt(2)) * numpy.array([
        [1., 0., 0., 1.j],
        [0., 1.j, 1., 0.],
        [0., 1.j, -1., 0.],
        [1., 0., 0., -1.j]
    ])
    MAGdg = (1/numpy.sqrt(2)) * numpy.array([
        [ 1.,    0.,    0.,  1.],
        [ 0.,   -1.j,  -1.j, 0.],
        [ 0.,    1.,   -1.,  0.],
        [ -1.j,  0.,    0.,  1.j]
    ])

    mat_expphase = mat_egh2su(matrix) ## this is to find a e^{i psi} such that e^{i  psi}V is special unitary
    matrix_su = matrix*mat_expphase ## special unitary (determinant 1)
    matrix_sup = MAGdg.dot(matrix_su).dot(MAG)

    D, P = numpy.linalg.eig( numpy.round(matrix_sup.T.dot(matrix_sup), _ROUND) )
    if numpy.abs(numpy.linalg.det(P) + 1) < 1e-8:
        if debug:
            print("invert sign of 1st col of p")
        P[:, 0]*=-1 
    ##
    K2 = P.conj().T
    Adiag = numpy.sqrt(D)
    if numpy.isclose(numpy.prod(Adiag), -1):
        if debug:
            print("invert sign of 1st eigval")
        Adiag[0] *= -1 # Multiply the first eigenvalue by -1

    A = numpy.diag(Adiag)
    Aconj = numpy.diag(Adiag.conj())
    K1 = matrix_sup @ K2.conj().T @ Aconj

    K1p = MAG@K1@MAGdg 
    K2p = MAG@K2@MAGdg 
    expphase1, K1l, K1r = decompose_one_qubit_product(K1p)
    expphase2, K2l, K2r = decompose_one_qubit_product(K2p)

    # Cinv = numpy.array([
    #     [0.5, 0, 0.5],
    #     [0.5, 0.5, 0],
    #     [0, -0.5, -0.5]
    # ])
    t0,t1,t2 = numpy.angle(Adiag[0]), numpy.angle(Adiag[1]), numpy.angle(Adiag[2])
    a0 = 0.5*(t0+t2) 
    a1 = 0.5*(t0+t1) 
    a2 = 0.5*(-t1-t2) 
    # c0,c1,c2 = weyl_chamber(2*a1, -2*a0, 2*a2) ## exp(i/2 * c0 XX + c1 YY + c2 ZZ)
    c0,c1,c2 = a1, -a0, a2 ## exp(i * c0 XX + c1 YY + c2 ZZ)

    total_expphase = numpy.conj(mat_expphase)*expphase1*expphase2
    phase_angle = numpy.angle(total_expphase)

    ## Verify the decomposition
    I = numpy.identity(2)
    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    Z = numpy.array([[1, 0], [0, -1]])
    XX = numpy.kron(X, X)
    YY = numpy.kron(Y, Y)
    ZZ = numpy.kron(Z, Z)
    mat_recon = numpy.exp(1j*phase_angle) * (numpy.kron(K1l, K1r).dot( scipy.linalg.expm(1j*(c0*XX+c1*YY+c2*ZZ)) ).dot(numpy.kron(K2l,K2r)))
    if not numpy.linalg.norm(mat_recon - matrix) < _ATOL_MAT:
        print(f"   >>> KAK decomposition error: K1: {K1}, K2: {K2} <<<")
        print(f"   >>> KAK decomposition error: mat_expphase = {mat_expphase}, expphase1 = {expphase1}, expphase2 = {expphase2} <<<")
        print(f"   >>> KAK decomposition error: total_expphase = {total_expphase}, phase_angle = {phase_angle}, c0 = {c0}, c1 = {c1}, c2 = {c2} <<<")
        raise ValueError(f"The decomposition is not correct with error {numpy.linalg.norm(mat_recon - matrix)} for ", mat_recon, " and ", matrix)
    
    if debug:
        IZ = numpy.kron(I, Z) 
        ZI = numpy.kron(Z, I)
        ZZ = numpy.kron(Z, Z)
        print(f"P is orthogonal: {numpy.allclose(P.T@P, numpy.identity(4))}")
        print(f"det(P) = 1: {numpy.isclose(numpy.linalg.det(P), 1)}")
        print(f"det(A) = 1: {numpy.isclose(numpy.prod(Adiag), 1)}")
        print(f"K1 is orthogonal: {numpy.allclose(K1.T@K1, numpy.identity(4))}")
        print(f"det(K1) = 1: {numpy.isclose(numpy.linalg.det(K1), 1)}")
        print(f"KAK = U': {numpy.allclose(matrix_sup, K1@A@K2)}")
        print(f"K2' Correct: {numpy.allclose(expphase2*numpy.kron(K2l, K2r), K2)}")
        print(f"e^i(a.h)=A: {numpy.allclose(A, scipy.linalg.expm(1j*(a0*IZ+a1*ZI + a2*ZZ)))}")

        mat_reconsu = expphase1*expphase2*(numpy.kron(K1l, K1r).dot( scipy.linalg.expm(1j*(c0*XX+c1*YY+c2*ZZ)) ).dot(numpy.kron(K2l,K2r)))
        print(f"Urecon Correct (su): {numpy.allclose(mat_reconsu, matrix_su)}")
        print(f"Urecon Correct (u): {numpy.allclose(mat_recon, matrix)}")

    return phase_angle, K1l, K1r, c0,c1,c2, K2l,K2r




def synthu_kak(circuit:qiskit.QuantumCircuit, matrix:numpy.ndarray, qubits:list[int], debug:bool=False):
    """
    KAK decomposition for a 4-qubit unitary 
    Math behind KAK decompsoition see comments in kak_decomp()
    For circuits, see Figure 6 in [1]
      U = exp(i s) (L_1  ⊗ L_2) exp(i (c_0 XX + c_1 YY + c_2 ZZ) ) (R_1  ⊗ R_2) for some global phase s
    L_1, L_2, R_1, and R_2 can be construct as a general 1-qubit unitaries
    For exp(i (c_0 XX + c_1 YY + c_2 ZZ) ), we have circuits
        --[Global Phase exp(i pi/4)]---------------X--[Rz(2c_2 - pi/2)]--.----------------------X--[Rz(-pi/2)]---
                                                   |                     |                      |
        -------------------------------[Rz(pi/2)]--.--[Ry(pi/2 - 2c_0)]--X--[Ry(2beta - pi/2)]--.----------------

    NOTE: The definitions of Ry and Rz in the paper are different than the ones in Qiskit, need to flip the sign of all angles
    [1] Figure 6 for the circuit https://arxiv.org/pdf/quant-ph/0308006
    [2] https://github.com/Qiskit/qiskit/blob/stable/0.45/qiskit/quantum_info/synthesis/two_qubit_decompose.py
    """
    rz11 = numpy.sqrt(1/2)+1j*numpy.sqrt(1/2)
    rz00 = numpy.sqrt(1/2)-1j*numpy.sqrt(1/2)

    if len(qubits) != 2:
        raise ValueError("The number of qubits should be 2, but", len(qubits), "were given")
    if matrix.shape[0] != 4 or matrix.shape[1] != 4:
        raise ValueError("The matrix should be 4x4, but", matrix.shape, "were given")
    ##
    phase_angle, K1l, K1r, c0,c1,c2, K2l,K2r = kak_decomp(matrix, debug=debug)
    ##
    kak_circ = qiskit.QuantumCircuit(2)
    kak_circ.name = "2q U"
    global_phase_gate(kak_circ, phase_angle+numpy.pi/4, 0, no_gate=True)
    ## As Right side of the equation comes first in the circuit
    # kak_circ.unitary(K2l, 0)
    synthu_1q(kak_circ, K2l, 0)
    ## Absorb RZ into K2r
    K2r[0,0] = K2r[0,0] * rz11
    K2r[0,1] = K2r[0,1] * rz11
    K2r[1,0] = K2r[1,0] * rz00
    K2r[1,1] = K2r[1,1] * rz00
    # kak_circ.unitary(K2r, 1)
    synthu_1q(kak_circ, K2r, 1)
    ## exp(i (c_0 XX + c_1 YY + c_2 ZZ) )
    # kak_circ.rz(-numpy.pi/2 , 1) ## Absorbed RZ into K2r
    kak_circ.cx(1, 0)
    kak_circ.rz(numpy.pi/2 - 2*c2, 0)
    kak_circ.ry(2*c0 - numpy.pi/2, 1)
    kak_circ.cx(0, 1)
    kak_circ.ry(numpy.pi/2 - 2*c1, 1)
    kak_circ.cx(1, 0)
    # kak_circ.rz(numpy.pi/2, 0) ## Absorbed RZ into K1l
    ##
    ## Absorbed RZ into K1l
    K1l[0,0] = K1l[0,0] * rz00
    K1l[0,1] = K1l[0,1] * rz11
    K1l[1,0] = K1l[1,0] * rz00
    K1l[1,1] = K1l[1,1] * rz11
    # kak_circ.unitary(K1l, 0)
    synthu_1q(kak_circ, K1l, 0)
    # kak_circ.unitary(K1r, 1)
    synthu_1q(kak_circ, K1r, 1)
    ## Verify the decomposition
    synthesis_error = numpy.linalg.norm(qiskit.quantum_info.Operator(kak_circ.reverse_bits()).data - matrix)
    if  synthesis_error > _ATOL_MAT:
        raise ValueError("Error on KAK circuit:", synthesis_error)
    circuit.append(kak_circ.reverse_bits(), qubits) ## need to do reverse_bits() as Qiskit uses little-endian




####----------------------------------------- Higher-order Unitary matrix Synthesis -----------------------------------------####





def second_decomp(block_u1:numpy.ndarray, block_u2:numpy.ndarray, debug:bool=True):
    """
    Accoding to Eq. (16) in [1]
    Cosine-Sine decompsotion gives 
    U = [ A_1     ] [C    -S ] [A_2     ]
        [     B_1 ] [S     C ] [    B_2 ]
    where A_1, B_1, A_2, B_2 are unitary matrices in equal shapes
    Then, this function construct the 2nd decompsotion for each left and right block diagonal matrices
    [ U_1    ] = [V   ] [D          ] [W   ]
    [     U_2]   [   V] [   D^†] [   W]
    the constructon is like the following  => U_1 = VDW, U_2 = VD^† W
    We cancel out W terms and get U_1 U_2^† = V D^2 V^†
    Then we diagonalize U_1 U_2^† to obtain V (eigenvector matrix) and D^2 (eigenvalue matrix)
    Then D = sqrt(D^2), W = D V^† U_2 -> this is the equation in paper, but W = D^{-1} V^† U_1 is correct
    Note that [D          ] is a R_z multiplexer
              [   D^†]
    [1] Synthesis of quantum-logic circuits 10.1109/TCAD.2005.855930  or https://arxiv.org/pdf/quant-ph/0406176  (Seems like arxiv version has more details)
    """
    if block_u1.shape[0] != block_u1.shape[1] or block_u2.shape[0] != block_u2.shape[1]:
        raise ValueError('Inumpyut matrices must be square, but', block_u1.shape, block_u2.shape, 'were given')
    if block_u1.shape[0] != block_u2.shape[0]:
        raise ValueError('Inumpyut matrices must have the same size, but', block_u1.shape[0], block_u2.shape[0], 'were given')
    
    from qiskit.quantum_info.operators.predicates import is_hermitian_matrix ## 
    if is_hermitian_matrix(block_u1.dot( block_u2.T.conj() )):
        bu_evals, bu_v = scipy.linalg.eig(block_u1.dot( block_u2.T.conj() ) )
    else:
        bu_evals, bu_v = scipy.linalg.schur(block_u1.dot( block_u2.T.conj() ), output="complex" )
        bu_evals = bu_evals.diagonal()

    bu_d_inv = numpy.diag( 1/numpy.sqrt(bu_evals) )
    bu_w = bu_d_inv @ bu_v.T.conj() @ block_u1

    if debug:
        bu_d = numpy.diag( numpy.sqrt(bu_evals) )
        zeroes = numpy.zeros_like(block_u1)
        prod_mat = numpy.array([[bu_v, zeroes], [zeroes, bu_v]]) @ numpy.array([[bu_d, zeroes], [zeroes, bu_d.conj().T]]) @ numpy.array([[bu_w, zeroes],[zeroes, bu_w]])
        ans = numpy.array([[block_u1, zeroes], [zeroes, block_u2]])
        print("2nd decomp error", numpy.linalg.norm(prod_mat - ans))  

    return bu_v, numpy.sqrt(bu_evals), bu_w




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
        if abs(thetas[0]) >  _ATOL:
            circuit.ry(thetas[0], target)
        circuit.cz(controls[0], target)
        if abs(thetas[1]) >  _ATOL:
            circuit.ry(thetas[1], target)
    else:
        muxry_cz(circuit, thetas[:len(thetas)//2], controls[1:], target, angle_convert_flag=False)
        circuit.cz(controls[0], target)
        muxry_cz(circuit, thetas[len(thetas)//2:], controls[1:], target, angle_convert_flag=False)






def synthu_qsdcircuit(unitary:numpy.ndarray, circuit:qiskit.QuantumCircuit, bottom_control:list[int], top_target:int, cz_opt:bool=True, debug:bool=False):
    """
    Use uniformly controlled rotations to synthesize a unitary matrix
    Based on Quantum Shannon Decomposition (QSD) in [1]
    
    QSD decomposition gives M = U CS Vh, 
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
    TODO: Apply the CNOT gate optimization in Appendix A2 in [1]
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
        if u1err >  _ATOL:
            raise ValueError('Invalid 2nd decomposition for U, the error is', u1err)
        if u2err >  _ATOL:
            raise ValueError('Invalid 2nd decomposition for U, the error is', u2err)
    
    ## Recursively synthesize the unitaries
    if len(bottom_control) == 0: ## general single qubit gate, l=1
        # circuit.unitary(unitary, top_target) ## gives (3/4)*4**n - 1.5*(2**n) CNOT gates
        synthu_1q(circuit, unitary, top_target, debug=debug)
        return
    if len(bottom_control) == 1: ## general two-qubit gate, l=2
        # circuit.unitary(unitary, list(bottom_control)+[top_target]) ## gives (9/16)*4**n - 1.5*(2**n) CNOT gates without CZ optimization
        synthu_kak(circuit, unitary, list(bottom_control)+[top_target], debug=debug) ## gives (9/16)*4**n - 1.5*(2**n) CNOT gates without CZ optimization
        return
    
    ## v
    v_v, v_dd, v_w = second_decomp(v1h, v2h, debug=debug)
    v_zangle = list(numpy.angle(v_dd)* (-2)) ## R_z(lambda) = exp(-i lambda Z/2)
    synthu_qsdcircuit(v_w, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)
    multiplexer_pauli(circuit, v_zangle, bottom_control, top_target, axis='Z') ## V_D
    synthu_qsdcircuit(v_v, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)

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
    u_v, u_dd, u_w = second_decomp(u1, u2, debug=debug)
    u_zangle = list(numpy.angle(u_dd)* (-2)) ## R_z(lambda) = exp(-i lambda Z/2)
    synthu_qsdcircuit(u_w, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)
    multiplexer_pauli(circuit, u_zangle, bottom_control, top_target, axis='Z') ## U_D
    synthu_qsdcircuit(u_v, circuit, bottom_control[1:], bottom_control[0], cz_opt=cz_opt, debug=debug)


def synthu_qsd(unitary:numpy.ndarray, cz_opt:bool=True, debug:bool=False):
    ## Wrapper for synthu_qsdcircuit
    num_qubits = int(numpy.log2(unitary.shape[0]))
    circuit = qiskit.QuantumCircuit(num_qubits)
    synthu_qsdcircuit(unitary, circuit, list(range(num_qubits))[1:], 0, cz_opt=cz_opt, debug=debug)
    return circuit




####----------------------------------------- State Preparation -----------------------------------------####


def vec_mag_angles(complex_vector:numpy.ndarray):
    norm_vector = numpy.array(complex_vector)
    for i in range(len(complex_vector)):
        entry_norm = numpy.abs(complex_vector[i])
        if entry_norm >  _ATOL:
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

    if numpy.abs(tmp2_sum) <  _ATOL:
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
    TODO: Implement the optimized method in [2]

    [1] Transformation of quantum states using uniformly controlled rotations http://arxiv.org/abs/quant-ph/0407010
    [2] Quantum circuits with uniformly controlled one-qubit gates http://arxiv.org/abs/quant-ph/0410066
    '''

    num_qubits = int(numpy.log2(len(init_state)))
    psi_mag, psi_angles = vec_mag_angles(init_state)

    ## Circuit
    global_phase_gate(circuit, numpy.sum(psi_angles)/(2**num_qubits), 0, no_gate=True)

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
        if numpy.linalg.norm(zangles, ord=1) >  _ATOL:
            multiplexer_pauli(circuit,zangles, list(range(j-1)), j-1, axis='Z')
            if debug:
                circuit.barrier()













####----------------------------------------- Tests -----------------------------------------####

if __name__ == "__main__":
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    from scipy.stats import unitary_group


    rng = numpy.random.default_rng(429096209556973234794512152190348779897183667923694427)

    test_1q = False
    test_kak = False

    test_state_prep = False #False 
    test_unitary_synth = True

    ########################################## 1Q ################################################################
    print("\n\n\n\n\n")
    print("="*100)

    if test_1q:
        print(">>>>>>>>>>1Q Test<<<<<<<<<<<")
        for _ in range(10):
            dummy_circ = qiskit.QuantumCircuit(1)
            dummy_qubit = 0
            synthu_1q(dummy_circ, unitary_group.rvs(2), dummy_qubit, debug=True)
            print()



    ########################################## KAK ################################################################
    print("\n\n\n\n\n")
    print("="*100)

    if test_kak:
        print(">>>>>>>>>>2Q (KAK) Test<<<<<<<<<<")
        for _ in range(10):
            ## Test in matrices if the circuit is correct
            # from _quick_test import * 
            # def XXYYZZ_mat(c0,c1,c2):
            #     return scipy.linalg.expm(1j*(c0*XX+c1*YY+c2*ZZ))
            # def XXYYZZ_circ(c0,c1,c2):
            #     prod = np.kron(I(), RZ(-pi/2))
            #     prod = CNOTINV() @ prod
            #     prod = np.kron(RZ(pi/2 - 2*c2), RY(2*c0 - pi/2)) @ prod
            #     prod = CNOT() @ prod
            #     prod = np.kron(I(), RY(pi/2 - 2*c1)) @ prod
            #     prod = CNOTINV() @ prod
            #     prod = np.kron(RZ(pi/2), I()) @ prod
            #     return prod*numpy.exp(1j*pi/4)
            # c0,c1,c2 = rng.uniform(-pi,pi,3)
            # ans = XXYYZZ_mat(c0,c1,c2)
            # circ_op = XXYYZZ_circ(c0,c1,c2)
            # numpy.linalg.norm(ans - circ_op)
            ##
            dummy_circ = qiskit.QuantumCircuit(2)
            dummy_qubits = [0,1]
            synthu_kak(dummy_circ, unitary_group.rvs(4), dummy_qubits, debug=True)
            # kak_decomp(unitary_group.rvs(4), debug=True)
            print()


    ##########################################################################################################
    print("\n\n\n\n\n")
    print("="*100)

    if test_state_prep:
        print(">>>>>>>>>>State Preparation Test<<<<<<<<<<")
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

                qiscirc_trans = transpile(qiscirc, basis_gates=['u', 'cx'], optimization_level=0)
                qiscirc_trans_opt = transpile(qiscirc, basis_gates=['u', 'cx'], optimization_level=2)
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

                my_ucr_circuit_trans = transpile(my_ucr_circuit, basis_gates=['u', 'cx'], optimization_level=0)
                my_ucr_circuit_trans_opt = transpile(my_ucr_circuit, basis_gates=['u', 'cx'], optimization_level=2)
                ucr_op_dict = dict(my_ucr_circuit_trans.count_ops())
                ucr_op_dict_opt = dict(my_ucr_circuit_trans_opt.count_ops())
                print("    - Theoretical UCR lower bound: ", 2*(2**n) - 2*n -2) ## See https://arxiv.org/pdf/quant-ph/0406176 from  https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/circuits/3_summary_of_quantum_operations.ipynb
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
                assert(numpy.linalg.norm(qiskit.quantum_info.Statevector(my_ucr_circuit).data-psi) < 1e-10)


    ##########################################################################################################
    print("\n\n\n\n\n")
    print("="*100)

    if test_unitary_synth:
        print(">>>>>>>>>>Unitary Synthesis Test<<<<<<<<<<")
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

                qiscirc_trans = transpile(qiscirc, basis_gates=['u', 'cx'], optimization_level=0)
                qiscirc_trans_opt = transpile(qiscirc, basis_gates=['u', 'cx'], optimization_level=2)
                print("    - Qiskit Unitary Circuit Op Count", dict(qiscirc_trans.count_ops()) )
                print("    - Qiskit Unitary Optimized Circuit Op Count", dict(qiscirc_trans_opt.count_ops()) )
                print("    - Qiskit Unitary error", numpy.linalg.norm(qiscirc_mat - U) )


                ## My QSD implementation
                print("  \nQSD Unitary Synthesis")
                # my_circuit = QuantumCircuit(n)
                # synthu_qsd(U, my_circuit, list(range(n))[1:], 0, cz_opt=True)
                my_circuit = synthu_qsd(U, cz_opt=True)
                my_circuit = my_circuit.reverse_bits() ## Hi, not gonna follow qiskit rule in my implementation

                my_circ_mat = Operator(my_circuit).data

                my_circuit_trans = transpile(my_circuit, basis_gates=['u', 'cx'], optimization_level=0)
                my_circuit_trans_opt = transpile(my_circuit, basis_gates=['u', 'cx'], optimization_level=2)

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
                    print(f">>>>Depth Summary: Qiskit={qiscirc_trans.depth()}, Qiskit_opt={qiscirc_trans_opt.depth()}, QSD={my_circuit_trans.depth()}, QSD_opt={my_circuit_trans_opt.depth()}<<<<")
                    print(f">>>>CX Summary: Qiskit={dict(qiscirc_trans.count_ops())['cx']}, QSD={dict(my_circuit_trans.count_ops())['cx']}, QSD_opt={dict(my_circuit_trans_opt.count_ops())['cx']}<<<<")
                    print(f">>>>Total Gates Summary: Qiskit={numpy.sum(list(dict(qiscirc_trans.count_ops()).values()))}, Qiskit_opt={numpy.sum(list(dict(qiscirc_trans_opt.count_ops()).values()))}, QSD={numpy.sum(list(dict(my_circuit_trans.count_ops()).values()))},QSD={numpy.sum(list(dict(my_circuit_trans_opt.count_ops()).values() ))}")
                assert(numpy.linalg.norm(my_circ_mat - U) < 1e-8)
