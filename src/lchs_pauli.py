## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng
import numpy
import qiskit.quantum_info
import scipy
import qiskit
from typing import Callable

from oracle_synth import *

from lchs import *





### Construct matrix from linear combination of Pauli strings
def pauli_str_to_mat(pauli_str:str) -> numpy.matrix:
    """
    Construct matrix from Pauli string
    Input:
    - pauli_str: string, Pauli string
    Output:
    - numpy matrix, the matrix representation of the Pauli string
    """
    X = numpy.matrix([[0,1],[1,0]])
    Y = numpy.matrix([[0,-1j],[1j,0]])
    Z = numpy.matrix([[1,0],[0,-1]])
    I = numpy.matrix([[1,0],[0,1]]) 
    num_qubits = len(pauli_str)
    ## Initialize the matrix
    mat = 1
    ## Fill the matrix
    for i, pauli in enumerate(pauli_str):
        if pauli == 'X':
            mat = numpy.kron(mat, X)        
        elif pauli == 'Y':
            mat = numpy.kron(mat, Y)
        elif pauli == 'Z':
            mat = numpy.kron(mat, Z)
        elif pauli == 'I':
            mat = numpy.kron(mat, I)
        else:
            raise ValueError("Invalid Pauli string")
    return mat

def pauli_arr_to_mat(coef_arr:numpy.ndarray, pauli_arr:list[str]) -> numpy.matrix:
    """
    Construct matrix from linear combination of Pauli strings
    Input:
    - coef_arr: numpy array, coefficients of the linear combination
    - pauli_arr: list of strings, Pauli strings
    Output:
    - numpy matrix, the matrix representation of the linear combination of Pauli strings
    """
    if len(coef_arr) != len(pauli_arr):
        raise ValueError("coef_arr and pauli_arr must have the same length")
    num_qubits = len(pauli_arr[0])
    ## Initialize the matrix
    mat = numpy.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for coef, pauli in zip(coef_arr, pauli_arr):
        mat += coef*pauli_str_to_mat(pauli)
    return mat





def pauli_lchs_tihs(L_coef_arr:numpy.ndarray, L_pauli_arr:list[str], 
                    H_coef_arr:numpy.ndarray, H_pauli_arr:list[str], 
                    u0:numpy.matrix, 
                    tT:float, beta:float, epsilon:float, 
                    trunc_multiplier=2,
                    qiskit_api:bool=False, verbose:int=0, 
                    no_state_prep:bool=False,
                    debug:bool=False, rich_return:bool=False) -> tuple[numpy.matrix,numpy.matrix]: 
    '''
    Solve Homogeneous IVP du/dt = -Au(t) with u(0) = u0, where A = L + H and L and H are linear combinations of Pauli strings
    For real symmetric A, H can be zero matrix, in this case, H_coef_arr and H_pauli_arr can be empty
    use Trotterization for exp(-i (kL+H)t) => exp(-i H t)exp(-i kL t) if true, so exp(i H t) is only applied once for all k
    Input:
    - L_coef_arr: numpy array, coefficients of the linear combination of L
    - L_pauli_arr: list of strings, Pauli strings of the linear combination of L
    - H_coef_arr: numpy array, coefficients of the linear combination of H
    - H_pauli_arr: list of strings, Pauli strings of the linear combination of H
    - u0: numpy.matrix, initial condition
    - tT: float, end time
    - beta: float, 0 < beta < 1, parameter in the kernel function
    - epsilon: float, error tolerance
    - trunc_multiplier: float, multiplier for truncation range, just for higher accuracy. See (63) in [1]. Default is 2.
    - verbose: bool, print out parameters
    Output:
    - res_mat: numpy matrix, the matrix representation of the solution operator (summation of all unitaties)
    - uT: numpy matrix, the solution state at time tT, uT = res_mat*u0
    '''
    if len(L_coef_arr) != len(L_pauli_arr):
        raise ValueError("L_coef_arr and L_pauli_arr must have the same length")
    if H_coef_arr is not None and H_pauli_arr is not None and len(H_coef_arr) != len(H_pauli_arr):
        raise ValueError("H_coef_arr and H_pauli_arr must have the same length")
    empty_H = False
    if len(H_coef_arr) == 0 and len(H_pauli_arr) == 0:
        empty_H = True
            

    from lcu import lcu_generator_pauli
    from utils_synth import qiskit_normal_order_switch, qiskit_normal_order_switch_vec
    # from oracle_synth import synthu_qsd, stateprep_ucr
    ##
    u0_norm = u0/numpy.linalg.norm(u0,ord=2)
    # L,H = cart_decomp(A)
    L = pauli_arr_to_mat(L_coef_arr, L_pauli_arr)
    if not empty_H:
        H = pauli_arr_to_mat(H_coef_arr, H_pauli_arr)
    else:
        H = numpy.zeros_like(L)
    h1 = step_size_h1(tT, L)  ## step size h1 for [-K, K], (65) in [1]
    K = trunc_multiplier*trunc_K(beta, epsilon, h1) ## integral range -K to K, (63) in [1] ## "2" here is just for higher accuracy since formula is big-O
    Q = n_node_Q(beta, epsilon, K)  ## number of nodes in each subinterval [mh_1, (m+1)h_1] for m = -K/h1 to K/h1, (65) in [1]
    ##
    kh1 = int(K/h1)
    M = int(2*kh1*Q) ## number of nodes in total, (65) in [1] ## just for checking
    if verbose:
        print("  Preset parameters T = ", tT, "beta = ", beta, "epsilon = ", epsilon)
        print("  Truncation range [-K,K] K =", K)
        print("  Step size h1 =", h1)
        print("  Number of nodes in [mh_1, (m+1)h_1] Q =", Q)
        print("  Total number of nodes M =", M)
        ## truncation error bound, (62) in [1]
        oneoverbeta_ceil = int(numpy.ceil(1/beta))
        truncation_error_bound_num = 2**(oneoverbeta_ceil+1)*scipy.special.factorial(oneoverbeta_ceil)*numpy.exp(-0.5*(K**beta)*numpy.cos(beta*numpy.pi*0.5))
        truncation_error_bound_denom = cbeta(beta)*(numpy.cos(beta*numpy.pi*0.5)**oneoverbeta_ceil)*K
        truncation_error_bound = truncation_error_bound_num/truncation_error_bound_denom
        print("  Truncation error bound =", truncation_error_bound)
        ## quadrature error bound, (64) in [1]
        quadrature_error_bound = 8/(3*cbeta(beta))*K*h1**(2*Q) * (0.5*numpy.exp(1)*tT*numpy.linalg.norm(L,ord=2))**(2*Q)
        print("  Quadrature error bound =", quadrature_error_bound)
        print("  Total error bound =", truncation_error_bound + quadrature_error_bound)
    ##
    c_sum = 0 ## compute ||c||_1
    coeffs_unrot = []
    for munshit in range(2*kh1):
        m = -kh1+munshit ## shift to start from -K/h1
        kqms, wqs = gauss_quadrature(m*h1, h1, Q) ## Gaussian quadrature points and weights in [mh_1, (m+1)h_1]
        ## Since using LCU, only compute coefficients and unitaries
        for qi in range(Q):
            cqm = wqs[qi]*gk(beta, kqms[qi])
            c_sum += numpy.abs(cqm)
            ## Collect coeffs for w_q*g(k_{q,m})*U(T, k_{q,m}) ## (61) in [1], also multiple by time
            coeffs_unrot.append( cqm*tT )
    if verbose > 0:
        print("  ||c||_1 =", c_sum)
        
    ## Obtain the linear combination by LCU
    # lcu_circ = lcu_generator(coeffs, unitaries, initial_state_circ=state_prep_circ, qiskit_api=qiskit_api) ## NOTE: the return circuit is in qiskit order
    lcu_circ, coeffs, coeffs_phases, coeffs_1norm = lcu_generator_pauli(coeffs_unrot, L_coef_arr, L_pauli_arr,
                                                    initial_state_circ=None, verbose=verbose, qiskit_api=qiskit_api, debug=debug) ## NOTE: the return circuit is NOT in qiskit order
    if not empty_H:
        num_control_qubits = nearest_num_qubit(len(coeffs))
        ## exp(i(A+B)) approx exp(iA/2) exp(iB) exp(iA/2), (4.104) in Nielsen and Chuang (10th anniversary edition)
        exph_circ = pauli_expoent_circ(0.5*tT, H_coef_arr, H_pauli_arr, qiskit_api=qiskit_api)
        exph_circ = exph_circ.reverse_bits()
        lcu_circ.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=True, inplace=True)
        lcu_circ.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=False, inplace=True)

    if verbose > 0:
        print("  Number of Qubits:", lcu_circ.num_qubits)

    if verbose > 0:
        trans_lcu_opt0 = qiskit.transpile(lcu_circ, basis_gates=['cx', 'u'], optimization_level=0)
        trans_lcu_opt2 = qiskit.transpile(lcu_circ, basis_gates=['cx', 'u'], optimization_level=2)
        print("  Transpiled LCU Circ Stats (Opt 0):", trans_lcu_opt0.count_ops())
        print("    Circuit Depth (Opt 0):", trans_lcu_opt0.depth())
        print("  Transpiled LCU Circ Stats (Opt 2):", trans_lcu_opt2.count_ops())
        print("    Circuit Depth (Opt 2):", trans_lcu_opt2.depth())

    if no_state_prep:
        # print(Warning("Simulation is DISABLED for a quick test"))
        print(">> State preparation for initial condition is DISABLED <<")
        circ_op = qiskit.quantum_info.Operator( lcu_circ ).data ## LCU has reversed qubits
        sum_op = coeffs_1norm * qiskit_normal_order_switch( circ_op[:L.shape[0],:L.shape[1]] ) ## See lcu_generator use case example
        ## Compute u0
        uT = sum_op.dot(u0_norm)

        # uT = qiskit.quantum_info.Statevector(lcu_circ).data[:H.shape[0]]
        ##\
        if rich_return:
            return uT, lcu_circ, circ_op, coeffs, unitaries, coeffs_unrot, unitaries_unrot
        return uT, lcu_circ

    ## State Preparation
    u0_norm_flatten = numpy.array(u0_norm).flatten() ## do not accept numpy.matrix
    state_prep_circ = qiskit.QuantumCircuit(int(numpy.log2(H.shape[0])))
    if qiskit_api:
        state_prep_circ.initialize(u0_norm_flatten)
        state_prep_circ = state_prep_circ.reverse_bits() ## my lcu do not use qiskit order
    else:
        stateprep_ucr(u0_norm, state_prep_circ)
    lcu_circ.compose(state_prep_circ, qubits=range(int(numpy.log2(H.shape[0]))), front=True, inplace=True)

    ## extra state vector when last (as I did not follow qiskit order) num_control_qubits of control qubits are in state |0>
    full_sv = qiskit.quantum_info.Statevector(lcu_circ).data
    uT = full_sv[:H.shape[0]]
    uT = coeffs_1norm*qiskit_normal_order_switch_vec(uT) ## no need to normalize
    if rich_return:
        return uT, lcu_circ, full_sv, coeffs, unitaries, coeffs_unrot, unitaries_unrot, 
    return uT, lcu_circ






## Test
if __name__ == "__main__":
    rng = numpy.random.Generator(numpy.random.PCG64(178984893489))
    ## Example
    # omegasq = 1.1**2
    # A_pauli_arr = ['X', 'Y']
    # A_coef_arr = numpy.array([0.5*(omegasq-1), 0.5*(-omegasq-1)])
    # L_pauli_arr = ['X']
    # L_coef_arr = numpy.array([0.5*(omegasq-1)])
    # H_pauli_arr = ['Y']
    # H_coef_arr = numpy.array([0.5*(-omegasq-1)])


    hf = 8
    A_pauli_arr = ['II', 'IX', 'XX', 'YY']
    A_coef_arr = numpy.array([2, -1, -0.5, -0.5]) * 1/(hf**2)
    L_pauli_arr = A_pauli_arr
    L_coef_arr = A_coef_arr.copy()
    H_pauli_arr = []
    H_coef_arr = numpy.array([])
    #
    A = pauli_arr_to_mat(A_coef_arr, A_pauli_arr)
    L = pauli_arr_to_mat(L_coef_arr, L_pauli_arr)
    if len(H_pauli_arr) > 0:
        H = pauli_arr_to_mat(H_coef_arr, H_pauli_arr)
    else:
        H = numpy.zeros_like(L)

    L_verify,H_verify = cart_decomp(A)
    dim = A.shape[0]
    print(f"Norm of A: {numpy.linalg.norm(A, ord=2)}, Norm of L: {numpy.linalg.norm(L, ord=2)}, Norm of H: {numpy.linalg.norm(H, ord=2)}")
    print(f"Verify L: {numpy.linalg.norm(L - L_verify, ord=2)}, Verify H: {numpy.linalg.norm(H - H_verify, ord=2)}")
    ### Quick check for eigenvalues
    print("Eigenvalues of L:", numpy.linalg.eigvals(L))
    ## Random initial state
    u0 = numpy.matrix( rng.random((dim,1)) ,dtype=complex) ## dtype complex is necessary for scipy to compute complex integral
    u0 = u0/numpy.linalg.norm(u0, ord=2)

    T = 0.1
    beta = 0.9 # 0< beta < 1
    epsilon = 0.05 #0.05
    debug = False


    ## Function for Scipy
    def ode_func_ho(t,u):
        return numpy.array(-A.dot(u).reshape(-1))[0]
    ### Scipy Homogenous solution
    spi_sol_ho = scipy.integrate.solve_ivp(ode_func_ho, [0,T],numpy.array(u0.reshape(-1))[0], method='RK45')
    spi_uT_ho = spi_sol_ho.y[:,-1]
    spi_uT_ho_norm = spi_uT_ho/numpy.linalg.norm(spi_uT_ho,ord=2)
    if numpy.linalg.norm(spi_uT_ho.imag,ord=2) < 1e-12:
        spi_uT_ho = spi_uT_ho.real

    print("\n\nTests with Classical Subroutine (Homogeneous, no trotter)")
    ## Solve homogenous part
    exp_op, uT = class_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=2, trotterLH=False, verbose=1)
    uT = numpy.array(uT).reshape(-1)
    if numpy.linalg.norm(uT.imag,ord=2) < 1e-12:
        uT = uT.real
    uT_err = numpy.linalg.norm(uT - spi_uT_ho,ord=2)
    norm_ratio = numpy.linalg.norm(uT)/numpy.linalg.norm(spi_uT_ho)
    uT2_err = numpy.linalg.norm(uT/norm_ratio - spi_uT_ho,ord=2)
    print("  Homogeneous u(T)=           ", uT, "  Norm=", numpy.linalg.norm(uT,ord=2))
    print("  SciPy Sol   u(T)=           ", spi_uT_ho, "  Norm=", numpy.linalg.norm(spi_uT_ho,ord=2))
    print("  Homogeneous u(T)/norm_ratio=", uT/norm_ratio, "  Norm=", numpy.linalg.norm(uT/norm_ratio,ord=2))
    print("  Homogeneous solution error u(T)           :", uT_err, "   Relative error:", uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))
    print("  Homogeneous solution error u(T)/norm_ratio:", uT2_err, "   Relative error:", uT2_err/numpy.linalg.norm(spi_uT_ho,ord=2))


    print("\n\nTests with Classical Subroutine (Homogeneous, w/ trotter)")
    ## Solve homogenous part
    exp_op, uT = class_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=2, trotterLH=True,verbose=0)
    uT = numpy.array(uT).reshape(-1)
    if numpy.linalg.norm(uT.imag,ord=2) < 1e-12:
        uT = uT.real
    uT_err = numpy.linalg.norm(uT - spi_uT_ho,ord=2)
    print("  Homogeneous solution error u(T)(trotter):", uT_err, "   Relative error:", uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))



    print("\n\nTests with Quantum Subroutine (Qiskit API)")
    quant_uT, lchs_circ_ho = pauli_lchs_tihs(L_coef_arr, L_pauli_arr, H_coef_arr, H_pauli_arr, 
                                            u0, T, beta, epsilon, trunc_multiplier=2, qiskit_api=True, verbose=1, debug=debug)

    quant_uT = numpy.array(quant_uT).reshape(-1)
    if numpy.linalg.norm(quant_uT.imag,ord=2) < 1e-12:
        quant_uT = quant_uT.real
    quant_uT_err = numpy.linalg.norm(quant_uT - spi_uT_ho,ord=2)
    norm_ratio = numpy.linalg.norm(quant_uT)/numpy.linalg.norm(spi_uT_ho)
    quant_uT2_err = numpy.linalg.norm(quant_uT/norm_ratio - spi_uT_ho,ord=2)
    print("  Homogeneous u(T)=           ", quant_uT, "  Norm=", numpy.linalg.norm(quant_uT,ord=2))
    print("  SciPy Sol   u(T)=           ", spi_uT_ho, "  Norm=", numpy.linalg.norm(spi_uT_ho,ord=2))
    print("  Homogeneous u(T)/norm_ratio=", quant_uT/norm_ratio, "  Norm=", numpy.linalg.norm(quant_uT,ord=2))
    print("  Homogeneous solution error u(T)           :", quant_uT_err, "   Relative error:", quant_uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))
    print("  Homogeneous solution error u(T)/norm_ratio:", quant_uT2_err, "   Relative error:", quant_uT2_err/numpy.linalg.norm(spi_uT_ho,ord=2))


