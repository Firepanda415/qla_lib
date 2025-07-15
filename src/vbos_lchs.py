import numpy
import scipy
import numpy as np
import qutip as qt
import qiskit
import qiskit.quantum_info as qi
from utils_synth import qiskit_normal_order_switch

from scipy.linalg import expm
from scipy.optimize import minimize

import c2qa

_KET0 = np.array([[1], [0]])
_KET1 = np.array([[0], [1]])
_K10B = _KET1 @ _KET0.T
_K01B = _KET0 @ _KET1.T
_K00B = _KET0 @ _KET0.T
_K11B = _KET1 @ _KET1.T

_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def rotation_gate(theta, phi):
    """
    Eq. (10)
    """
    # c, s = np.cos(theta / 2), np.sin(theta / 2)
    return expm( -1j * theta * 0.5 * (np.cos(phi) * _X + np.sin(phi) * _Y) )

def d_gate(beta, b, b_dag):
    """
    Eq. (12)
    """
    return expm(beta * b_dag - beta.conjugate()*b)

def ecd_gate(beta, b, b_dag):
    """
    Eq. (11)
    """
    return np.kron(_K10B, d_gate(0.5*beta, b, b_dag)) + np.kron(_K01B, d_gate(-0.5*beta, b, b_dag))

def simulate_unitary_sequence(seq_params, N_d, b, b_dag, L):
    """
    Simulate a single unitary sequence (one term in the linear combination).
    
    Parameters:
        seq_params: numpy array of shape (N_d, 3) containing the parameters for each block.
                    Each row is [beta, theta, phi].
        N_d: number of blocks (depth) in the sequence.
        b, b_dag: bosonic annihilation/creation operators.
        L: dimension of the bosonic (Fock) space.
        
    Returns:
        U_seq: A numpy array of shape (L, L) representing the effective operator 
               ⟨0|U_seq|0⟩ on the bosonic mode when the ancilla qubit is in |0>.
               
    The simulation is done by applying, for each input basis state |m> (with m = 0,...,L-1),
    the sequence of rotation and ECD gates starting from |0>_Q ⊗ |m>_R.
    """
    U_seq = np.identity(L*2, dtype=complex)
    I_temp = np.identity(L, dtype=complex)
    for d in range(N_d):
        beta_mag, beta_ang, theta, phi = seq_params[d]
        beta = beta_mag * np.exp(1j * beta_ang)
        r_temp = rotation_gate(theta, phi) ## Eq. (10)
        ecd_temp = ecd_gate(beta, b, b_dag)  ## Eq. (11)
        temp_gate = np.kron(r_temp, I_temp)
        unitary_temp = ecd_temp.dot(temp_gate) ## Eq. (9b)
        U_seq = U_seq.dot(unitary_temp) ## Eq. (9a)
    return U_seq


def qbos_unitary_sequence(seq_params, N_d, L):
    """
    Simulate a single unitary sequence (one term in the linear combination).

    Note that the definition of gate in the paper and in bosonic qiskit are different
    ECD in the paper using bosonic qiskit is: (X otimes I) ECD(beta/2, L) where I is the same size as L
    NOTE: ECD matrix has size L*2 (1 extra qubit for controlling)
    
    Parameters:
        seq_params: numpy array of shape (N_d, 3) containing the parameters for each block.
                    Each row is [beta, theta, phi].
        N_d: number of blocks (depth) in the sequence.
        b, b_dag: bosonic annihilation/creation operators.
        L: dimension of the bosonic (Fock) space.
        
    Returns:
        U_seq: A numpy array of shape (L, L) representing the effective operator 
               ⟨0|U_seq|0⟩ on the bosonic mode when the ancilla qubit is in |0>.
               
    The simulation is done by applying, for each input basis state |m> (with m = 0,...,L-1),
    the sequence of rotation and ECD gates starting from |0>_Q ⊗ |m>_R.
    """
    import c2qa

    qr = qiskit.QuantumRegister(size=1)
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int( np.log2(L) ))
    circuit = c2qa.CVCircuit(qr, qmr)

    for d in range(N_d):
        beta_mag, beta_ang, theta, phi = seq_params[d]
        beta = beta_mag * np.exp(1j * beta_ang)
        ## Rotation
        circuit.rz(-phi, qr[0])
        circuit.rx(theta, qr[0])
        circuit.rz(phi, qr[0])
        ## ECD
        circuit.cv_ecd(beta/2, qmr[0], qr[0], duration=100, unit="ns")
        circuit.x(qr)

    return circuit



def tran_val_qubit_cavity(op, n, m):
    """
    Compute <0, n| O |0, m>.

    Arguments:
    op -- Operator matrix
    n -- Fock level
    """
    # Check
    L = op.shape[0] // 2
    if n > L:
        raise ValueError("n > L.")
    if m > L:
        raise ValueError("m > L.")

    # |0, n> and |0, m>
    state1 = qt.tensor(qt.basis(2, 0), qt.basis(L, n)).full()
    state2 = qt.tensor(qt.basis(2, 0), qt.basis(L, m)).full()

    # <0, n| O |0, m>
    t1 = np.matmul(op, state2)
    ov = np.dot(np.conj(state1).T, t1)

    return np.squeeze(ov)



def cost_function(params, target_W, N_t, N_d, b, b_dag, L, use_circuit = True):
    """
    Compute the cost (Frobenius norm squared) between the effective operator
    and the target operator.
    
    Parameters:
        params: 1D numpy array containing the flattened parameters.
                It consists of:
                  - The first N_t entries are the weight coefficients (lambdas).
                  - The remaining entries are gate parameters for each term,
                    reshaped to (N_t, N_d, 3) where each row is [beta, theta, phi].
        target_W: Target operator matrix of shape (L, L) (from the Pauli term).
        N_t: Number of sequences (terms) in the linear combination.
        N_d: Depth (number of blocks) per sequence.
        b, b_dag: Bosonic operators.
        L: Fock space cutoff dimension.
        
    Returns:
        cost: A float representing the squared Frobenius norm difference between
              the effective operator and target_W.
              
    The effective operator is defined as:
      U_total = sum_{j=1}^{N_t} lambda_j * U_seq^(j)
    computed on the qubit-0 subspace.
    """
    # lambdas = params[:N_t]
    gate_params = params[N_t:].reshape((N_t, N_d, 4))
    U_total = np.zeros((L*2, L*2), dtype=complex)
    IW = np.kron(_I, target_W)

    # Sum contributions from each term
    for j in range(N_t):
        if use_circuit:
            U_circ = qbos_unitary_sequence(gate_params[j], N_d, L)
            # U_seq = qiskit_normal_order_switch( qi.Operator(U_circ).to_matrix() )
            U_seq = qi.Operator(U_circ).to_matrix()
        else:
            U_seq = simulate_unitary_sequence(gate_params[j], N_d, b, b_dag, L)
        # U_total += lambdas[j] * U_seq
        U_total += U_seq
    
    ov = 0.0
    for j in range(L):
        for k in range(L):
            t0 = tran_val_qubit_cavity(IW, j, k)
            t1 = tran_val_qubit_cavity(U_total, j, k)
            ov += np.abs( t0 - t1 )**2
    return ov/(L**2)


def var_recd(target_matrix, N_t=1, N_d=10, exp_coeff=None, use_circuit=True, verbose=1):
    """
    Maps a qubit Hamiltonian (given by a list of Pauli strings and coefficients)
    to a qubit-qumode operator using an ECD-based ansatz.
    
    This function optimizes, for each Pauli term, the parameters in a linear combination
    of ECD-rotation unitary sequences so that the effective operator on the qumode approximates 
    the target Pauli operator (scaled by its coefficient). The optimization follows Eq. (15) 
    of the paper.
    
    Parameters:
        pauli_terms: list of strings (e.g., ["XIZ", "YZX", ...]) defining the Pauli terms.
        coefficients: list of floats giving the coefficient for each Pauli term.
        N_t: Number of unitary sequences in the linear combination (default 1).
        N_d: Depth (number of rotation+ECD blocks per sequence, default 10).
    
    Returns:
        List of dictionaries, one per Pauli term, each containing:
          'pauli': the Pauli string,
          'coeff': its coefficient,
          'lambdas': optimized weight coefficients (array of length N_t),
          'betas': optimized beta parameters (shape (N_t, N_d)),
          'thetas': optimized theta parameters (shape (N_t, N_d)),
          'phis': optimized phi parameters (shape (N_t, N_d)).
    """
    # Precompute bosonic operators for the Fock space.
    n_qubits = int(np.log2(target_matrix.shape[0]))
    L = 2 ** n_qubits  # Fock cutoff dimension.
    b = np.zeros((L, L), dtype=complex)
    for m in range(1, L):
        b[m - 1, m] = np.sqrt(m)
    b_dag = b.T

    # Process each Pauli term
    if exp_coeff is None:
        W = target_matrix
    else:
        W = expm(exp_coeff*target_matrix)
    
    
    # Define a lambda wrapper for the cost function with fixed parameters.
    def cost_wrapper(params):
        return cost_function(params, W, N_t, N_d, b, b_dag, L, use_circuit=use_circuit)
    
    # Initial guess: uniform weights and small random gate parameters.
    init_lambdas = np.full(N_t, 1.0 / N_t, dtype=float)
    init_gate_params = np.zeros((N_t, N_d, 4), dtype=float)
    init_gate_params[:, :, 0] = np.random.uniform(0,3, size=(N_t, N_d))   # β mag
    init_gate_params[:, :, 1] = np.random.uniform(0, np.pi, size=(N_t, N_d))   # β angle
    init_gate_params[:, :, 2] = np.random.uniform(0, np.pi, size=(N_t, N_d))   # θ
    init_gate_params[:, :, 3] = np.random.uniform(0, np.pi, size=(N_t, N_d))   # φ
    init_params = np.concatenate([init_lambdas, init_gate_params.flatten()])
    
    # Optimize
    opt_result = minimize(cost_wrapper, init_params, method='BFGS',
                            options={'maxiter': 1000, 'disp': verbose>0})
    # print(opt_result)
    opt_params = opt_result.x
    opt_lambdas = opt_params[:N_t]
    opt_gate_params = opt_params[N_t:].reshape((N_t, N_d, 4))
    
    res = {
        'W':W,
        'fun':opt_result.fun,
        'full_params':opt_params,
        'lambdas': opt_lambdas,
        'betas_mag': opt_gate_params[:, :, 0],
        'betas_ang': opt_gate_params[:, :, 1],
        'thetas': opt_gate_params[:, :, 2],
        'phis': opt_gate_params[:, :, 3],
    }
    if use_circuit:
        res['circuit'] = qbos_unitary_sequence(opt_gate_params[0], N_d, L) ## WARNING: assume N_t = 1
    return res


#####################################################

## In Little endian
def bos_lcu_qiskit(coeff_array:list, unitary_array: list[numpy.ndarray], verbose:int=0) -> qiskit.QuantumCircuit:
    '''
    NOTE: this is in little endian (qiskit order)
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
    def vec_mag_angles(complex_vector:numpy.ndarray):
        norm_vector = numpy.array(complex_vector)
        for i in range(len(complex_vector)):
            entry_norm = numpy.abs(complex_vector[i])
            if entry_norm > 1e-12:
                norm_vector[i] = complex_vector[i]
            else:
                norm_vector[i] = 0
        return numpy.abs(complex_vector), numpy.angle(norm_vector)
    ## Check if all coefficients are non-negative
    if numpy.allclose(numpy.abs(coeff_array), coeff_array, rtol=1e-12, atol=1e-12):
        coef_abs = coeff_array.real
        absorbed_unitaries = unitary_array
    else:
        ## Absorb the phase into the unitaries
        coef_abs, coef_phase = vec_mag_angles(coeff_array)
        absorbed_unitaries = [numpy.exp(1j*coef_phase[i])*unitary_array[i] for i in range(len(unitary_array))]
    ##
    num_terms = len(absorbed_unitaries)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = int(numpy.log2(absorbed_unitaries[0].shape[0]))
    if verbose > 0:
        print("  LCU-Oracle: num_qubits_control=", num_qubits_control, "num_qubits_op=", num_qubits_op)
    prep_circ = bos_prep_oracle(coef_abs)
    select_circ = bos_select_oracle(absorbed_unitaries)
    ##
    lcu_circ = select_circ.copy(name='LCU')
    lcu_circ.compose(prep_circ, qubits=range(num_qubits_control), front=True, inplace=True)
    lcu_circ.compose(prep_circ.inverse(), qubits=range(num_qubits_control), front=False, inplace=True)

    return lcu_circ, coef_abs, absorbed_unitaries, numpy.sum(coef_abs)




def nearest_num_qubit(x):
    return int(numpy.ceil(numpy.log2(x)))

## In Little endian
def bos_prep_oracle(coeff_array: list) -> numpy.array:
    '''
    LCU Oracle for PREPARE for T = sum_i=0^{K-1} a_i U_i
    Synthesis unitary V such that
    V|0000...0> = 1/sqrt(||a||_1) sum_i=0^{K-1} sqrt(|a_i|)|i> 
    '''
    ## Make the length of coeff_array to be 2^n
    num_terms = len(coeff_array)
    num_qubits = nearest_num_qubit(num_terms)
    l1norm = numpy.linalg.norm(coeff_array, ord=1)
    ## Check if all coefficients are non-negative
    if not numpy.allclose(numpy.abs(coeff_array), coeff_array, rtol=1e-12, atol=1e-12):
        raise ValueError("All coefficients should be non-negative, but we have", coeff_array)
    ##
    coeff_array_normedsqrt = numpy.sqrt(numpy.abs(coeff_array)/l1norm)
    full_coeffs = [0]*(2**num_qubits)
    full_coeffs[:num_terms] = coeff_array_normedsqrt
    ##

    qis_prep_isometry = qiskit.circuit.library.StatePreparation(full_coeffs)
    qis_prep_isometry.name = "PREP"
    return qis_prep_isometry

## In Little endian
def bos_select_oracle(unitary_array: list[numpy.ndarray]) -> qiskit.QuantumCircuit:
    num_terms = len(unitary_array)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = int(numpy.log2(unitary_array[0].shape[0]))
    bin_string = '{0:0'+str(num_qubits_control)+'b}'
    ##
    # select_circ = qiskit.QuantumCircuit(num_qubits_control+num_qubits_op)
    qr = qiskit.QuantumRegister(size=1+num_qubits_control)
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_op)
    select_circ = c2qa.CVCircuit(qr, qmr)

    for i in range(num_terms):
        print(f"    Optimization process {i+1}/{num_terms}", end='\r')
        ibin = bin_string.format(i)
        control_u = var_recd(unitary_array[i], N_t=1, N_d=5, use_circuit=True, verbose=0)['circuit'].control(num_qubits_control)

        ## For 0-control
        for q in range(len(ibin)):
            qbit = ibin[q]
            if qbit == '0':
                select_circ.x(q)
        ## Apply the controlled-U gate
        select_circ.append( control_u, list(range(num_qubits_control+num_qubits_op+1)) )
        ## UNDO the X gate for 0-control
        for q in range(len(ibin)):
            qbit = ibin[q]
            if qbit == '0':
                select_circ.x(q)
    ##
    select_circ.name = 'SELECT'
    return select_circ




####################################################

from lchs import cart_decomp, step_size_h1, trunc_K, n_node_Q, cbeta, gauss_quadrature, utk, utk_L, gk


## Homogenous Part of the Solution
def bos_var_lchs_tihs(A:numpy.matrix, u0:numpy.matrix, tT:float, beta:float, epsilon:float, 
                    trunc_multiplier=2, trotterLH:bool=False,
                    verbose:int=0): 
    '''
    Solve Homogeneous IVP du/dt = -Au(t) with u(0) = u0
    Input:
    - A: numpy matrix, coefficient matrix, NOTE the minus sign in the equation
    - u0: numpy.matrix, initial condition
    - tT: float, end time
    - beta: float, 0 < beta < 1, parameter in the kernel function
    - epsilon: float, error tolerance
    - trunc_multiplier: float, multiplier for truncation range, just for higher accuracy. See (63) in [1]. Default is 2.
    - trotterLH: bool, use Trotterization for exp(-i (kL+H)t) => exp(-i H t)exp(-i kL t) if true, so exp(i H t) is only applied once for all k
    - verbose: bool, print out parameters
    Output:
    - res_mat: numpy matrix, the matrix representation of the solution operator (summation of all unitaties)
    - uT: numpy matrix, the solution state at time tT, uT = res_mat*u0
    '''
    from utils_synth import qiskit_normal_order_switch
    ##
    u0_norm = u0/numpy.linalg.norm(u0,ord=2)
    L,H = cart_decomp(A)
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
    unitaries_unrot = []
    coeffs = []
    unitaries = []
    for munshit in range(2*kh1):
        m = -kh1+munshit ## shift to start from -K/h1
        kqms, wqs = gauss_quadrature(m*h1, h1, Q) ## Gaussian quadrature points and weights in [mh_1, (m+1)h_1]
        ## Since using LCU, only compute coefficients and unitaries
        for qi in range(Q):
            cqm = wqs[qi]*gk(beta, kqms[qi])
            c_sum += numpy.abs(cqm)
            if trotterLH:
                umat = utk_L(tT, kqms[qi], L)
            else:
                umat = utk(tT, kqms[qi], L, H)
            ## Collect coeffs and unitaries for w_q*g(k_{q,m})*U(T, k_{q,m}) ## (61) in [1] (This should be quantum)
            coeffs_unrot.append(cqm)
            unitaries_unrot.append(umat)
    if verbose > 0:
        print("  ||c||_1 =", c_sum)
        
    ## Obtain the linear combination by LCU
    # lcu_circ = lcu_generator(coeffs, unitaries, initial_state_circ=state_prep_circ, qiskit_api=qiskit_api) ## NOTE: the return circuit is in qiskit order
    lcu_circ, coeffs, unitaries, coeffs_1norm = bos_lcu_qiskit(coeffs_unrot, unitaries_unrot, verbose=verbose) ## NOTE: the return circuit is NOT in qiskit order
    num_control_qubits = nearest_num_qubit(len(coeffs))
    if trotterLH:
        exph_circ = qiskit.QuantumCircuit(int(numpy.log2(H.shape[0])))
        exph_circ = var_recd(H, N_t=1, N_d=5, exp_coeff=-1j*0.5*tT, use_circuit=True, verbose=0)['circuit']
        qubit_range = num_control_qubits + np.array(list(range(exph_circ.num_qubits)))
        lcu_circ.compose(exph_circ, qubits=qubit_range, front=True, inplace=True)
        lcu_circ.compose(exph_circ, qubits=qubit_range, front=False, inplace=True)
    if verbose > 0:
        print("  Number of Qubits:", lcu_circ.num_qubits)

    # print(Warning("Simulation is DISABLED for a quick test"))
    print(">> State preparation for initial condition is DISABLED <<")
    circ_op = qiskit.quantum_info.Operator( lcu_circ ).data
    circ_op_rev = qiskit.quantum_info.Operator( lcu_circ.reverse_bits() ).data
    sum_op = coeffs_1norm * qiskit_normal_order_switch( circ_op_rev[:L.shape[0]*2,:L.shape[1]*2] )[:L.shape[0],:L.shape[1]] ## get ansatz matrix frist, then reverse the endianness, then get the submatrix for qumodes
    ## Compute u0
    uT = sum_op.dot(u0_norm)

    return uT, lcu_circ, circ_op, coeffs, unitaries, coeffs_unrot, unitaries_unrot