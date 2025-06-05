import numpy as np
import qutip as qt
import qiskit
import qiskit.quantum_info as qi
from utils_synth import qiskit_normal_order_switch

from scipy.linalg import expm
from scipy.optimize import minimize

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
            U_seq = qiskit_normal_order_switch( qi.Operator(U_circ).to_matrix() )
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


def var_recd(target_matrix, N_t=1, N_d=10, exp_coeff=None, use_circuit=True):
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
        N_t: Number of unitary sequences in the linear combination (default 15).
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
    
    # Optimize using a method such as Nelder-Mead.
    opt_result = minimize(cost_wrapper, init_params, method='BFGS',
                            options={'maxiter': 1000, 'disp': True})
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