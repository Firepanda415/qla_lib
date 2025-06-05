import numpy as np
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

def simulate_unitary_sequence(seq_params, ansatz_depth, b, b_dag, L):
    """
    Simulate a single unitary sequence (one rotation+ECD expansion term for a Pauli word).
    
    Parameters:
        seq_params: numpy array of shape (ansatz_depth, 3) containing the parameters for each block.
                    Each row is [beta, theta, phi].
        ansatz_depth: number of blocks (depth) in the sequence.
        b, b_dag: bosonic annihilation/creation operators.
        L: dimension of the bosonic (Fock) space.
        
    Returns:
        U_seq: A numpy array of shape (L, L) representing the effective operator 
               ⟨0|U_seq|0⟩ on the bosonic mode (projected onto the qubit |0> state).
               
    The simulation is done by applying the sequence to each Fock basis state |m> (m = 0,...,L-1)
    where the ancilla qubit is initially in |0>.
    """
    U_seq = np.identity(L*2, dtype=complex)
    I_temp = np.identity(L, dtype=complex)
    for d in range(ansatz_depth):
        beta, theta, phi = seq_params[d]
        r_temp = rotation_gate(theta, phi) ## Eq. (10)
        ecd_temp = ecd_gate(beta, b, b_dag)  ## Eq. (11)
        temp_gate = np.kron(r_temp, I_temp)
        unitary_temp = ecd_temp.dot(temp_gate) ## Eq. (9b)
        U_seq = U_seq.dot(unitary_temp) ## Eq. (9a)
    return U_seq

def cost_function(params, target_W, ansatz_depth, b, b_dag, L):
    """
    Compute the cost (Frobenius norm squared) between the effective operator and the target operator.
    
    Parameters:
        params: 1D numpy array containing the flattened gate parameters.
                It is reshaped to (ansatz_depth, 3), where each row is [beta, theta, phi].
        target_W: Target operator matrix of shape (L, L) (constructed from a Pauli word).
        ansatz_depth: depth (number of Rotation+ECD blocks) in the sequence.
        b, b_dag: bosonic operators.
        L: Fock space cutoff dimension.
        
    Returns:
        cost: A float representing the squared Frobenius norm difference between
              the effective operator U_seq and target_W.
              
    Here, the effective operator is defined as U_seq = ⟨0|U|0⟩ for the optimized sequence.
    """
    gate_params = params.reshape((ansatz_depth, 3))
    U_seq = simulate_unitary_sequence(gate_params, ansatz_depth, b, b_dag, L)
    diff = U_seq - np.kron(_I, target_W)
    return np.linalg.norm(diff, ord=2)**2 ## 

# ---------------------------------------------------------------------------
# Main function: qumode_lchs_tihs()
# ---------------------------------------------------------------------------
def qumode_lchs_tihs(pauli_terms, coefficients, ansatz_depth=10):
    """
    Maps a qubit Hamiltonian to a qubit-qumode operator using an ECD-based ansatz.
    
    The Hamiltonian is specified as a list of Pauli strings (pauli_terms) and their
    corresponding coefficients. Each Pauli word (a term in the Hamiltonian) is approximated
    by a single unitary sequence composed of 'ansatz_depth' Rotation+ECD blocks.
    
    Returns:
        A list of dictionaries (one per Pauli word) containing:
          'pauli': the Pauli string,
          'coeff': its coefficient,
          'betas': optimized beta parameters (array of shape (ansatz_depth,)),
          'thetas': optimized theta parameters (array of shape (ansatz_depth,)),
          'phis': optimized phi parameters (array of shape (ansatz_depth,)).
    """
    # Precompute bosonic operators.
    n_qubits = len(pauli_terms[0])
    L = 2 ** n_qubits  # Fock cutoff determined by number of qubits.
    b = np.zeros((L, L), dtype=complex)
    for m in range(1, L):
        b[m - 1, m] = np.sqrt(m)
    b_dag = b.T

    # Define Pauli matrices for constructing the target operator.
    pauli_map = {'I': _I, 'X': _X, 'Y': _Y, 'Z': _Z}
    
    results = []
    # For each Pauli word in the Hamiltonian:
    for term, coeff in zip(pauli_terms, coefficients):
        # Construct the target operator W for this Pauli term.
        W = 1
        for p in term:
            W = np.kron(W, pauli_map[p])
        W = coeff * W
        W = expm(1j*W)
        # W = W.reshape(L, L)
        
        # Define a lambda wrapper for the cost function.
        def cost_wrapper(params):
            return cost_function(params, W, ansatz_depth, b, b_dag, L)
        
        # Initial guess: small random gate parameters (no lambda weight since we use a single term).
        init_gate_params = 0.1 * np.random.randn(ansatz_depth, 3)
        init_params = init_gate_params.flatten()
        
        # Optimize using an algorithm (e.g., Nelder-Mead).
        opt_result = minimize(cost_wrapper, init_params, method='Nelder-Mead',
                              options={'maxiter': 10000, 'disp': False})
        opt_params = opt_result.x.reshape((ansatz_depth, 3))
        
        results.append({
            'pauli': term,
            'coeff': coeff,
            'W':W,
            'full_params':opt_params,
            'betas': opt_params[:, 0],
            'thetas': opt_params[:, 1],
            'phis': opt_params[:, 2]
        })
    return results

# ---------------------------------------------------------------------------
# Example usage:
# ---------------------------------------------------------------------------
# if __name__ == '__main__':
#     # Example: a two-qubit Hamiltonian composed of two Pauli words.
#     pauli_terms = ["ZI", "IZ"]
#     coefficients = [0.5, -0.5]
    
#     # Optimize mapping parameters for each Pauli word.
#     mapping_params = qumode_lchs_tihs(pauli_terms, coefficients, ansatz_depth=10)
    
#     # You can independently call the helper functions.
#     sample_state_q0 = np.array([1, 0, 0, 0], dtype=complex)  # For L=4 (2 qubits)
#     sample_state_q1 = np.zeros(4, dtype=complex)
#     new_q0, new_q1 = apply_rotation(sample_state_q0, sample_state_q1, np.pi/4, np.pi/3)
    
#     print("Optimized mapping parameters:")
#     for term_params in mapping_params:
#         print(term_params)
    
#     print("\nExample rotation output (first few amplitudes):")
#     print(new_q0)
