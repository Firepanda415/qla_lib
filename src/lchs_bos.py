from ast import List
import numpy
import qiskit.quantum_info
import scipy
import qiskit
from typing import Callable
from lchs import *
from lcu import *
from oracle_synth import *
from utils_synth import *
from utils_synth import _ACCEPTED_GATES
import c2qa


##----------------------------------------------------------------------------------------------------------------


def bos_cart_decomp(coef_list:List):
    ## trivially use a*hat{n} + ib*hat{n}, 
    ## so L = a*hat{n}, and H = b*hat{n}
    return [coef_list[0]], [coef_list[1]]


def bos_gate(theta, num_qubits_per_qumode, return_circ=False):
    # trivial diagonal gate
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode)
    circ = c2qa.CVCircuit(qmr)
    circ.cv_r(theta, qmr[0])
    if return_circ:
        return circ
    return circ.to_gate()


# def bos_select_oracle(theta_array, num_qubits_per_qumode, qiskit_api:bool=False, debug:bool=False) -> qiskit.QuantumCircuit:
#     ## See (7.55) in https://arxiv.org/pdf/2201.08309
#     num_terms = len(theta_array)
#     num_qubits_control = nearest_num_qubit(num_terms)
#     num_qubits_op = num_qubits_per_qumode
#     bin_string = '{0:0'+str(num_qubits_control)+'b}'
#     ##
#     select_circ = qiskit.QuantumCircuit(num_qubits_control+num_qubits_op)
#     for i in range(num_terms):
#         ibin = bin_string.format(i)[::-1] ## NOTE: Qiskit uses reverse order
#         # bos
#         try:
#             control_u = bos_gate(theta_array[i], num_qubits_per_qumode, return_circ=True).control(num_qubits_control)
#         except:
#             print("Error in bos_gate")
#             print("theta_array[i] =", theta_array[i])
#             print("num_qubits_per_qumode =", num_qubits_per_qumode)
#             print("num_qubits_control =", num_qubits_control)
#             raise
#         # control_u = qiskit.circuit.library.UnitaryGate(bos_gate(theta_array[i], num_qubits_per_qumode)).control(num_qubits_control)
#         ## For 0-control
#         for q in range(len(ibin)):
#             qbit = ibin[q]
#             if qbit == '0':
#                 select_circ.x(q)
#         ## Apply the controlled-U gate
#         select_circ.append( control_u, list(range(num_qubits_control+num_qubits_op)) )
#         ## UNDO the X gate for 0-control
#         for q in range(len(ibin)):
#             qbit = ibin[q]
#             if qbit == '0':
#                 select_circ.x(q)
#     ##
#     select_circ.name = 'SELECT'
#     return select_circ ## NOTE: in qiskit order


# def bos_lcu_generator(coeff_array:list, theta_array:list, num_qubits_per_qumode, initial_state_circ=None,verbose:int=0, qiskit_api:bool=False, debug:bool=False) -> qiskit.QuantumCircuit:
#     '''
#     NOTE: Check example usage for big endian
#     NOTE: initial_state_circ in big endian (not qiskit one anyway)
#     Example usage in big endian:
#             rng = numpy.random.Generator(numpy.random.PCG64(726348874394184524479665820111))
#             scipy_uni = scipy.stats.unitary_group
#             scipy_uni.random_state = rng
#             ##
#             test_coefs =  rng.random(2**n)
#             test_coefs_normed = test_coefs/numpy.linalg.norm(test_coefs, ord=1)
#             test_unitaries = [scipy_uni.rvs(2**n) for _ in range(2**n)]
#             ##
#             correct_answer = numpy.zeros(test_unitaries[0].shape, dtype=complex)
#             for i in range(len(test_coefs_normed)):
#                 correct_answer += test_coefs_normed[i]*test_unitaries[i]
#             ##
#             LCU = lcu_generator(test_coefs, test_unitaries, verbose=1, qiskit_api=qiskit_api)
#             circ_mat = qiskit.quantum_info.Operator(LCU).data
#             lcu_sol = qiskit_normal_order_switch(circ_mat[:test_unitaries[0].shape[0],:test_unitaries[0].shape[1]]) 
#             ## need the endianness switch for each coordinates on submatrix
#             ## Only in this case, numpy.linalg.norm(correct_answer - lcu_sol, ord=2) gives no error
#     '''
#     ##
#     def vec_mag_angles(complex_vector:numpy.ndarray):
#         norm_vector = numpy.array(complex_vector)
#         for i in range(len(complex_vector)):
#             entry_norm = numpy.abs(complex_vector[i])
#             if entry_norm > 1e-12:
#                 norm_vector[i] = complex_vector[i]
#             else:
#                 norm_vector[i] = 0
#         return numpy.abs(complex_vector), numpy.angle(norm_vector)
#     coef_abs, coef_phase = vec_mag_angles(coeff_array)
#     theta_array = [numpy.exp(1j*coef_phase[i])*theta_array[i] for i in range(len(unitatheta_arrayry_array))] ## this is wrong

#     # coef_abs = coeff_array
#     ##
#     num_terms = len(coeff_array) #len(absorbed_unitaries)
#     num_qubits_control = nearest_num_qubit(num_terms)
#     # num_qubits_op = int(numpy.log2(absorbed_unitaries[0].shape[0]))
#     num_qubits_op = num_qubits_per_qumode
#     if verbose > 0:
#         print("  LCU-Oracle: num_qubits_control=", num_qubits_control, "num_qubits_op=", num_qubits_op)
#     ##
#     prep_circ = prep_oracle(coef_abs, qiskit_api=qiskit_api)
#     select_circ = bos_select_oracle(theta_array, num_qubits_per_qumode, qiskit_api=qiskit_api, debug=debug)
#     ##
#     lcu_circ = qiskit.QuantumCircuit(num_qubits_control+num_qubits_op)
#     if initial_state_circ:
#         lcu_circ.append(initial_state_circ, list(range(num_qubits_control+num_qubits_op))[num_qubits_control:])
#     ## Apply the preparation oracle
#     lcu_circ.append(prep_circ, list(range(num_qubits_control)))
#     ## Apply the selection oracle
#     lcu_circ.append(select_circ, list(range(num_qubits_control+num_qubits_op)))
#     ## Apply the preparation oracle dagger
#     lcu_circ.append(prep_circ.inverse(), list(range(num_qubits_control)))
#     return lcu_circ.reverse_bits() ## NOTE: not in qiskit order







# ## Homogenous Part of the Solution
# def bos_lchs_tihs(Acoef_list:List, num_qubits_per_qumode:int, u0:numpy.matrix, tT:float, beta:float, epsilon:float, 
#                     trunc_multiplier=2, trotterLH:bool=True,
#                     qiskit_api:bool=True, verbose:int=0, debug:bool=False) -> tuple[numpy.matrix,numpy.matrix]: 
#     '''
#     For bosons
#     Solve Homogeneous IVP du/dt = -Au(t) with u(0) = u0, where A = a hat{n} + ib hat{n} 
#     Input:
#     - Acoef_list: list, [a,b] for coefficients Only single element for now
#     - u0: numpy.matrix, initial condition
#     - tT: float, end time
#     - beta: float, 0 < beta < 1, parameter in the kernel function
#     - epsilon: float, error tolerance
#     - trunc_multiplier: float, multiplier for truncation range, just for higher accuracy. See (63) in [1]. Default is 2.
#     - trotterLH: bool, use Trotterization for exp(-i (kL+H)t) => exp(-i H t)exp(-i kL t) if true, so exp(i H t) is only applied once for all k
#     - verbose: bool, print out parameters
#     Output:
#     - res_mat: numpy matrix, the matrix representation of the solution operator (summation of all unitaties)
#     - uT: numpy matrix, the solution state at time tT, uT = res_mat*u0
#     '''
#     if not trotterLH:
#         raise NotImplementedError("Non-TrotterLH is not implemented for bosons")
#     from lcu import lcu_generator
#     from utils_synth import qiskit_normal_order_switch, qiskit_normal_order_switch_vec
#     from oracle_synth import synthu_qsd, stateprep_ucr
#     ##
#     L_coefs,H_coefs = bos_cart_decomp(Acoef_list)
#     ##
#     from c2qa.operators import CVOperators
#     cv_ops = CVOperators()
#     L = L_coefs[0]*cv_ops.get_N(2**num_qubits_per_qumode).todense()
#     h1 = step_size_h1(tT, L)  ## step size h1 for [-K, K], (65) in [1]
#     ##
#     K = trunc_multiplier*trunc_K(beta, epsilon, h1) ## integral range -K to K, (63) in [1] ## "2" here is just for higher accuracy since formula is big-O
#     Q = n_node_Q(beta, epsilon, K)  ## number of nodes in each subinterval [mh_1, (m+1)h_1] for m = -K/h1 to K/h1, (65) in [1]
#     ##
#     kh1 = int(K/h1)
#     M = int(2*kh1*Q) ## number of nodes in total, (65) in [1] ## just for checking
#     if verbose>0:
#         print("  Preset parameters T = ", tT, "beta = ", beta, "epsilon = ", epsilon)
#         print("  Truncation range [-K,K] K =", K)
#         print("  Step size h1 =", h1)
#         print("  Number of nodes in [mh_1, (m+1)h_1] Q =", Q)
#         print("  Total number of nodes M =", M)
#     ##
#     c_sum = 0 ## compute ||c||_1
#     coeffs = []
#     thetas = [] ## theta for exp(i b * L) = exp(i b*a*hat{n}) = exp(i theta hat{n}), i.e., theta = b*a
#     for munshit in range(2*kh1):
#         m = -kh1+munshit ## shift to start from -K/h1
#         kqms, wqs = gauss_quadrature(m*h1, h1, Q) ## Gaussian quadrature points and weights in [mh_1, (m+1)h_1]
#         ## Since using LCU, only compute coefficients and unitaries
#         for qi in range(Q):
#             cqm = wqs[qi]*gk(beta, kqms[qi])
#             c_sum += numpy.abs(cqm)
#             if trotterLH:
#                 theta = L_coefs[0]
#             else:
#                 theta =  L_coefs[0] + 1j*H_coefs[0] # TODO:?
#             ## Collect coeffs and unitaries for w_q*g(k_{q,m})*U(T, k_{q,m}) ## (61) in [1] (This should be quantum)
#             # if numpy.abs(numpy.angle(cqm)) > 1e-12:
#             #     cqm = cqm
#                 # cqm = cqm/numpy.exp(1j*numpy.angle(cqm)) ## absorb the phase to the unitaries
#                 # theta = numpy.exp(1j*numpy.angle(cqm))*theta
#             coeffs.append(cqm)
#             thetas.append(theta)
#     if verbose > 0:
#         print("  ||c||_1 =", c_sum)
        
#     ## Obtain the linear combination by LCU
#     lcu_circ = bos_lcu_generator(coeffs, thetas, num_qubits_per_qumode, initial_state_circ=None, verbose=verbose, qiskit_api=qiskit_api, debug=debug) ## NOTE: the return circuit is in qiskit order
#     if trotterLH:
#         if abs(H_coefs[0]) > 1e-12:
#             ## exp(i(A+B)) approx exp(iA/2) exp(iB) exp(iA/2), (4.104) in Nielsen and Chuang (10th anniversary edition)
#             exph_circ = bos_gate(H_coefs[0]/2, num_qubits_per_qumode, return_circ=True).reverse_bits()
#             lcu_circ.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=True, inplace=True)
#             lcu_circ.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=False, inplace=True)
#     if verbose > 0:
#         print("  Number of Qubits:", lcu_circ.num_qubits)


#     # print(Warning("Simulation is DISABLED for a quick test"))
#     circ_op = qiskit.quantum_info.Operator( lcu_circ ).data ## LCU has reversed qubits
#     sum_op = qiskit_normal_order_switch( circ_op[:2**num_qubits_per_qumode,:2**num_qubits_per_qumode] ) ## See lcu_generator use case example
#     ## Compute u0
#     u0_norm = u0/numpy.linalg.norm(u0,ord=2)
#     uT = sum_op.dot(u0_norm)

#     # uT = qiskit.quantum_info.Statevector(lcu_circ).data[:H.shape[0]]
#     ##
#     return uT, lcu_circ




#----------------------------------------------------------------------------------------------------------------

def bos_prolong_select_oracle(theta_array, num_qubits_per_qumode, debug:bool=False) -> qiskit.QuantumCircuit:
    ## See (7.55) in https://arxiv.org/pdf/2201.08309
    num_terms = len(theta_array)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = num_qubits_per_qumode
    bin_string = '{0:0'+str(num_qubits_control)+'b}'
    ##
    qbr = qiskit.QuantumRegister(num_qubits_control)
    qbr_ancilla = qiskit.QuantumRegister(1)
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode)
    select_circ = c2qa.CVCircuit(qbr, qbr_ancilla, qmr)
    for i in range(num_terms):
        ibin = bin_string.format(i)[::-1] ## NOTE: Qiskit uses reverse order
        ## For 0-control
        for q in range(len(ibin)):
            qbit = ibin[q]
            if qbit == '0':
                select_circ.x(q)
        ## multiple-control X targeting ancilla qubit
        select_circ.mcx(qbr, qbr_ancilla[0])
        # Sanwitch the the controlled-U bosonic gate
        try:
            select_circ.cv_c_r(theta_array[i], qmr[0], qbr_ancilla[0])
        except:
            print("Error in bos_gate")
            print("theta_array[i] =", theta_array[i])
            print("num_qubits_per_qumode =", num_qubits_per_qumode)
            print("num_qubits_control =", num_qubits_control)
            raise
        ## Sanwitch back the multiple-control X targeting ancilla qubit
        select_circ.mcx(qbr, qbr_ancilla[0])
        ## UNDO the X gate for 0-control
        for q in range(len(ibin)):
            qbit = ibin[q]
            if qbit == '0':
                select_circ.x(q)
    ##
    select_circ.name = 'SELECT'
    return select_circ ## NOTE: in qiskit order


def bos_prolong_lcu(coeff_array:numpy.array, theta_array:numpy.array, num_qubits_per_qumode, initial_state_circ=None,verbose:int=0, qiskit_api:bool=False, debug:bool=False) -> qiskit.QuantumCircuit:
    num_terms = len(coeff_array) #len(absorbed_unitaries)
    num_qubits_control = nearest_num_qubit(num_terms)
    num_qubits_op = num_qubits_per_qumode
    ## separate the real and imaginary parts
    real_coeffs = coeff_array.real/numpy.linalg.norm(coeff_array.real,ord=1)
    imag_coeffs = coeff_array.imag/numpy.linalg.norm(coeff_array.imag,ord=1)
    print("LCU bos debug")
    print(real_coeffs)
    print(imag_coeffs)
    ## prep circuit for real and imag parts
    real_prep_circ = prep_oracle(real_coeffs, qiskit_api=qiskit_api)
    imag_prep_circ = prep_oracle(imag_coeffs, qiskit_api=qiskit_api)
    ## select circuit for real and imag parts
    real_select_circ = bos_prolong_select_oracle(theta_array, num_qubits_per_qumode, debug=debug)
    imag_select_circ = bos_prolong_select_oracle(theta_array, num_qubits_per_qumode, debug=debug)

    ## prolonged LCU circuit
    ## use 1 extra ancilla qubit to take the result of multiple-control, 
    ## sanwitch the single-control bosonic gate with multiple-control qubit gate
    qbr = qiskit.QuantumRegister(num_qubits_control)
    qbr_ancilla = qiskit.QuantumRegister(1)
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode)
    ## Real part
    lcu_circ_real = c2qa.CVCircuit(qbr, qbr_ancilla, qmr)
    if initial_state_circ:
        lcu_circ_real.append(initial_state_circ.copy(), list(range(num_qubits_control+num_qubits_op+1))[num_qubits_control:])
    ## Apply the preparation oracle
    lcu_circ_real.append(real_prep_circ, list(range(num_qubits_control)))
    ## Apply the selection oracle
    lcu_circ_real.append(real_select_circ, list(range(num_qubits_control+num_qubits_op+1)))
    ## Apply the preparation oracle dagger
    lcu_circ_real.append(real_prep_circ.inverse(), list(range(num_qubits_control)))


    ## Imaginary part
    qbr_imag = qiskit.QuantumRegister(num_qubits_control)
    qbr_ancilla_imag = qiskit.QuantumRegister(1)
    qmr_imag = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode)
    lcu_circ_imag = c2qa.CVCircuit(qbr_imag, qbr_ancilla_imag, qmr_imag)
    if initial_state_circ:
        lcu_circ_imag.append(initial_state_circ.copy(), list(range(num_qubits_control+num_qubits_op+1))[num_qubits_control:])
    ## Apply the preparation oracle
    lcu_circ_imag.append(imag_prep_circ, list(range(num_qubits_control)))
    ## Apply the selection oracle
    lcu_circ_imag.append(imag_select_circ, list(range(num_qubits_control+num_qubits_op+1)))
    ## Apply the preparation oracle dagger
    lcu_circ_imag.append(imag_prep_circ.inverse(), list(range(num_qubits_control)))

    return lcu_circ_real.reverse_bits(), lcu_circ_imag.reverse_bits() ## NOTE: i.e., not in qiskit order after reverse_bits



## Homogenous Part of the Solution
def lchs_coeffs_unitaries(Acoef_list:List, num_qubits_per_qumode, u0:numpy.matrix, tT:float, beta:float, epsilon:float, 
                    trunc_multiplier=2, trotterLH:bool=True,
                    verbose:int=0, debug:bool=False): 
    if not trotterLH:
        raise NotImplementedError("Non-TrotterLH is not implemented for bosons")
    from lcu import lcu_generator
    from utils_synth import qiskit_normal_order_switch, qiskit_normal_order_switch_vec
    from oracle_synth import synthu_qsd, stateprep_ucr
    ##
    L_coefs,H_coefs = bos_cart_decomp(Acoef_list)
    ##
    from c2qa.operators import CVOperators
    cv_ops = CVOperators()
    L = L_coefs[0]*cv_ops.get_N(2**num_qubits_per_qumode).todense()
    h1 = step_size_h1(tT, L)  ## step size h1 for [-K, K], (65) in [1]
    ##
    K = trunc_multiplier*trunc_K(beta, epsilon, h1) ## integral range -K to K, (63) in [1] ## "2" here is just for higher accuracy since formula is big-O
    Q = n_node_Q(beta, epsilon, K)  ## number of nodes in each subinterval [mh_1, (m+1)h_1] for m = -K/h1 to K/h1, (65) in [1]
    ##
    kh1 = int(K/h1)
    M = int(2*kh1*Q) ## number of nodes in total, (65) in [1] ## just for checking
    if verbose>0:
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
    coeffs = []
    thetas = [] ## theta for exp(i b * L) = exp(i b*a*hat{n}) = exp(i theta hat{n}), i.e., theta = b*a
    for munshit in range(2*kh1):
        m = -kh1+munshit ## shift to start from -K/h1
        kqms, wqs = gauss_quadrature(m*h1, h1, Q) ## Gaussian quadrature points and weights in [mh_1, (m+1)h_1]
        ## Since using LCU, only compute coefficients and unitaries
        for qi in range(Q):
            cqm = wqs[qi]*gk(beta, kqms[qi])
            c_sum += numpy.abs(cqm)
            if trotterLH:
                theta = -L_coefs[0]*kqms[qi]*tT
            else:
                raise NotImplementedError("Non-TrotterLH is not implemented for bosons")
            coeffs.append(cqm)
            thetas.append(theta)
    if verbose > 0:
        print("  ||c||_1 =", c_sum)
    return numpy.array(coeffs), numpy.array(thetas)

def bos_pro_long_lchs_tihs(Acoef_list:List, num_qubits_per_qumode:int, u0:numpy.matrix, tT:float, beta:float, epsilon:float, 
                    trunc_multiplier=2, trotterLH:bool=True,
                    qiskit_api:bool=True, verbose:int=0, debug:bool=False): 
    L_coefs,H_coefs = bos_cart_decomp(Acoef_list)
    coeffs, thetas = lchs_coeffs_unitaries(Acoef_list, num_qubits_per_qumode, u0, tT, beta, epsilon, 
                    trunc_multiplier=trunc_multiplier, trotterLH=trotterLH, verbose=verbose, debug=debug)
    lcu_circ_real, lcu_circ_imag = bos_prolong_lcu(coeffs, thetas, num_qubits_per_qumode, initial_state_circ=None, verbose=verbose, qiskit_api=qiskit_api, debug=debug)
    if trotterLH:
        if abs(H_coefs[0]) > 1e-12:
            ## exp(i(A+B)) approx exp(iA/2) exp(iB) exp(iA/2), (4.104) in Nielsen and Chuang (10th anniversary edition)
            exph_circ = bos_gate(tT*H_coefs[0]/2, num_qubits_per_qumode, return_circ=True).reverse_bits()
            lcu_circ_real.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=True, inplace=True)
            lcu_circ_real.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=False, inplace=True)
            lcu_circ_imag.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=True, inplace=True)
            lcu_circ_imag.compose(exph_circ, qubits=range(exph_circ.num_qubits), front=False, inplace=True)
    if verbose > 0:
        print("  Number of Qubits:", lcu_circ_real.num_qubits, ", ",lcu_circ_imag.num_qubits)
    # print(Warning("Simulation is DISABLED for a quick test"))
    circ_op_real = qiskit.quantum_info.Operator( lcu_circ_real ).data ## LCU has reversed qubits
    circ_op_imag = qiskit.quantum_info.Operator( lcu_circ_imag ).data ## LCU has reversed qubits
    sum_op_real = qiskit_normal_order_switch( circ_op_real[:2**num_qubits_per_qumode,:2**num_qubits_per_qumode] ) ## See lcu_generator use case example
    sum_op_imag = qiskit_normal_order_switch( circ_op_imag[:2**num_qubits_per_qumode,:2**num_qubits_per_qumode] ) ## See lcu_generator use case example
    ## Compute u0
    u0_norm = u0/numpy.linalg.norm(u0,ord=2)
    uT_real = sum_op_real.dot(u0_norm)
    uT_imag = sum_op_imag.dot(u0_norm)
    return uT_real, uT_imag, lcu_circ_real, lcu_circ_imag, circ_op_real, circ_op_imag, coeffs, thetas