## This is prototype code for NWQSim https://github.com/pnnl/NWQ-Sim
## Author: Muqing Zheng

## TODO: add state preparation for u0 into quant_lchs_tihs()


## References:
## [1] Quantum algorithm for linear non-unitary dynamics with near-optimal dependence on all parameters https://arxiv.org/pdf/2312.03916
## [2] Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics https://dl.acm.org/doi/pdf/10.1145/3313276.3316366
## Main algorithm: [1]
##  - LCU comes from [2], coefficients could be complex numbers
##  - Amplitude amplification

import numpy
import qiskit.quantum_info
import scipy
import qiskit
from typing import Callable

from oracle_synth import *



######
## LCHS for Time-independent ODE IVP

# Solve for du/dt = -Au(t) + b(t) with u(0) = u0

# References
# [1] Quantum algorithm for linear non-unitary dynamics with near-optimal dependence on all parameters https://arxiv.org/pdf/2312.03916

####

# Input preparation
def cart_decomp(A:numpy.matrix, eigcheck=True) -> tuple[numpy.matrix,numpy.matrix]:
    """
    Cartesian decomposition of a numpy A
    Generate L and H matrices from a given numpy A, L = 0.5*(A + A^T), H = -0.5j*(A - A^T), A = L + iH
    inumpyut: 
     - A, a square numpy
    output:
        - L, a square numpy, must be positive semidefinite for the stability of LCHS method
        - H, a square numpy
    """
    ## Return L and H
    L = 0.5*(A+A.conjugate().transpose())
    H = 0.5j*(A-A.conjugate().transpose())

    ## Check if L is positive semidefinite
    if eigcheck:
        L_eigvals = numpy.linalg.eigh(L)[0]
        for eigval in L_eigvals:
            if eigval < 0:
                raise Warning("L has negative eigenvalues:", eigval)
    return L, H 


##----------------------------------------------------------------------------------------------------------------

# Shared Calculations of Parameters between homogenous and inhomogenous solutions

## Define Kernal functions
def cbeta(beta:float) -> float:
    ## This can be changed if different C_beta is used
    ## C_beta = 2\pi e^{-2^beta}
    return 2*numpy.pi*numpy.exp(-2**beta)

def kernel_func(beta:float, z:float) -> complex:
    ## 0 < beta < 1
    ## Kernal functon Eq. (7) in [1]
    if beta <=0 or beta >= 1:
        raise ValueError("beta must be in (0,1), inputted beta = ", beta)
    return 1/(cbeta(beta)*numpy.exp( (1+1j*z)**beta ) )

def gk(beta:float, k:float) -> complex:
    ## 0 < beta < 1
    ## Coefficient f(k)/(1-ik) in Eq. (6) in [1], or g(k) in Eq. (60) in [1]
    return kernel_func(beta, k)/(1- 1j*k)

## Truncation of the infinite integral
def trunc_K(beta:float, epsilon:float, h1:float) -> float:
    ## Truncate the integral \int_\mathbb{R} g(k)U(T,k)dk to \int_{-K}^K g(k)U(T,k)dk
    ## (63) in [1]
    temp = (numpy.log(1/epsilon))**(1/beta)
    return numpy.ceil(temp/h1)*h1

# def trunc_Kexact(beta, epsilon, h1):
#     ## Truncate the integral \int_\mathbb{R} g(k)U(T,k)dk to \int_{-K}^K g(k)U(T,k)dk
#     ## (63) in [1]
#     temp_B = int(numpy.ceil(1/beta))
#     temp_1 = 2**(temp_B+1)*numpy.prod(numpy.arange(1,temp_B+1))/( cbeta(beta)*(numpy.cos(beta*numpy.pi*0.5)**temp_B) )
#     temp_2 = (numpy.log(temp_1)+numpy.log(1/epsilon))/(numpy.cos(beta*numpy.pi*0.5)) ## should also - ln(K), but maybe too small to care
#     temp = (2*temp_2)**(1/beta)
#     return numpy.ceil(temp/h1)*h1

## Step size h1
def step_size_h1(T:float, L:numpy.matrix) -> float:
    ## step size h1 to discretize \int_{-K}^K g(k)U(T,k)dk
    ## since L is time-independent, ||L|| = max_t ||L(t)||
    ## Eq. (65) in [1]
    return 1/(numpy.exp(1)*T*numpy.linalg.norm(L, ord=2))

## Number of nodes in each shorter interval
def n_node_Q(beta:float, epsilon:float, rangeK:float, cb_func=cbeta) -> int:
    ## Number of nodes in each shorter interval [mh_1, (m+1)h_1] for m = -K/h1 to K/h1
    ## Eq. (65) in [1]
    temp = 1/numpy.log(4) * numpy.log( (8/(3*cb_func(beta))) * (rangeK/epsilon) )
    return int(numpy.ceil(temp))

## Gaussian-Legendre quadrature
def gauss_quadrature(lower_end:float,interval_step:float,num_slices:float) -> tuple[numpy.ndarray,numpy.ndarray]:
    ## obtain the standard Sampling points and weights 
    ## for Gauss-Legendre quadrature in [lower_end, lower_end+interval_step]
    ##
    ## Convert C_i, X_i in range [-1,1] to c_i, x_i in range [a+mh,a+mh+h], 
    ## the convertion is c_i = h/2*C_i and x_i = h/2 x_i + (a+mh+a+mh+h)/2 = h/2 x_i + (a + mh) +h/2
    ##
    samp_points, weights = numpy.polynomial.legendre.leggauss(num_slices)
    ## Change of variable to [a,b], see A.4 in [1]
    weights_ab = 0.5*interval_step*weights
    samp_points_ab = 0.5*interval_step*samp_points + lower_end + 0.5*interval_step
    return samp_points_ab, weights_ab ## kqm and wq in Eq. (61) in [1]

##----------------------------------------------------------------------------------------------------------------

# Parameter Calculation for Inhomogenous Solution
def trunc_K_inho(beta:float, epsilon:float, bnorm:float, h1:float) -> float:
    ## Lemma 12 (Eq. (73) ) in [1]
    ## Input
    ## beta: 0 < beta < 1
    ## epsilon: error tolerance
    ## bnorm: \int_{s\in [0,T]} \|b(s)\| ds
    ## h1: step size h1 for [-K, K], (65) in [1]
    if beta <= 0 or beta >= 1:
        raise ValueError("beta must be in (0,1), inputted beta = ", beta)
    temp = (numpy.log(1+bnorm/epsilon))**(1/beta)
    return numpy.ceil(temp/h1)*h1

def norm_lambda(A:numpy.matrix) -> float:
    ## \Lambda = \sup_{p \geq 0, t\in [0,T]} \|A^{(p)}\|^{1/(p+1)}
    ## A is time independent, so just 0th order time derivative
    ## Input
    ## A: numpy A
    ## Output
    ## \Lambda, largest norm \|A\|^{1/(p+1)} over p and t
    return numpy.linalg.norm(A, ord=2)

def norm_Xi(t0:float, tT:float, p:int, func_btp:Callable[[complex], complex]) -> float:
    ## Use func_bt for the function output time deritatives of b(t)
    ## \Xi = \sup_{p \geq 0, t\in [0,T]} \|b^{(p)}\|^{1/(p+1)}
    ## Input
    ## t0: start time
    ## tT: end time
    ## p: order of derivative
    ## func_btp: function to compute the p-th derivative of b(t)
    ## Output
    ## \Xi, largest norm \|b^{(p)}\|^{1/(p+1)} over p and t
    time_slices = 100
    time_points = numpy.linspace(t0, tT, time_slices)
    max_norm = 0
    for time_point in time_points:
        norm_val = numpy.linalg.norm(func_btp(time_point), ord=2)
        temp = numpy.power( norm_val, 1/(p+1) ) if numpy.abs(norm_val-1e-8) >0 else 0
        if temp > max_norm:
            max_norm = temp
    return max_norm

def norm_bL1(t0:float, tT:float, func_bt:Callable[[complex], complex]) -> float:
    ## \int_{s\in [0,T]} \|b(s)\| ds
    ## Input
    ## t0: start time
    ## tT: end time
    ## func_bt: function to compute b(t)
    time_slices = 100
    samp_points, weights = numpy.polynomial.legendre.leggauss(time_slices)
    time_step = (tT-t0)
    weights = 0.5*time_step*weights
    samp_points = 0.5*time_step*samp_points + (t0 + 0.5*time_step)
    temp_sum = 0
    for i in range(len(samp_points)):
        temp_sum += weights[i]*numpy.linalg.norm(func_bt(samp_points[i]),ord=2)
    return temp_sum

# def scipy_int(t0, tT, int_func=bt):
#     ## an alternative for norm_bL1, just for checking
#     return scipy.integrate.quad(lambda x: numpy.linalg.norm(int_func(x),ord=2), t0, tT)

def step_size_h2(K:float, Lam:float, Xi:float, tT:float) -> float:
    ## Lemma 12 (Eq. (73) ) in [1]
    ## h_2 = \frac{1}{eK(\Lambda+\Xi)}
    ## Note, the return is <= \lceil T/h_2  \rceil to make the number of slices integer
    ## Input
    ## K: truncation range
    ## Lam: \Lambda, largest norm \|A\|^{1/(p+1)} over p and t
    ## Xi: \Xi, largest norm \|b^{(p)}\|^{1/(p+1)} over p and t
    ## tT: end time
    theo_h2 = 1/(numpy.exp(1)*K*(Lam + Xi))
    slice_T = numpy.ceil(tT/theo_h2)
    return tT/slice_T

def n_node_Q2_inho(epsilon:float, tT:float, Lam:float, Xi:float, cnorm:float) -> int:
    ## Combine Eq. (234) and (235) in [1]
    ## Input
    ## epsilon: error tolerance
    ## tT: end time
    ## Lam: \Lambda, largest norm \|A\|^{1/(p+1)} over p and t
    ## Xi: \Xi, largest norm \|b^{(p)}\|^{1/(p+1)} over p and t
    ## cnorm: ||c||_1, could just use 1 for simplicity, this norm should be the same as the one in homogenous solution
    return int(numpy.ceil(cnorm*1/(numpy.log(4))*numpy.log( 2*numpy.exp(1)*tT*(Lam+Xi)/epsilon )))



##----------------------------------------------------------------------------------------------------------------

# Quantum Operators but in Classical Operation
## Define Time-Independent Unitary Time-Evolution Operator $U(T, s, k) = \mathcal{T} e^{-i \int^T_{s} (kL+H) \mathrm{d} s'}= \mathcal{T} e^{-i (T-s) (kL+H)}$
def utsk(tT, s, k, L, H):
    ## U(T,s,k) in Eq. (70) in [1]
    ## ignore time-ordering operator since we don't trotter it
    return scipy.linalg.expm(-1j*(tT-s)*(k*L+H))

def utk(tT, k, L, H):
    ## Wrapper for utsk with s=0, used in homogenous solution
    return utsk(tT, 0, k, L, H)

## trotter exp(-i (kL+H)t) => exp(-i H t)exp(-i kL t)
def utsk_L(tT, s, k, L):
    return scipy.linalg.expm(-1j*(tT-s)*(k*L))

def utsk_H(tT, s, H):
    return scipy.linalg.expm(-1j*(tT-s)*H)

def utk_L(tT, k, L):
    return utsk_L(tT, 0, k, L)

def utk_H(tT, H):
    return utsk_H(tT, 0, H)


##----------------------------------------------------------------------------------------------------------------

# LCHS but in Classical Operations
## Homogenous Part of the Solution
def class_lchs_tihs(A:numpy.matrix, u0:numpy.matrix, tT:float, beta:float, epsilon:float, 
                    trunc_multiplier:float=2.0, trotterLH:bool=False,
                    verbose:int=0) -> tuple[numpy.matrix,numpy.matrix]: 
    '''
    Solve Homogeneous IVP du/dt = -Au(t) with u(0) = u0
    Input:
    - A: numpy matrix, coefficient matrix, NOTE the minus sign in the equation
    - u0: numpy matrix, initial condition
    - tT: float, end time
    - beta: float, 0 < beta < 1, parameter in the kernel function
    - epsilon: float, error tolerance
    - trunc_multiplier: float, multiplier for truncation range, just for higher accuracy. See (63) in [1]. Default is 2.
    - verbose: bool, print out parameters
    Output:
    - res_mat: numpy matrix, the matrix representation of the solution operator (summation of all unitaties)
    - uT: numpy matrix, the solution state at time tT, uT = res_mat*u0
    '''
    L,H = cart_decomp(A)
    h1 = step_size_h1(tT, L)  ## step size h1 for [-K, K], (65) in [1]
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
    res_mat = numpy.zeros(A.shape, dtype=complex)
    c_sum = 0 ## compute ||c||_1
    for munshit in range(2*kh1):
        m = -kh1+munshit ## shift to start from -K/h1
        kqms, wqs = gauss_quadrature(m*h1, h1, Q) ## Gaussian quadrature points and weights in [mh_1, (m+1)h_1]
        
        for qi in range(Q):
            cqm = wqs[qi]*gk(beta, kqms[qi])
            c_sum += numpy.abs(cqm)
            if trotterLH:
                res_mat += cqm* utk_L(tT, kqms[qi], L)  ## w_q*g(k_{q,m})*U(T, k_{q,m}) ## (61) in [1] (This should be quantum)
            else:
                res_mat += cqm* utk(tT, kqms[qi], L, H)  ## w_q*g(k_{q,m})*U(T, k_{q,m}) ## (61) in [1] (This should be quantum)
    if trotterLH:
        res_mat = utk_H(tT,  0.5*H) @ res_mat @ utk_H(tT, 0.5*H) ## exp(i(A+B)) approx exp(iA/2) exp(iB) exp(iA/2), (4.104) in Nielsen and Chuang (10th anniversary edition)
    if verbose>0:
        print("  ||c||_1 =", c_sum)
    u0_norm = u0/numpy.linalg.norm(u0,ord=2)
    uT = res_mat.dot(u0_norm)
    return res_mat, uT


## Inhomogenous Part of the Solution
def class_lchs_tips(A:numpy.matrix, u0:numpy.matrix, func_bt:Callable[[complex],complex], tT:float, beta:float, epsilon:float, trunc_multiplier:float=2.0, verbose:int=0): 
    '''
    Solve Inhomogeneous IVP du/dt = -Au(t)+b(t) with u(0) = u0, where b(t) is a user-defined function return the evluation of b(t) at time t
    Input:
    - A: numpy matrix, coefficient matrix, NOTE the minus sign in the equation
    - u0: numpy matrix, initial condition
    - func_bt: function, return the evaluation of b(t) at time t
    - tT: float, end time
    - beta: float, 0 < beta < 1, parameter in the kernel function
    - epsilon: float, error tolerance
    - trunc_multiplier: float, multiplier for truncation range, just for higher accuracy. See (63) in [1]. Default is 2.
    - verbose: bool, print out parameters
    Output:
    - res_vec: numpy matrix, the vector from the summation of the inner product between unitaries and |b(s)> for all the discretized s in [0,T]
    '''
    ## Lemma 12 in [1]
    L,H = cart_decomp(A)
    bL1 = norm_bL1(0, T, func_bt)
    h1 = step_size_h1(tT, L)  ## step size h1 for [-K, K], (65) in [1]
    K = trunc_multiplier*trunc_K_inho(beta, epsilon, bL1, h1)
    Lam = norm_lambda(A)
    Xi = norm_Xi(0, T, 0, func_bt)
    h2 = step_size_h2(K, Lam, Xi, tT)
    Q1 = n_node_Q(beta, epsilon, K)
    Q2 = n_node_Q2_inho(epsilon, T, Lam, Xi, 1)

    kh1 = int(K/h1)
    Th2 = int(T/h2)
    M = int(2*kh1*Q1)
    Mp = int(Th2*Q2)

    if verbose>0:
        print("  Preset parameters T = ", tT, "beta = ", beta, "epsilon = ", epsilon)
        print("  Truncation range [-K,K] K =", K)
        print("  Step size h1 =", h1)
        print("  Step size h2 =", h2)
        print("  Number of nodes in [mh_1, (m+1)h_1] Q1 =", Q1)
        print("  Number of nodes in [mh_2, (m+1)h_2] Q2 =", Q2)
        print("  Lambda=", Lam)
        print("  Xi=", Xi)
        print("  Total number of nodes M =", M)
        print("  Total number of nodes M' =", Mp)
        print("  Total number of nodes M*M' =", M*Mp)

    res_vec = numpy.zeros(u0.shape, dtype=complex)
    c_sum = 0 ## compute ||c||_1
    
    for munshit in range(2*kh1):
        m = -kh1+munshit ## shift to start from -K/h1
        kqms, wqs = gauss_quadrature(m*h1,h1,Q1) ## Gaussian quadrature points and weights in [mh_1, (m+1)h_1]
        for qi in range(Q1):
            c1qm = wqs[qi]*gk(beta, kqms[qi])
            c_sum += numpy.abs(c1qm)
            for m2 in range(Th2):
                sqms, w2qs = gauss_quadrature(m2*h2,h2,Q2)
                for qi2 in range(Q2):
                    sqm = sqms[qi2]
                    bsqm = numpy.array(func_bt(sqm))
                    c2qm = w2qs[qi2]*numpy.linalg.norm(bsqm, ord=2)
                    basqm_norm = bsqm/numpy.linalg.norm(bsqm, ord=2)
                    print("Progress: ", f"{munshit}/{2*kh1}",f"{m2}/{Th2}", end="\r")
                    res_vec += c2qm*c1qm* (utsk(tT,sqms[qi2],kqms[qi],L,H).dot(basqm_norm))  ## This should be quantum
    if verbose>0:
        print()
        print("  ||c||_1 =", c_sum)
    return res_vec

##----------------------------------------------------------------------------------------------------------------



##----------------------------------------------------------------------------------------------------------------


## Homogenous Part of the Solution
def quant_lchs_tihs(A:numpy.matrix, u0:numpy.matrix, tT:float, beta:float, epsilon:float, 
                    trunc_multiplier=2, trotterLH:bool=False,
                    qiskit_api:bool=False, verbose:int=0, 
                    no_state_prep:bool=False,
                    debug:bool=False, rich_return:bool=False) -> tuple[numpy.matrix,numpy.matrix]: 
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
    from lcu import lcu_generator
    from utils_synth import qiskit_normal_order_switch, qiskit_normal_order_switch_vec
    from oracle_synth import synthu_qsd, stateprep_ucr
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
            # if numpy.abs(numpy.angle(cqm)) > 1e-12:
            #     cqm = cqm/numpy.exp(1j*numpy.angle(cqm)) ## absorb the phase to the unitaries
            #     umat = numpy.exp(1j*numpy.angle(cqm))*umat
            # coeffs.append(cqm)
            # unitaries.append(umat)
    if verbose > 0:
        print("  ||c||_1 =", c_sum)
        
    ## Obtain the linear combination by LCU
    # lcu_circ = lcu_generator(coeffs, unitaries, initial_state_circ=state_prep_circ, qiskit_api=qiskit_api) ## NOTE: the return circuit is in qiskit order
    lcu_circ, coeffs, unitaries, coeffs_1norm = lcu_generator(coeffs_unrot, unitaries_unrot, initial_state_circ=None, verbose=verbose, qiskit_api=qiskit_api, debug=debug) ## NOTE: the return circuit is NOT in qiskit order
    num_control_qubits = nearest_num_qubit(len(coeffs))
    if trotterLH:
        exph_circ = qiskit.QuantumCircuit(int(numpy.log2(H.shape[0])))
        synthu_qsd(utk_H(tT, 0.5*H), exph_circ) ## exp(i(A+B)) approx exp(iA/2) exp(iB) exp(iA/2), (4.104) in Nielsen and Chuang (10th anniversary edition)
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


##----------------------------------------------------------------------------------------------------------------











## Example Demonstration
if __name__ == "__main__":
    ###----------------------------------------------------------------------------------------------------------------
    ## Problem Dimension
    nq = 2 #2
    ## Define random A and u0
    dim = 2**nq
    rng = numpy.random.Generator(numpy.random.PCG64(178984893489))
    ## Random numpy
    A = (numpy.matrix(rng.random((dim,dim)),dtype=complex)-0.5)  + 1j*(numpy.matrix(rng.random((dim,dim)),dtype=complex)-0.5)
    A = 0.1*A + 0.25*numpy.identity(dim)
    L,H = cart_decomp(A)
    print(f"Norm of A: {numpy.linalg.norm(A, ord=2)}, Norm of L: {numpy.linalg.norm(L, ord=2)}, Norm of H: {numpy.linalg.norm(H, ord=2)}")
    ## Define b(t) and its first derivative as functions
    def bt(t):
        ## dtype complex is necessary for scipy to compute complex integral
        # return 0.1*numpy.array([ 1*t+1j, 1, 3j*t, (2+1j)*t],dtype=complex).reshape(-1,1) 
        # return 0.1*numpy.array([ -0.3*t+1j, 0.5-0.1j],dtype=complex).reshape(-1,1) 
        return rng.random((dim,1))-0.5 + 1j*(rng.random((dim,1))-0.5)
    ## Random initial state
    u0 = numpy.matrix( rng.random((dim,1)) ,dtype=complex) ## dtype complex is necessary for scipy to compute complex integral
    u0 = u0/numpy.linalg.norm(u0, ord=2)

    ## Define the time interval [0,T], beta and epsilon for LCHS parameters, epislon is the error tolerance
    T = 1
    beta = 0.9 # 0< beta < 1
    epsilon = 0.05 #0.05
    tests_class_ho = True
    tests_quant_qis = True
    tests_quant_my = True
    tests_quant_mytrotter = False

    tests_class_inho = False

    tests_bulk_hoinho= False

    debug=False


    ### Quick check for eigenvalues
    print("Eigenvalues of L:", numpy.linalg.eigvals(L))
    ## Function for Scipy
    def ode_func_ho(t,u):
        return numpy.array(-A.dot(u).reshape(-1))[0]
    def ode_func_inho(t,u):
        return numpy.array(-A.dot(u)+bt(t).reshape(-1))[0]
        # return numpy.array(-(L_exp+1j*H_exp).real.dot(u)+bt(t).reshape(-1))[0]
    ### Scipy Homogenous solution
    spi_sol_ho = scipy.integrate.solve_ivp(ode_func_ho, [0,T],numpy.array(u0.reshape(-1))[0], method='RK45')
    spi_uT_ho = spi_sol_ho.y[:,-1]
    spi_uT_ho_norm = spi_uT_ho/numpy.linalg.norm(spi_uT_ho,ord=2)
    ### Scipy Inhomogenous solution
    spi_sol_inho = scipy.integrate.solve_ivp(ode_func_inho, [0,T],numpy.array(u0.reshape(-1))[0], method='RK45')
    spi_sol_ps = spi_sol_inho.y[:,-1] - spi_sol_ho.y[:,-1] ## subtract homogeneous solution
    spi_sol_ps_norm = spi_sol_ps/numpy.linalg.norm(spi_sol_ps,ord=2)


    ###----------------------------------------------------------------------------------------------------------------
    print("="*60)
    if tests_class_ho:
        print("\n\nTests with Classical Subroutine (Homogeneous, no trotter)")
        ## Solve homogenous part
        exp_op, uT = class_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=2, trotterLH=False, verbose=1)
        uT = numpy.array(uT).reshape(-1)
        if numpy.linalg.norm(uT.imag,ord=2) < 1e-12:
            uT = uT.real
        uT_err = numpy.linalg.norm(uT - spi_uT_ho,ord=2)
        print("  Homogeneous u(T)=", uT, "  Norm=", numpy.linalg.norm(uT,ord=2))
        print("  SciPy Sol   u(T)=", spi_uT_ho, "  Norm=", numpy.linalg.norm(spi_uT_ho,ord=2))
        print("  Homogeneous solution error u(T)         :", uT_err, "   Relative error:", uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))

        print("\n\nTests with Classical Subroutine (Homogeneous, w/ trotter)")
        ## Solve homogenous part
        exp_op, uT = class_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=2, trotterLH=True,verbose=0)
        uT = numpy.array(uT).reshape(-1)
        if numpy.linalg.norm(uT.imag,ord=2) < 1e-12:
            uT = uT.real
        uT_err = numpy.linalg.norm(uT - spi_uT_ho,ord=2)
        print("  Homogeneous solution error u(T)(trotter):", uT_err, "   Relative error:", uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))
   
    ###----------------------------------------------------------------------------------------------------------------
    print("="*60)
    if tests_quant_my:
        print("\n\nTests with Quantum Subroutine (My Implementation)")
        quant_uT, lchs_circ_ho = quant_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=2, qiskit_api=False, verbose=1, debug=debug)
        # qiskit.qasm2.dump(lchs_circ_ho, f"./lchs_ho_Asize{A.shape[0]}.qasm")

        quant_uT = numpy.array(quant_uT).reshape(-1)
        if numpy.linalg.norm(quant_uT.imag,ord=2) < 1e-12:
            quant_uT = quant_uT.real

        quant_uT_err = numpy.linalg.norm(quant_uT - spi_uT_ho,ord=2)
        print("  Homogeneous u(T)=", quant_uT, "  Norm=", numpy.linalg.norm(quant_uT,ord=2))
        print("  SciPy Sol   u(T)=", spi_uT_ho, "  Norm=", numpy.linalg.norm(spi_uT_ho,ord=2))
        print("  Homogeneous solution error u(T)         :", quant_uT_err, "   Relative error:", quant_uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))

    if tests_quant_mytrotter:
        print("\n\nTests with Quantum Trottered Subroutine (My Implementation)")
        quant_uT, lchs_circ_ho = quant_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=2, trotterLH=True,qiskit_api=False, verbose=1, debug=debug)


        quant_uT = numpy.array(quant_uT).reshape(-1)
        if numpy.linalg.norm(quant_uT.imag,ord=2) < 1e-12:
            quant_uT = quant_uT.real

        quant_uT_err = numpy.linalg.norm(quant_uT - spi_uT_ho,ord=2)
        print("  Homogeneous u(T)=", quant_uT, "  Norm=", numpy.linalg.norm(quant_uT,ord=2))
        print("  SciPy Sol   u(T)=", spi_uT_ho, "  Norm=", numpy.linalg.norm(spi_uT_ho,ord=2))
        print("  Homogeneous solution error u(T)         :", quant_uT_err, "   Relative error:", quant_uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))


    if tests_quant_qis:
        print("\n\nTests with Quantum Subroutine (Qiskit API)")
        quant_uT, lchs_circ_ho = quant_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=2, qiskit_api=True, verbose=1, debug=debug)

        quant_uT = numpy.array(quant_uT).reshape(-1)
        if numpy.linalg.norm(quant_uT.imag,ord=2) < 1e-12:
            quant_uT = quant_uT.real
        quant_uT_err = numpy.linalg.norm(quant_uT - spi_uT_ho,ord=2)
        print("  Homogeneous u(T)=", quant_uT, "  Norm=", numpy.linalg.norm(quant_uT,ord=2))
        print("  SciPy Sol   u(T)=", spi_uT_ho, "  Norm=", numpy.linalg.norm(spi_uT_ho,ord=2))
        print("  Homogeneous solution error u(T)         :", quant_uT_err, "   Relative error:", quant_uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))

    

    ###----------------------------------------------------------------------------------------------------------------
    print("="*60)
    if tests_class_inho:
        ## solve inhomogenous part
        print("\n\nTests with Classical Subroutine (Inhomogeneous)")
        uT_inho = class_lchs_tips(A, u0, bt, T, beta, epsilon, trunc_multiplier=2, verbose=1)
        uT_inho = numpy.array(uT_inho).reshape(-1)
        if numpy.linalg.norm(uT_inho.imag,ord=2) < 1e-12:
            uT_inho = uT_inho.real
        inho_sol_err = numpy.linalg.norm(uT_inho - spi_sol_ps,ord=2)
        print("  Inhomogeneous u(T)=", uT_inho)

        ## Validation with Scipy
        # print("\n Verification with Scipy")

        ### Full solution
        uT_full = uT + uT_inho
        spi_full_sol = spi_sol_inho.y[:,-1]
        full_sol_err = numpy.linalg.norm(uT_full - spi_full_sol,ord=2)

        print(" Full Solution")
        print(" epsilon:", epsilon)
        print("\n ------------------------------------------------------------------------")
        print("\n Un-normalized, as in the paper")
        print(" Full solution from scipy:", spi_full_sol, "  Norm=", numpy.linalg.norm(spi_full_sol,ord=2))
        print(" Full solution from LCHS :", uT_full, "  Norm=", numpy.linalg.norm(uT_full,ord=2))
        print("\n Full solution error             :", full_sol_err, "    Relative error:", full_sol_err/numpy.linalg.norm(spi_full_sol,ord=2))
        print("  - Homogeneous solution error  :", uT_err, "   Relative error:", uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))
        print("  - Inhomogeneous solution error:", inho_sol_err, "    Relative error:", inho_sol_err/numpy.linalg.norm(spi_sol_ps,ord=2))
        print("\n ------------------------------------------------------------------------")
    ###----------------------------------------------------------------------------------------------------------------



    if tests_bulk_hoinho:
        ## Define the time interval [0,T], beta and epsilon for LCHS parameters, epislon is the error tolerance
        T = 1
        beta = 0.9 # 0< beta < 1
        for epsilon in [0.01, 0.001]:
            for nq in [3,4,5]:
                for trunc_multiplier in [2, 4]:
                    print("\n\n\n")
                    print(">"*20, f"NumQubits {nq} Epsilon {epsilon} Trunc_Multipler {trunc_multiplier}", "<"*20)
                    ## Define random A and u0
                    dim = 2**nq
                    rng = numpy.random.Generator(numpy.random.PCG64(17))
                    ## Random numpy
                    A = (numpy.matrix(rng.random((dim,dim)),dtype=complex)-0.5)  + 1j*(numpy.matrix(rng.random((dim,dim)),dtype=complex)-0.5)
                    A = 0.1*A + 0.5*numpy.identity(dim)
                    L,H = cart_decomp(A)
                    print(f"Norm of A: {numpy.linalg.norm(A, ord=2)}, Norm of L: {numpy.linalg.norm(L, ord=2)}, Norm of H: {numpy.linalg.norm(H, ord=2)}")
                    ## Define b(t) and its first derivative as functions
                    def bt(t):
                        ## dtype complex is necessary for scipy to compute complex integral
                        # return 0.1*numpy.array([ 1*t+1j, 1, 3j*t, (2+1j)*t],dtype=complex).reshape(-1,1) 
                        # return 0.1*numpy.array([ -0.3*t+1j, 0.5-0.1j],dtype=complex).reshape(-1,1) 
                        # return rng.random((dim,1))-0.5 + 1j*(rng.random((dim,1))-0.5)
                        return (numpy.matrix( rng.random((dim,1)) ,dtype=complex)-0.5) + 1j*(numpy.matrix( rng.random((dim,1)) ,dtype=complex)-0.5)
                    ## Random initial state
                    u0 = numpy.matrix( rng.random((dim,1)) ,dtype=complex) ## dtype complex is necessary for scipy to compute complex integral
                    u0 = u0/numpy.linalg.norm(u0, ord=2)

                    ##
                    print("Eigenvalue of L", numpy.linalg.eigvals(L).real )
                    for ev in numpy.linalg.eigvals(L).real:
                        if ev < 1e-10:
                            raise ValueError(f"Eigenvalue of L is too small {ev}, LCHS may not converge")
                    ## Function for Scipy
                    def ode_func_ho(t,u):
                        return numpy.array(-A.dot(u).reshape(-1))[0]
                    def ode_func_inho(t,u):
                        return numpy.array(-A.dot(u)+bt(t).reshape(-1))[0]
                        # return numpy.array(-(L_exp+1j*H_exp).real.dot(u)+bt(t).reshape(-1))[0]
                    ### Scipy Homogenous solution
                    spi_sol_ho = scipy.integrate.solve_ivp(ode_func_ho, [0,T],numpy.array(u0.reshape(-1))[0], method='RK45')
                    spi_uT_ho = spi_sol_ho.y[:,-1]
                    spi_uT_ho_norm = spi_uT_ho/numpy.linalg.norm(spi_uT_ho,ord=2)
                    ### Scipy Inhomogenous solution
                    spi_sol_inho = scipy.integrate.solve_ivp(ode_func_inho, [0,T],numpy.array(u0.reshape(-1))[0], method='RK45')
                    spi_sol_ps = spi_sol_inho.y[:,-1] - spi_sol_ho.y[:,-1] ## subtract homogeneous solution
                    spi_sol_ps_norm = spi_sol_ps/numpy.linalg.norm(spi_sol_ps,ord=2)

                    ###
                    print("\n\nTests with Classical Subroutine (Homogeneous)")
                    ## Solve homogenous part
                    exp_op, uT = class_lchs_tihs(A, u0, T, beta, epsilon, trunc_multiplier=trunc_multiplier, verbose=1)
                    uT = numpy.array(uT).reshape(-1)
                    if numpy.linalg.norm(uT.imag,ord=2) < 1e-12:
                        uT = uT.real
                    uT_err = numpy.linalg.norm(uT - spi_uT_ho,ord=2)
                    print("  Homogeneous u(T)=", uT, "  Norm=", numpy.linalg.norm(uT,ord=2))
                    print("  SciPy Sol   u(T)=", spi_uT_ho, "  Norm=", numpy.linalg.norm(spi_uT_ho,ord=2))
                    print("  Homogeneous solution error u(T)         :", uT_err, "   Relative error:", uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))
            


                    ## solve inhomogenous part
                    print("\n\nTests with Classical Subroutine (Inhomogeneous)")
                    uT_inho = class_lchs_tips(A, u0, bt, T, beta, epsilon, trunc_multiplier=trunc_multiplier, verbose=1)
                    uT_inho = numpy.array(uT_inho).reshape(-1)
                    if numpy.linalg.norm(uT_inho.imag,ord=2) < 1e-12:
                        uT_inho = uT_inho.real
                    inho_sol_err = numpy.linalg.norm(uT_inho - spi_sol_ps,ord=2)
                    print("  Inhomogeneous u(T)=", uT_inho)

                    ## Validation with Scipy
                    # print("\n Verification with Scipy")

                    ### Full solution
                    uT_full = uT + uT_inho
                    spi_full_sol = spi_sol_inho.y[:,-1]
                    full_sol_err = numpy.linalg.norm(uT_full - spi_full_sol,ord=2)
                    
                    print("\n ------------------------------------------------------------------------")
                    print(" Full Solution")
                    print(" epsilon:", epsilon, "num_qubits:", nq, "trunc_multiplier:", trunc_multiplier)
                    print("\n Un-normalized, as in the paper")
                    print(" Full solution from scipy:", spi_full_sol, "  Norm=", numpy.linalg.norm(spi_full_sol,ord=2))
                    print(" Full solution from LCHS :", uT_full, "  Norm=", numpy.linalg.norm(uT_full,ord=2))
                    print("\n Full solution error             :", full_sol_err, "    Relative error:", full_sol_err/numpy.linalg.norm(spi_full_sol,ord=2))
                    print("  - Homogeneous solution error  :", uT_err, "   Relative error:", uT_err/numpy.linalg.norm(spi_uT_ho,ord=2))
                    print("  - Inhomogeneous solution error:", inho_sol_err, "    Relative error:", inho_sol_err/numpy.linalg.norm(spi_sol_ps,ord=2))
                    print("\n ------------------------------------------------------------------------")
