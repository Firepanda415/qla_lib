
# LCHS

```
python==3.12.7
qiskit==1.1.2
qiskit-aer==0.14.2
qiskit-algorithms==0.3.1
qiskit-ibm-catalog==0.4.0
qiskit-ibm-runtime==0.32.0
qiskit-iqm==15.6
qiskit-nature==0.7.2
qiskit-qasm3-import==0.5.1
qiskit-serverless==0.20.0
```


## For CD-DV LCHS

Need to install [Bosonic Qiskit](https://github.com/C2QA/bosonic-qiskit)

### Varational CD-DV LCHS
Also need
```
qutip==5.0.4
```
This implementation uses varational hybrid qubit-qumode ansatz to approximate the Hamiltonian evolution in LCHS. The ansatz is based on [this work](https://arxiv.org/abs/2404.10222). The related implementation is in 
 - `src/vbos_lchs.py` include all functions for vaeational bosonic LCHS
 - `VARATIONAL_BOSONIC_LCHS_Example.ipynb` include a test run for varational bosonic LCHS
Ansatz optimization is based on the paper author's own implementation in [this repo](https://github.com/CQDMQD/qumode_est_paper).

### Continuous-LCU-based LCHS

Still under development.

## For DV LCHS

### Current Status and Bugs
 - WARNING: because this code is supposed to link to NWQSim, I may swtich the endianness in different functions. I should have left the comments when I switch the endianness.
 - Currently only contain functions deal with time-independent Hamiltonians.
 - It has a quantum-circuit version and a pure classical matrix multiplication version of LCHS. Homogenoueus ODE has both versions, but inhomogenous ODE only has the classical matrix multiplication version.
 - Oracles target general state preparation and unitary synthesis. So no quantum advantage.
 - Oracles have two version: call Qiskit API directly or my own implementaitons. My own implementations has some stability issues. Try the Qiskit API version first.
 - Could need some check on constant calculations. Then number of nodes seems too large.

### Files and Explainations

 - `src/lchs.py` contains the main functions for LCHS
 - `src/lcu.py` for contsructing LCU circuit, called by  `src/lchs.py`
 - `src/oracle_synth.py` for constructing oracles for general state preparation and unitary synthesis, called by `src/lchs.py`. BUT you can just use Qiskit API and ignore this file.
 - `src/utils_synth.py` for some utility functions for `src/oracle_synth.py`.


 Other files are irrelevant to qubit-based LCHS. They are for hybrid qubit-qumode things and I did not finish those code.

 ### Test run   

Run `python src/lchs.py` with
```
    ## Problem Dimension
    nq = 2 #2
    ## Define random A and u0
    dim = 2**nq
    rng = numpy.random.Generator(numpy.random.PCG64(178984893489))

    ## Random numpy coefficients
    A = (numpy.matrix(rng.random((dim,dim)),dtype=complex)-0.5)  + 1j*(numpy.matrix(rng.random((dim,dim)),dtype=complex)-0.5)
    A = 0.1*A + 0.25*numpy.identity(dim)
    L,H = cart_decomp(A)

    T = 1
    beta = 0.9 # 0< beta < 1
    epsilon = 0.05 #0.05
```

Output

```


Norm of A: 0.31010276226810485, Norm of L: 0.3068583169544767, Norm of H: 0.05352769583735938
Eigenvalues of L: [0.18311124+3.85648591e-18j 0.30685832+2.86318642e-18j
 0.23002699+6.70112736e-18j 0.2704607 +1.43347759e-17j]
============================================================


Tests with Classical Subroutine (Homogeneous, no trotter)
  Preset parameters T =  1 beta =  0.9 epsilon =  0.1
  Truncation range [-K,K] K = 7.1931459082991385
  Step size h1 = 1.1988576513831897
  Number of nodes in [mh_1, (m+1)h_1] Q = 4
  Total number of nodes M = 48
  Truncation error bound = 58.90965244543972
  Quadrature error bound = 0.07707081076763965
  Total error bound = 58.986723256207355
  ||c||_1 = 1.7283574918199
  ||u0||_2 = 1.0
  Homogeneous u(T)=            [0.50114113-0.02330927j 0.11933171-0.00525424j 0.43429712-0.00789454j
 0.33944308+0.02063702j]   Norm= 0.7551670150643808
  SciPy Sol   u(T)=            [0.51733798+0.01563855j 0.12604876-0.01308981j 0.42970522+0.01149142j
 0.36258908-0.00795471j]   Norm= 0.7747615932962054
  Homogeneous u(T)/norm_ratio= [0.51414442-0.02391408j 0.12242805-0.00539058j 0.44556597-0.00809938j
 0.34825072+0.0211725j ]   Norm= 0.7747615932962055
  Homogeneous solution error u(T)           : 0.06029851589347491    Relative error: 0.07782847835414279
  Homogeneous solution error u(T)/norm_ratio: 0.05776108844808741    Relative error: 0.07455337093097786


Tests with Classical Subroutine (Homogeneous, w/ trotter)
  Homogeneous solution error u(T)(trotter): 0.060326484526332484    Relative error: 0.07786457801770327
============================================================


Tests with Quantum Subroutine (Qiskit API)
  Preset parameters T =  1 beta =  0.9 epsilon =  0.1
  Truncation range [-K,K] K = 7.1931459082991385
  Step size h1 = 1.1988576513831897
  Number of nodes in [mh_1, (m+1)h_1] Q = 4
  Total number of nodes M = 48
  Truncation error bound = 58.90965244543972
  Quadrature error bound = 0.07707081076763965
  Total error bound = 58.986723256207355
  ||c||_1 = 1.7283574918199
  LCU-Oracle: num_qubits_control= 6 num_qubits_op= 2
  Number of Qubits: 8
  Homogeneous u(T)=            [0.50114113-0.02330927j 0.11933171-0.00525424j 0.43429712-0.00789454j
 0.33944308+0.02063702j]   Norm= 0.7551670149824311
  SciPy Sol   u(T)=            [0.51733798+0.01563855j 0.12604876-0.01308981j 0.42970522+0.01149142j
 0.36258908-0.00795471j]   Norm= 0.7747615932962054
  Homogeneous u(T)/norm_ratio= [0.51414442-0.02391408j 0.12242805-0.00539058j 0.44556597-0.00809938j
 0.34825072+0.0211725j ]   Norm= 0.7747615932962054
  Homogeneous solution error u(T)           : 0.060298515917665524    Relative error: 0.07782847838536609
  Homogeneous solution error u(T)/norm_ratio: 0.05776108844860847    Relative error: 0.0745533709316504
============================================================


Tests with Classical Subroutine (Inhomogeneous)
  Preset parameters T =  1 beta =  0.9 epsilon =  0.1
  Truncation range [-K,K] K = 7.1931459082991385
  Step size h1 = 1.1988576513831897
  Step size h2 = 0.03333333333333333
  Number of nodes in [mh_1, (m+1)h_1] Q1 = 4
  Number of nodes in [mh_2, (m+1)h_2] Q2 = 4
  Lambda= 0.31010276226810485
  Xi= 1.1898653629907012
  Total number of nodes M = 48
  Total number of nodes M' = 120
  Total number of nodes M*M' = 5760
Progress:  11/12 29/30
  ||c||_1 = 1.7283574918199
  Inhomogeneous u(T)= [-0.01981235+0.00501802j -0.01361305-0.00192741j -0.00319253+0.00216004j
  0.01270127-0.00142639j]
 Full Solution
 epsilon: 0.1

 ------------------------------------------------------------------------

 Un-normalized, as in the paper
 Full solution from scipy: [0.49293136+0.01053282j 0.11896813+0.00090358j 0.42726984+0.0008412j
 0.36932005-0.0079321j ]   Norm= 0.7591224787449988
 Full solution from LCHS : [0.4813246 -0.01831468j 0.10572134-0.00719716j 0.4311196 -0.00574651j
 0.35212988+0.01921794j]   Norm= 0.7439754791129101

 Full solution error             : 0.04794621276887492     Relative error: 0.06316004875542726
  - Homogeneous solution error  : 0.060326484526332484    Relative error: 0.07786457801770327
  - Inhomogeneous solution error: 0.024943178273736058     Relative error: 0.7763093966756908

 ------------------------------------------------------------------------Norm of A: 0.31010276226810485, Norm of L: 0.3068583169544767, Norm of H: 0.05352769583735938
Eigenvalues of L: [0.18311124+3.85648591e-18j 0.30685832+2.86318642e-18j
 0.23002699+6.70112736e-18j 0.2704607 +1.43347759e-17j]
============================================================


Tests with Classical Subroutine (Homogeneous, no trotter)
  Preset parameters T =  1 beta =  0.9 epsilon =  0.1
  Truncation range [-K,K] K = 7.1931459082991385
  Step size h1 = 1.1988576513831897
  Number of nodes in [mh_1, (m+1)h_1] Q = 4
  Total number of nodes M = 48
  Truncation error bound = 58.90965244543972
  Quadrature error bound = 0.07707081076763965
  Total error bound = 58.986723256207355
  ||c||_1 = 1.7283574918199
  ||u0||_2 = 1.0
  Homogeneous u(T)=            [0.50114113-0.02330927j 0.11933171-0.00525424j 0.43429712-0.00789454j
 0.33944308+0.02063702j]   Norm= 0.7551670150643808
  SciPy Sol   u(T)=            [0.51733798+0.01563855j 0.12604876-0.01308981j 0.42970522+0.01149142j
 0.36258908-0.00795471j]   Norm= 0.7747615932962054
  Homogeneous u(T)/norm_ratio= [0.51414442-0.02391408j 0.12242805-0.00539058j 0.44556597-0.00809938j
 0.34825072+0.0211725j ]   Norm= 0.7747615932962055
  Homogeneous solution error u(T)           : 0.06029851589347491    Relative error: 0.07782847835414279
  Homogeneous solution error u(T)/norm_ratio: 0.05776108844808741    Relative error: 0.07455337093097786


Tests with Classical Subroutine (Homogeneous, w/ trotter)
  Homogeneous solution error u(T)(trotter): 0.060326484526332484    Relative error: 0.07786457801770327
============================================================


Tests with Quantum Subroutine (Qiskit API)
  Preset parameters T =  1 beta =  0.9 epsilon =  0.1
  Truncation range [-K,K] K = 7.1931459082991385
  Step size h1 = 1.1988576513831897
  Number of nodes in [mh_1, (m+1)h_1] Q = 4
  Total number of nodes M = 48
  Truncation error bound = 58.90965244543972
  Quadrature error bound = 0.07707081076763965
  Total error bound = 58.986723256207355
  ||c||_1 = 1.7283574918199
  LCU-Oracle: num_qubits_control= 6 num_qubits_op= 2
  Number of Qubits: 8
  Homogeneous u(T)=            [0.50114113-0.02330927j 0.11933171-0.00525424j 0.43429712-0.00789454j
 0.33944308+0.02063702j]   Norm= 0.7551670149824311
  SciPy Sol   u(T)=            [0.51733798+0.01563855j 0.12604876-0.01308981j 0.42970522+0.01149142j
 0.36258908-0.00795471j]   Norm= 0.7747615932962054
  Homogeneous u(T)/norm_ratio= [0.51414442-0.02391408j 0.12242805-0.00539058j 0.44556597-0.00809938j
 0.34825072+0.0211725j ]   Norm= 0.7747615932962054
  Homogeneous solution error u(T)           : 0.060298515917665524    Relative error: 0.07782847838536609
  Homogeneous solution error u(T)/norm_ratio: 0.05776108844860847    Relative error: 0.0745533709316504
============================================================


Tests with Classical Subroutine (Inhomogeneous)
  Preset parameters T =  1 beta =  0.9 epsilon =  0.1
  Truncation range [-K,K] K = 7.1931459082991385
  Step size h1 = 1.1988576513831897
  Step size h2 = 0.03333333333333333
  Number of nodes in [mh_1, (m+1)h_1] Q1 = 4
  Number of nodes in [mh_2, (m+1)h_2] Q2 = 4
  Lambda= 0.31010276226810485
  Xi= 1.1898653629907012
  Total number of nodes M = 48
  Total number of nodes M' = 120
  Total number of nodes M*M' = 5760
Progress:  11/12 29/30
  ||c||_1 = 1.7283574918199
  Inhomogeneous u(T)= [-0.01981235+0.00501802j -0.01361305-0.00192741j -0.00319253+0.00216004j
  0.01270127-0.00142639j]
 Full Solution
 epsilon: 0.1

 ------------------------------------------------------------------------

 Un-normalized, as in the paper
 Full solution from scipy: [0.49293136+0.01053282j 0.11896813+0.00090358j 0.42726984+0.0008412j
 0.36932005-0.0079321j ]   Norm= 0.7591224787449988
 Full solution from LCHS : [0.4813246 -0.01831468j 0.10572134-0.00719716j 0.4311196 -0.00574651j
 0.35212988+0.01921794j]   Norm= 0.7439754791129101

 Full solution error             : 0.04794621276887492     Relative error: 0.06316004875542726
  - Homogeneous solution error  : 0.060326484526332484    Relative error: 0.07786457801770327
  - Inhomogeneous solution error: 0.024943178273736058     Relative error: 0.7763093966756908

 ------------------------------------------------------------------------
 ```