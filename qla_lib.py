### Plan
## References:
## [1] Quantum algorithm for linear non-unitary dynamics with near-optimal dependence on all parameters https://arxiv.org/pdf/2312.03916
## [2] Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics https://dl.acm.org/doi/pdf/10.1145/3313276.3316366
## [3] FABLE: Fast Approximate Quantum Circuits for Block-Encodings https://arxiv.org/abs/2205.00081
## [4] Explicit Quantum Circuits for Block Encodings of Certain Sparse Matrices https://arxiv.org/pdf/2203.10236
## Main algorithm: [1]
##  - (Dense) Block-encoding is [3] (FABLE), this is not in the paper [1], [1] only assumes the existence of a block-encoding oracle
##    - For sparse matrix, may consider [4], but only MATLAB code
##  - LCU comes from [2], coefficients could be complex numbers
##  - Amplitude amplification
## Related articles
## Block encoding with matrix access oracles https://pennylane.ai/qml/demos/tutorial_block_encoding/
## Linear combination of unitaries and block encodings https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/




if __name__ == '__main__':


    print("Success")