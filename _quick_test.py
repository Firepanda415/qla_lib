import numpy as np
import scipy

KET0 = np.array([[1], [0]])
KET1 = np.array([[0], [1]])
K00B = KET0 @ KET0.T
K11B = KET1 @ KET1.T


def RY(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

def RZ(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

def RX(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

def I(n=2):
    return np.eye(n)

def X():
    return np.array([[0, 1],
                     [1, 0]])

def Y():
    return np.array([[0, -1j],
                     [1j, 0]])

def Z():
    return np.array([[1, 0],
                     [0, -1]])

def H():
    return np.array([[1, 1],
                     [1, -1]]) / np.sqrt(2)

# def CNOT():
#     return np.array([[1, 0, 0, 0],
#                      [0, 1, 0, 0],
#                      [0, 0, 0, 1],
#                      [0, 0, 1, 0]])

def CNOT():
    return np.kron(K00B, I()) + np.kron(K11B, X())

def CNOTINV():
    return np.kron(I(), K00B) + np.kron(X(), K11B)

def CZ():
    return np.kron(K00B, I()) + np.kron(K11B, Z())  