from qiskit import QuantumCircuit

circ = QuantumCircuit(2)
circ.initialize([1,0,0,0])
circ.x(0)
