import cirq
import numpy as np

# This function creates a simple quantum circuit that applies a square root of NOT gate (also known as a Hadamard gate) to a qubit and measures it.
def quantum_square_root_not_gate():
    print("\nCreating a sample quantum circuit that applies a square root of NOT gate to a qubit and measures it...")

    # Pick a qubit.
    qubit = cirq.GridQubit(0, 0)

    # Create a circuit that applies a square root of NOT gate, then measures the qubit.
    circuit = cirq.Circuit(cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m'))
    print("\nCircuit:")
    print(circuit)

    # Simulate the circuit several times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)
    print("\nResults:")
    print(result)