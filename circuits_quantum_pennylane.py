import pennylane as qml
import numpy as np

def lightning_gpu_test():
    """
    This function attempts to load the 'lightning.qubit' device from PennyLane.
    If successful, it runs a simple quantum circuit, formats the state output,
    and also draws the circuit.
    """
    print("\nAttempting to load the '*.qubit' device from PennyLane")

    try:
        # Run test script with GPU support
        #print("--- Loading lightning GPU device ---")
        #dev = qml.device("lightning.gpu", wires=2)
        
        # Run test script with CPU support
        print("\n--- Loading lightning CPU device ---")
        dev = qml.device("lightning.qubit", wires=2)
        
        print("\n--- Successfully loaded lightning.qubit device! ---")

        @qml.qnode(dev)
        def my_circuit():
            qml.Hadamard(wires=0) # Qubit 0 is now in |+⟩ state (superposition of 0 and 1)
            qml.CNOT(wires=[0, 1]) # This creates a Bell state: (|00⟩ + |11⟩) / sqrt(2)
            return qml.state()

        print("\n--- Running a sample circuit: ---")
        state_vector = my_circuit()
        print(f"\n--- Show raw vector output: ---")
        print(f"\nRaw state vector: {state_vector}")

        # Format the output for readability
        num_wires = len(dev.wires)
        print(f"\n--- Show state vector amplitudes for {num_wires} qubits (Basis State |amplitude|²): ---\n")
        for i, amplitude in enumerate(state_vector):
            # Convert integer index to binary string representation
            # For 2 wires, '02b' ensures a 2-bit binary string (e.g., 0 -> "00", 1 -> "01")
            basis_state = format(i, f'0{num_wires}b')
            probability = np.abs(amplitude)**2
            print(f"|{basis_state}⟩: {amplitude:.4f} (Probability: {probability:.4f})")
        
        # --- Draw the circuit ---
        print("\n--- Drawing the quantum circuit: ---\n")
        # qml.draw() returns a function that, when called with the QNode's arguments,
        # will print the circuit. Since my_circuit has no arguments, call it with ().
        print(qml.draw(my_circuit)())

    except Exception as e:
        print(f"Failed to load lightning.qubit device or run circuit: {e}")
        print("Please ensure PennyLane-Lightning is correctly installed.")

