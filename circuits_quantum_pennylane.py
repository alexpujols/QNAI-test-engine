import pennylane as qml

def lightning_gpu_test():
    """
    This function attempts to load the 'lightning.gpu' device from PennyLane.
    If successful, it runs a simple quantum circuit to demonstrate functionality.
    """
    print("Attempting to load the 'lightning.gpu' device from PennyLane\n")

    try:
        # Run test script with GPU support
        #print("--- Loading lightning GPU device ---")
        #dev = qml.device("lightning.gpu", wires=2)
        
        # Run test script with CPU support
        print("--- Loading lightning CPU device ---")
        dev = qml.device("lightning.qubit", wires=2)
        
        print("--- Successfully loaded lightning.gpu device! ---\n")

        @qml.qnode(dev)
        def my_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        print("--- Running a sample circuit: ---\n")
        print(my_circuit())

    except Exception as e:
        print(f"Failed to load lightning.gpu device or run circuit: {e}")
        print("Please ensure your CUDA Toolkit, cuQuantum SDK, and GPU drivers are correctly installed and configured.")