import numpy as np
import pennylane as qml

class QuantumHopfieldNetworkPennyLane:
    """
    A Quantum Hopfield Network (QHN) for pattern matching tasks,
    implemented using PennyLane with GPU support.
    """
    def __init__(self, num_neurons):
        """
        Initializes the QHN.

        Args:
            num_neurons (int): The number of neurons (qubits) in the network.
        """
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        print(f"✅ PennyLane QHNN Initialized with {self.num_neurons} neurons/qubits.")

    def store_patterns(self, patterns):
        """
        Stores patterns by computing the Hopfield weight matrix (Classical part).

        Args:
            patterns (list of np.array): List of 1D patterns with values +1 or -1.
        """
        print("🧠 Storing patterns in PennyLane QHNN memory...")
        num_patterns = len(patterns)
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= num_patterns
        print("   - Weight matrix computed.")

    def retrieve(self, noisy_pattern, repetitions=100):
        """
        Retrieves a pattern from a noisy input using a PennyLane QNode.

        Args:
            noisy_pattern (np.array): The 2D noisy pattern.
            repetitions (int): The number of shots for the simulation.

        Returns:
            np.array: The retrieved 2D pattern.
        """
        pattern_shape = noisy_pattern.shape
        flat_pattern = noisy_pattern.flatten()
        print(f"\n🔍 Retrieving pattern of shape {pattern_shape} with PennyLane...")

        # 1. Define the quantum device (high-performance GPU simulator)
        #device = qml.device("lightning.gpu", wires=self.num_neurons, shots=repetitions)
        device = qml.device("lightning.qubit", wires=self.num_neurons, shots=repetitions)

        # 2. Define the Quantum Node (QNode)
        @qml.qnode(device)
        def retrieval_circuit():
            # State Preparation: Encode pattern (-1 -> |1>, +1 -> |0>)
            for i in range(self.num_neurons):
                if flat_pattern[i] == -1:
                    qml.PauliX(wires=i)
            
            # Hopfield-inspired evolution using Ising ZZ interactions
            for i in range(self.num_neurons):
                for j in range(i + 1, self.num_neurons):
                    if self.weights[i, j] != 0:
                        angle = 2 * self.weights[i, j]
                        qml.IsingZZ(angle, wires=[i, j])

            # Return samples from the computational basis
            return qml.sample()

        # 3. Execute the QNode
        samples = retrieval_circuit()
        
        # 4. Process results to find the most common outcome
        # Convert samples to a tuple of tuples to be hashable for counting
        samples_tuple = tuple(map(tuple, samples.numpy()))
        counts = {s: samples_tuple.count(s) for s in set(samples_tuple)}
        most_common_outcome = max(counts, key=counts.get)
        
        # Convert bitstring (0s and 1s) back to pattern (+1s and -1s)
        retrieved_flat = np.array([1 if bit == 0 else -1 for bit in most_common_outcome])
        
        print("   - Retrieval complete. Most frequent outcome selected.")
        return retrieved_flat.reshape(pattern_shape)
