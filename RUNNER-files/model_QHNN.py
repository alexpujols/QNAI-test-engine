import numpy as np
import pennylane as qml

class QuantumHopfieldNetworkPennyLane:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        # Removed print statement for batch processing

    def store_patterns(self, patterns):
        num_patterns = len(patterns)
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= num_patterns

    def retrieve(self, noisy_pattern, original_pattern, repetitions=100):
        """
        Retrieves a pattern and calculates accuracy.

        Args:
            noisy_pattern (np.array): The 2D noisy input pattern.
            original_pattern (np.array): The 2D original pattern for accuracy check.
            repetitions (int): The number of shots for the simulation.

        Returns:
            dict: A dictionary containing the retrieved pattern, accuracy, and raw bitstring.
        """
        pattern_shape = noisy_pattern.shape
        flat_pattern = noisy_pattern.flatten()
        original_flat = original_pattern.flatten()

        device = qml.device("lightning.gpu", wires=self.num_neurons, shots=repetitions)

        @qml.qnode(device)
        def retrieval_circuit():
            for i in range(self.num_neurons):
                if flat_pattern[i] == -1:
                    qml.PauliX(wires=i)
            
            for i in range(self.num_neurons):
                for j in range(i + 1, self.num_neurons):
                    if self.weights[i, j] != 0:
                        angle = 2 * self.weights[i, j]
                        qml.IsingZZ(angle, wires=[i, j])

            return qml.sample()

        samples = retrieval_circuit()
        samples_tuple = tuple(map(tuple, samples))
        counts = {s: samples_tuple.count(s) for s in set(samples_tuple)}
        most_common_outcome = max(counts, key=counts.get)
        
        retrieved_flat = np.array([1 if bit == 0 else -1 for bit in most_common_outcome])
        
        # Calculate prediction accuracy
        accuracy = np.mean(retrieved_flat == original_flat)
        
        return {
            "retrieved_pattern": retrieved_flat.reshape(pattern_shape),
            "prediction_accuracy": accuracy,
            "raw_output_bitstring": most_common_outcome
        }