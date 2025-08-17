import numpy as np
import pennylane as qml

class QuantumAssociativeMemoryPennyLane:
    def __init__(self, num_concepts):
        self.num_concepts = num_concepts
        self.memory_states = []
        self.wires = list(range(num_concepts))

    def store_memories(self, memory_vectors):
        self.memory_states = [v for v in memory_vectors]

    def _oracle_operator(self, prompt_vector):
        prompt_indices = np.where(prompt_vector)[0]
        for memory_vec in self.memory_states:
            memory_indices = np.where(memory_vec)[0]
            if any(idx in prompt_indices for idx in memory_indices):
                control_wires = [self.wires[i] for i in memory_indices]
                qml.ctrl(qml.PauliZ, control_wires=control_wires)(wires=self.wires[0])

    def _diffusion_operator(self):
        qml.broadcast(qml.Hadamard, wires=self.wires, pattern="single")
        qml.broadcast(qml.PauliX, wires=self.wires, pattern="single")
        qml.ctrl(qml.PauliZ, control_wires=self.wires[:-1])(wires=self.wires[-1])
        qml.broadcast(qml.PauliX, wires=self.wires, pattern="single")
        qml.broadcast(qml.Hadamard, wires=self.wires, pattern="single")

    def query(self, prompt_vector, concept_map, num_iterations=1, repetitions=100):
        """
        Queries the memory and decodes the results.

        Args:
            prompt_vector (np.array): The input prompt vector.
            concept_map (dict): The mapping from concept name to index.
            num_iterations (int): Number of Grover iterations.
            repetitions (int): Number of shots for simulation.

        Returns:
            dict: A dictionary containing decoded concepts and the raw output vector.
        """
        device = qml.device("lightning.gpu", wires=self.num_concepts, shots=repetitions)
        @qml.qnode(device)
        def query_circuit():
            qml.broadcast(qml.Hadamard, wires=self.wires, pattern="single")
            for _ in range(num_iterations):
                self._oracle_operator(prompt_vector)
                self._diffusion_operator()
            return qml.sample()

        samples = query_circuit()
        samples_tuple = tuple(map(tuple, samples.numpy()))
        counts = {s: samples_tuple.count(s) for s in set(samples_tuple)}
        most_common_outcome = max(counts, key=counts.get)
        output_vector = np.array(most_common_outcome)
        
        # Decode concepts
        index_to_concept = {v: k for k, v in concept_map.items()}
        prompt_concepts = [index_to_concept[i] for i, bit in enumerate(prompt_vector) if bit == 1]
        output_concepts = [index_to_concept[i] for i, bit in enumerate(output_vector) if bit == 1]
        
        return {
            "prompt_concepts": prompt_concepts,
            "output_concepts": output_concepts,
            "raw_output_vector": output_vector
        }