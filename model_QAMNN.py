import numpy as np
import pennylane as qml

class QuantumAssociativeMemoryPennyLane:
    """
    A QAM for creative association tasks, implemented using PennyLane.
    Uses a Grover-like algorithm to query a superposition of memories.
    """
    def __init__(self, num_concepts):
        """
        Initializes the QAM.

        Args:
            num_concepts (int): The number of concepts (qubits).
        """
        self.num_concepts = num_concepts
        self.memory_states = []
        self.wires = list(range(num_concepts))
        print(f"✅ PennyLane QAM Initialized for {self.num_concepts} concepts.")

    def store_memories(self, memory_vectors):
        """
        Prepares memory states for circuit construction.
        """
        print(f"🧠 Storing {len(memory_vectors)} memories in PennyLane QAM...")
        self.memory_states = [v for v in memory_vectors]
        print("   - Memory states prepared for circuit construction.")

    def _oracle_operator(self, prompt_vector):
        """
        Marks memory states that have overlap with the prompt.
        """
        prompt_indices = np.where(prompt_vector)[0]
        for memory_vec in self.memory_states:
            memory_indices = np.where(memory_vec)[0]
            
            if any(idx in prompt_indices for idx in memory_indices):
                # Apply a phase flip to this memory state
                # qml.ctrl with PauliZ is a multi-controlled Z gate
                control_wires = [self.wires[i] for i in memory_indices]
                qml.ctrl(qml.PauliZ, control_wires=control_wires)(wires=self.wires[0])

    def _diffusion_operator(self):
        """
        The Grover diffusion operator.
        """
        qml.broadcast(qml.Hadamard, wires=self.wires, pattern="single")
        qml.broadcast(qml.PauliX, wires=self.wires, pattern="single")
        qml.ctrl(qml.PauliZ, control_wires=self.wires[:-1])(wires=self.wires[-1])
        qml.broadcast(qml.PauliX, wires=self.wires, pattern="single")
        qml.broadcast(qml.Hadamard, wires=self.wires, pattern="single")

    def query(self, prompt_vector, num_iterations=1, repetitions=100):
        """
        Queries the memory with a prompt vector.
        """
        print(f"\n🔍 Querying PennyLane QAM with prompt...")
        device = qml.device("lightning.gpu", wires=self.num_concepts, shots=repetitions)   # Uncomment this line to use the GPU device
        #device = qml.device("lightning.qubit", wires=self.num_concepts, shots=repetitions)  # Uncomment this line to use the CPU device

        @qml.qnode(device)
        def query_circuit():
            # 1. State Preparation (uniform superposition)
            qml.broadcast(qml.Hadamard, wires=self.wires, pattern="single")

            # 2. Grover Iterations
            for _ in range(num_iterations):
                self._oracle_operator(prompt_vector)
                self._diffusion_operator()
            
            # 3. Measurement
            return qml.sample()

        samples = query_circuit()
        
        # Process results
        samples_tuple = tuple(map(tuple, samples.numpy()))
        counts = {s: samples_tuple.count(s) for s in set(samples_tuple)}
        most_common_outcome = max(counts, key=counts.get)
        
        print("   - Query complete. Most frequent outcome selected.")
        return np.array(most_common_outcome)