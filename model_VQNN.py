import numpy as np
import pennylane as qml
import tensorflow as tf

class VariationalQuantumAgentPennyLane:
    """
    A VQNN agent for reinforcement learning, implemented using PennyLane's
    KerasLayer for seamless TensorFlow integration.
    """
    def __init__(self, maze_size, num_actions=4, learning_rate=0.01):
        """
        Initializes the VQNN agent.

        Args:
            maze_size (tuple): The (height, width) of the maze.
            num_actions (int): The number of possible actions.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.num_qubits = maze_size[0] * maze_size[1]
        self.num_actions = num_actions
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        print(f"✅ PennyLane VQNN Agent Initialized for a {maze_size} maze.")

    def _build_model(self):
        """
        Builds the Keras model with an embedded PennyLane QNode.
        """
        # 1. Define the quantum device
        # Use shots=None for exact expectation values during training
        #device = qml.device("lightning.gpu", wires=self.num_qubits, shots=None)
        device = qml.device("lightning.qubit", wires=self.num_qubits, shots=None)

        # 2. Define the PQC ansatz as a QNode
        @qml.qnode(device, interface="tf")
        def pqc(inputs, weights):
            # Data Encoding Layer
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True)
            
            # Variational Layers (weights are trainable)
            num_layers = 2
            for layer in range(num_layers):
                for i in range(self.num_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Return expectation values for each action
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_actions)]

        # 3. Define the shape of the trainable weights
        weight_shapes = {"weights": (2, self.num_qubits, 2)}
        
        # 4. Create the Keras model
        qlayer = qml.qnn.KerasLayer(pqc, weight_shapes, output_dim=self.num_actions)
        model = tf.keras.models.Sequential([qlayer])
        
        return model

    def choose_action(self, state, epsilon=0.1):
        """
        Chooses an action based on the current state.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            # The KerasLayer expects a batch, so add a batch dimension
            state_batch = np.expand_dims(state.flatten(), axis=0)
            action_q_values = self.model(state_batch)[0]
            return tf.argmax(action_q_values).numpy()

    def train(self, state, action, reward, next_state, gamma=0.99):
        """
        Performs a single training step.
        """
        # Add batch dimension for Keras model
        state_batch = np.expand_dims(state.flatten(), axis=0)
        next_state_batch = np.expand_dims(next_state.flatten(), axis=0)
        
        with tf.GradientTape() as tape:
            # Predict Q-values for the future state to calculate the target
            future_q_values = self.model(next_state_batch)[0]
            target_q = reward + gamma * tf.reduce_max(future_q_values)
            
            # Predict Q-values for the current state
            current_q_values = self.model(state_batch)[0]
            action_q_value = current_q_values[action]
            
            # Calculate loss (Mean Squared Error)
            loss = tf.math.square(action_q_value - target_q)

        # Compute and apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))