import numpy as np
import pennylane as qml
import tensorflow as tf
import time

class VariationalQuantumAgentPennyLane:
    def __init__(self, maze_size, num_actions=4, learning_rate=0.01):
        self.num_qubits = maze_size[0] * maze_size[1]
        self.maze_size = maze_size
        self.num_actions = num_actions
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_model(self):
        device = qml.device("lightning.gpu", wires=self.num_qubits, shots=None)
        @qml.qnode(device, interface="tf")
        def pqc(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True)
            num_layers = 2
            for layer in range(num_layers):
                for i in range(self.num_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_actions)]

        weight_shapes = {"weights": (2, self.num_qubits, 2)}
        qlayer = qml.qnn.KerasLayer(pqc, weight_shapes, output_dim=self.num_actions)
        return tf.keras.models.Sequential([qlayer])

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            state_batch = np.expand_dims(state.flatten(), axis=0)
            action_q_values = self.model(state_batch)[0]
            return tf.argmax(action_q_values).numpy()

    def _train_step(self, state, action, reward, next_state, gamma=0.99):
        state_batch = np.expand_dims(state.flatten(), axis=0)
        next_state_batch = np.expand_dims(next_state.flatten(), axis=0)
        with tf.GradientTape() as tape:
            future_q_values = self.model(next_state_batch)[0]
            target_q = reward + gamma * tf.reduce_max(future_q_values)
            current_q_values = self.model(state_batch)[0]
            action_q_value = current_q_values[action]
            loss = tf.math.square(action_q_value - target_q)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_and_evaluate(self, maze, start_pos, goal_pos, episodes, max_steps_per_episode=None):
        """Encapsulates the training loop and returns final performance."""
        if max_steps_per_episode is None:
            max_steps_per_episode = self.maze_size[0] * self.maze_size[1] * 2

        start_time = time.time()
        best_path = []
        min_steps = float('inf')

        for episode in range(episodes):
            agent_pos = start_pos
            path = [start_pos]
            done = False
            for step in range(max_steps_per_episode):
                current_state = maze.copy()
                current_state[agent_pos] = 4
                action = self.choose_action(current_state)
                
                next_pos = list(agent_pos)
                if action == 0: next_pos[0] -= 1
                elif action == 1: next_pos[0] += 1
                elif action == 2: next_pos[1] -= 1
                elif action == 3: next_pos[1] += 1

                if tuple(next_pos) == goal_pos:
                    reward = 100
                    done = True
                elif not (0 <= next_pos[0] < self.maze_size[0] and 0 <= next_pos[1] < self.maze_size[1] and maze[tuple(next_pos)] != 1):
                    reward = -10
                    next_pos = agent_pos
                else:
                    reward = -1

                next_state = maze.copy()
                next_state[tuple(next_pos)] = 4
                self._train_step(current_state, action, reward, next_state)
                agent_pos = tuple(next_pos)
                path.append(agent_pos)

                if done:
                    if (step + 1) < min_steps:
                        min_steps = step + 1
                        best_path = path
                    break
        
        end_time = time.time()
        
        return {
            "steps_to_goal": min_steps if min_steps != float('inf') else -1,
            "time_to_goal_s": round(end_time - start_time, 2),
            "final_reward": reward if done else -1, # Reward from the final successful step
            "solution_path": best_path
        }