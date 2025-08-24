#!/usr/bin/env python
'''
                      ::::::
                    :+:  :+:
                   +:+   +:+
                  +#++:++#++:::::::
                 +#+     +#+     :+:
                #+#      #+#     +:+
               ###       ###+:++#""
                         +#+
                         #+#
                         ###
'''
__author__ = "Alex Pujols"
__copyright__ = "Alex Pujols"
__credits__ = ["Alex Pujols"]
__license__ = "MIT"
__version__ = "1.04-alpha"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title         : {Variational Quantum Neural Network for Adaptive Problem Solving}
Date          : {05-18-2025}
Description   : {This code implements a Variational Quantum Neural Network (VQNN) that uses Q-learning to train an agent to solve 10 different 5x5 maze navigation tasks, measuring emergent problem-solving behaviors through complexity and entropy analysis of the learned solution paths.}
Options       : {GPU acceleration via PennyLane-Lightning-GPU (NVIDIA cuQuantum SDK) or CPU fallback}
Dependencies  : {numpy scipy pennylane pennylane-lightning-gpu matplotlib
}
Requirements  : {Python 3.8+, Optional: CUDA 11.0+ and cuQuantum for GPU acceleration}
Usage         : {python run-ps-test.py}
Notes         : {Available at Github at https://github.com/alexpujols/QNAI-test-engine}
'''

import json
import csv
import os
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# QUANTUM BACKEND DETECTION AND CONFIGURATION
# ============================================================================

def detect_quantum_backend():
    """
    Detect and configure the best available quantum backend.
    Priority: PennyLane Lightning.GPU (NVIDIA) > Default.Qubit (CPU)
    """
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        
        # First, try NVIDIA GPU backend (requires pennylane-lightning-gpu)
        try:
            # Test if lightning.gpu is available
            test_dev = qml.device("lightning.gpu", wires=2)
            del test_dev  # Clean up test device
            
            print("=" * 60)
            print("✔ NVIDIA cuQuantum SDK detected!")
            print("✔ Using PennyLane Lightning.GPU for quantum acceleration")
            print("=" * 60)
            
            return {
                "available": True,
                "backend": "lightning.gpu",
                "interface": "autograd",
                "diff_method": "adjoint",
                "gpu": True,
                "pnp": pnp
            }
            
        except Exception as e:
            # Lightning.GPU not available, fall back to CPU
            print("=" * 60)
            print("ℹ NVIDIA GPU backend not available")
            print("ℹ Using CPU-based quantum simulation")
            print("ℹ For GPU acceleration, install: pip install pennylane-lightning-gpu")
            print("=" * 60)
            
            return {
                "available": True,
                "backend": "default.qubit",
                "interface": "autograd",
                "diff_method": "backprop",
                "gpu": False,
                "pnp": pnp
            }
            
    except ImportError:
        print("=" * 60)
        print("⚠ Warning: PennyLane not installed!")
        print("⚠ Using classical neural network fallback")
        print("⚠ Install PennyLane: pip install pennylane")
        print("=" * 60)
        
        return {
            "available": False,
            "backend": None,
            "interface": None,
            "diff_method": None,
            "gpu": False,
            "pnp": np
        }

# Initialize quantum backend
QUANTUM_CONFIG = detect_quantum_backend()
PENNYLANE_AVAILABLE = QUANTUM_CONFIG["available"]
QUANTUM_BACKEND = QUANTUM_CONFIG["backend"]
QUANTUM_INTERFACE = QUANTUM_CONFIG["interface"]
QUANTUM_DIFF_METHOD = QUANTUM_CONFIG["diff_method"]
GPU_AVAILABLE = QUANTUM_CONFIG["gpu"]
pnp = QUANTUM_CONFIG["pnp"]

# Import PennyLane if available
if PENNYLANE_AVAILABLE:
    import pennylane as qml

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Seed management for reproducibility
GLOBAL_SEED = None  # Set to integer for reproducibility, None for true randomness
QUANTUM_NOISE_LEVEL = 0.01  # Quantum noise parameter for realistic simulation

# Batch processing
BATCH_SIZE = 32  # Process multiple states simultaneously
PARALLEL_CIRCUITS = 4  # Number of parallel quantum circuits

# Maze constants
EMPTY = 0
WALL = 1
START = 2
GOAL = 3
VISITED = 4  # For visualization

# Actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DELTAS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(slots=True)
class MazeSolution:
    """Container for maze solution metrics."""
    maze_id: str
    steps_to_goal: int
    optimal_steps: int
    efficiency_score: float
    solution_path: List[Tuple[int, int]]
    action_sequence: List[str]
    learning_curve: List[float]
    final_reward: float
    convergence_episode: int
    performance_discontinuity: bool

# ============================================================================
# MAZE GENERATION MODULE
# ============================================================================

class MazeGenerator:
    """Generate and manage maze environments."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
    
    @staticmethod
    def get_fixed_mazes() -> List[Tuple[np.ndarray, str]]:
        """
        Generate 10 fixed 5x5 mazes with varying complexity.
        
        Returns:
            List of (maze, name) tuples
        """
        mazes = []
        
        # Maze 1: Simple corridor
        m1 = np.array([
            [2, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 0, 3],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1]
        ], dtype=np.float32)
        mazes.append((m1, "simple_corridor"))
        
        # Maze 2: Spiral
        m2 = np.array([
            [2, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 3]
        ], dtype=np.float32)
        mazes.append((m2, "spiral"))
        
        # Maze 3: Multiple paths
        m3 = np.array([
            [2, 0, 1, 0, 3],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1]
        ], dtype=np.float32)
        mazes.append((m3, "multiple_paths"))
        
        # Maze 4: Dead ends
        m4 = np.array([
            [2, 0, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 3]
        ], dtype=np.float32)
        mazes.append((m4, "dead_ends"))
        
        # Maze 5: Central barrier
        m5 = np.array([
            [2, 0, 0, 0, 1],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 3]
        ], dtype=np.float32)
        mazes.append((m5, "central_barrier"))
        
        # Maze 6: Zigzag
        m6 = np.array([
            [2, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [3, 0, 0, 1, 0]
        ], dtype=np.float32)
        mazes.append((m6, "zigzag"))
        
        # Maze 7: Open field with obstacles
        m7 = np.array([
            [2, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 3]
        ], dtype=np.float32)
        mazes.append((m7, "open_field"))
        
        # Maze 8: Narrow passages
        m8 = np.array([
            [2, 1, 0, 1, 1],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 3]
        ], dtype=np.float32)
        mazes.append((m8, "narrow_passages"))
        
        # Maze 9: Complex branching
        m9 = np.array([
            [2, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 3]
        ], dtype=np.float32)
        mazes.append((m9, "complex_branching"))
        
        # Maze 10: Deceptive path
        m10 = np.array([
            [2, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 3]
        ], dtype=np.float32)
        mazes.append((m10, "deceptive_path"))
        
        return mazes
    
    @staticmethod
    def find_position(maze: np.ndarray, marker: int) -> Tuple[int, int]:
        """Find position of a marker in the maze."""
        pos = np.where(maze == marker)
        return (pos[0][0], pos[1][0])
    
    @staticmethod
    def bfs_shortest_path(maze: np.ndarray) -> int:
        """
        Find shortest path using Breadth-First Search.
        
        Args:
            maze: Maze array
            
        Returns:
            Length of shortest path
        """
        from collections import deque
        
        start = MazeGenerator.find_position(maze, START)
        goal = MazeGenerator.find_position(maze, GOAL)
        
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (row, col), dist = queue.popleft()
            
            if (row, col) == goal:
                return dist
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < 5 and 0 <= new_col < 5 and
                    (new_row, new_col) not in visited and
                    maze[new_row, new_col] != WALL):
                    
                    visited.add((new_row, new_col))
                    queue.append(((new_row, new_col), dist + 1))
        
        return -1  # No path found

# ============================================================================
# VARIATIONAL QUANTUM NEURAL NETWORK
# ============================================================================

class VQNN:
    """
    Variational Quantum Neural Network for Q-learning.
    
    Implements:
    - Amplitude encoding for state representation
    - Two-layer variational circuit with RX, RY rotations and CNOT entanglement
    - Q-value prediction for action selection
    - GPU acceleration via PennyLane Lightning.GPU when available
    """
    
    def __init__(self, num_qubits: int = 25, num_layers: int = 2, 
                 learning_rate: float = 0.01):
        """
        Initialize VQNN with automatic GPU/CPU selection.
        
        Args:
            num_qubits: Number of qubits (25 for 5x5 maze)
            num_layers: Number of variational layers
            learning_rate: Learning rate for optimization
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # Initialize parameters
        self.params = pnp.random.randn(num_layers, num_qubits, 2) * 0.1
        
        # Adam optimizer state
        self.m = pnp.zeros_like(self.params)  # First moment
        self.v = pnp.zeros_like(self.params)  # Second moment
        self.t = 0  # Time step
        
        # Initialize quantum device based on availability
        if PENNYLANE_AVAILABLE:
            print(f"Initializing VQNN with {num_qubits} qubits...")
            
            if QUANTUM_BACKEND == "lightning.gpu":
                # NVIDIA GPU backend with cuQuantum
                self.dev = qml.device(
                    "lightning.gpu",
                    wires=num_qubits,
                    batch_obs=True  # Enable batched observables for GPU efficiency
                )
                print(f"  ✔ Using NVIDIA cuQuantum acceleration")
            else:
                # CPU backend
                self.dev = qml.device(
                    "default.qubit",
                    wires=num_qubits
                )
                print(f"  ✔ Using CPU quantum simulation")
            
            # Create quantum circuit
            self.circuit = self._create_circuit()
        else:
            self.dev = None
            self.circuit = None
            print("  ⚠ Using classical neural network fallback")
        
        self.rng = np.random.RandomState()
    
    def _create_circuit(self):
        """Create variational quantum circuit optimized for the backend."""
        
        @qml.qnode(
            self.dev, 
            interface=QUANTUM_INTERFACE,
            diff_method=QUANTUM_DIFF_METHOD
        )
        def circuit(inputs, params):
            # Amplitude encoding
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.num_qubits),
                normalize=True,
                pad_with=0.0
            )
            
            # Add quantum noise for realistic simulation
            if QUANTUM_NOISE_LEVEL > 0:
                for i in range(self.num_qubits):
                    qml.RY(QUANTUM_NOISE_LEVEL * self.rng.randn(), wires=i)
            
            # Variational layers
            for layer in range(self.num_layers):
                # Layer of single-qubit rotations
                for i in range(self.num_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                
                # Entangling layer (optimized pattern for GPU)
                if layer < self.num_layers - 1:
                    # Linear entanglement pattern (efficient for GPU)
                    for i in range(0, self.num_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(1, self.num_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
            
            # Measure expectation values for Q-values (4 actions)
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        return circuit
    
    def encode_state(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Encode maze state for quantum processing.
        
        Args:
            maze: Current maze
            position: Agent position
            
        Returns:
            Encoded state vector
        """
        state = pnp.zeros(25, dtype=np.float32)
        
        # Encode maze structure
        flat_maze = maze.flatten()
        for i in range(25):
            if flat_maze[i] == WALL:
                state[i] = -1.0
            elif flat_maze[i] == GOAL:
                state[i] = 0.5
            else:
                state[i] = 0.0
        
        # Encode agent position
        agent_idx = position[0] * 5 + position[1]
        state[agent_idx] = 1.0
        
        return state
    
    def get_q_values(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Get Q-values for state using quantum circuit.
        
        Args:
            maze: Current maze
            position: Agent position
            
        Returns:
            Q-values for each action
        """
        state = self.encode_state(maze, position)
        
        if PENNYLANE_AVAILABLE and self.circuit:
            # Quantum circuit evaluation
            q_values = pnp.array(self.circuit(state, self.params))
            return q_values
        else:
            # Classical fallback
            return self._classical_forward(state)
    
    def get_q_values_from_state(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values directly from encoded state."""
        if PENNYLANE_AVAILABLE and self.circuit:
            q_values = pnp.array(self.circuit(state, self.params))
            return q_values
        else:
            return self._classical_forward(state)
    
    def _classical_forward(self, state: np.ndarray) -> np.ndarray:
        """Classical neural network fallback."""
        weights = self.params.reshape(-1, 4)[:len(state)]
        q_values = pnp.tanh(pnp.dot(state, weights))
        return q_values
    
    def update(self, state: np.ndarray, action: int, target: float, 
               current_q: float) -> float:
        """
        Update network parameters using gradient descent.
        
        Args:
            state: Encoded state
            action: Action taken
            target: Target Q-value
            current_q: Current Q-value
            
        Returns:
            Loss value
        """
        self.t += 1
        loss = (target - current_q) ** 2
        
        # Compute gradient
        gradient = pnp.zeros_like(self.params)
        epsilon = 0.01
        
        # Numerical gradient computation
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                for param in range(2):
                    # Forward difference
                    self.params[layer, qubit, param] += epsilon
                    q_plus = self.get_q_values_from_state(state)[action]
                    
                    self.params[layer, qubit, param] -= 2 * epsilon
                    q_minus = self.get_q_values_from_state(state)[action]
                    
                    self.params[layer, qubit, param] += epsilon
                    
                    gradient[layer, qubit, param] = (q_plus - q_minus) / (2 * epsilon)
        
        # Adam optimizer update
        gradient *= 2 * (current_q - target)
        
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
        
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        
        self.params -= self.learning_rate * m_hat / (pnp.sqrt(v_hat) + eps)
        
        return float(loss)

# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ExperienceReplayBuffer:
    """Experience replay buffer for batch training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self, batch_size: int) -> List:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """
    Q-learning agent using VQNN.
    
    Implements:
    - Epsilon-greedy exploration
    - Experience replay with batch training
    """
    
    def __init__(self, vqnn: VQNN, epsilon: float = 0.1, 
                 gamma: float = 0.99, epsilon_decay: float = 0.995,
                 use_replay: bool = True, batch_size: int = 32):
        """
        Initialize Q-learning agent.
        
        Args:
            vqnn: VQNN for Q-value approximation
            epsilon: Exploration rate
            gamma: Discount factor
            epsilon_decay: Epsilon decay rate
            use_replay: Whether to use experience replay
            batch_size: Batch size for training
        """
        self.vqnn = vqnn
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.use_replay = use_replay
        self.batch_size = batch_size
        
        if use_replay:
            self.replay_buffer = ExperienceReplayBuffer(capacity=10000)
        
        self.rng = np.random.RandomState()
    
    def select_action(self, maze: np.ndarray, position: Tuple[int, int],
                     training: bool = True) -> Tuple[int, str]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            maze: Current maze
            position: Agent position
            training: Whether in training mode
            
        Returns:
            Action index and action name
        """
        if training and self.rng.random() < self.epsilon:
            # Exploration
            action_idx = self.rng.randint(4)
        else:
            # Exploitation
            q_values = self.vqnn.get_q_values(maze, position)
            action_idx = int(pnp.argmax(q_values))
        
        return action_idx, ACTIONS[action_idx]
    
    def train_step(self, maze: np.ndarray, position: Tuple[int, int],
                  action: int, reward: float, next_position: Tuple[int, int],
                  done: bool) -> float:
        """
        Perform one training step.
        
        Args:
            maze: Current maze
            position: Current position
            action: Action taken
            reward: Reward received
            next_position: Next position
            done: Whether episode ended
            
        Returns:
            Loss value
        """
        # Store experience
        if self.use_replay:
            state = self.vqnn.encode_state(maze, position)
            next_state = self.vqnn.encode_state(maze, next_position)
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train on batch if buffer is large enough
            if len(self.replay_buffer) >= self.batch_size:
                return self._train_batch()
        
        # Direct training (no replay)
        current_q_values = self.vqnn.get_q_values(maze, position)
        current_q = current_q_values[action]
        
        if done:
            target = reward
        else:
            next_q_values = self.vqnn.get_q_values(maze, next_position)
            target = reward + self.gamma * pnp.max(next_q_values)
        
        state = self.vqnn.encode_state(maze, position)
        loss = self.vqnn.update(state, action, target, current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def _train_batch(self) -> float:
        """Train on a batch of experiences from replay buffer."""
        batch = self.replay_buffer.sample_batch(self.batch_size)
        
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            # Compute current Q-value
            current_q_values = self.vqnn.get_q_values_from_state(state)
            current_q = current_q_values[action]
            
            # Compute target Q-value
            if done:
                target = reward
            else:
                next_q_values = self.vqnn.get_q_values_from_state(next_state)
                target = reward + self.gamma * pnp.max(next_q_values)
            
            # Update network
            loss = self.vqnn.update(state, action, target, current_q)
            total_loss += loss
        
        return total_loss / self.batch_size

# ============================================================================
# MAZE ENVIRONMENT
# ============================================================================

class MazeEnvironment:
    """
    Maze environment for agent interaction.
    
    Handles:
    - State transitions
    - Reward calculation
    - Episode management
    """
    
    def __init__(self, maze: np.ndarray):
        """Initialize environment with maze."""
        self.maze = maze.copy()
        self.original_maze = maze.copy()
        self.start_pos = MazeGenerator.find_position(maze, START)
        self.goal_pos = MazeGenerator.find_position(maze, GOAL)
        self.agent_pos = self.start_pos
        self.steps = 0
        self.max_steps = 100
        self.path = [self.start_pos]
        self.actions = []
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment to initial state."""
        self.maze = self.original_maze.copy()
        self.agent_pos = self.start_pos
        self.steps = 0
        self.path = [self.start_pos]
        self.actions = []
        return self.agent_pos
    
    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            Next position, reward, done flag
        """
        self.steps += 1
        self.actions.append(action)
        
        # Calculate new position
        dr, dc = ACTION_DELTAS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        
        # Check boundaries
        if not (0 <= new_row < 5 and 0 <= new_col < 5):
            # Hit boundary
            reward = -10
            done = self.steps >= self.max_steps
            return self.agent_pos, reward, done
        
        # Check for wall
        if self.maze[new_row, new_col] == WALL:
            # Hit wall
            reward = -10
            done = self.steps >= self.max_steps
            return self.agent_pos, reward, done
        
        # Valid move
        self.agent_pos = (new_row, new_col)
        self.path.append(self.agent_pos)
        
        # Check for goal
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
        else:
            reward = -1  # Step cost
            done = self.steps >= self.max_steps
        
        return self.agent_pos, reward, done

# ============================================================================
# COMPLEXITY METRICS
# ============================================================================

@lru_cache(maxsize=512)
def lz_complexity(s: str) -> int:
    """Calculate Lempel-Ziv complexity of a string."""
    if len(s) <= 1:
        return len(s)
    
    n = len(s)
    i = 0
    c = 0
    dictionary = set()
    
    while i < n:
        j = i + 1
        while j <= n and s[i:j] in dictionary:
            j += 1
        if j <= n:
            dictionary.add(s[i:j])
            c += 1
            i = j
        else:
            if s[i:] not in dictionary:
                c += 1
            break
    
    return c

@lru_cache(maxsize=512)
def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1
    
    n = len(s)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * np.log2(p)
    
    return entropy

def approximate_entropy(U: List[Any], m: int = 2, r: float = 0.2) -> float:
    """Calculate approximate entropy of a sequence."""
    if len(U) < m:
        return 0.0
    
    def _maxdist(xi, xj, m):
        return max([abs(float(a) - float(b)) for a, b in zip(xi, xj)])
    
    def _phi(m):
        patterns = [U[i:i + m] for i in range(len(U) - m + 1)]
        C = []
        for i, template in enumerate(patterns):
            matches = sum(1 for j, pattern in enumerate(patterns)
                        if _maxdist(template, pattern, m) <= r)
            C.append(matches / (len(U) - m + 1))
        return sum(np.log(c) for c in C if c > 0) / (len(U) - m + 1)
    
    try:
        return _phi(m) - _phi(m + 1)
    except:
        return 0.0

# ============================================================================
# EMERGENCE DETECTION
# ============================================================================

def detect_emergence(solution: MazeSolution, baseline_steps: Dict[str, int],
                    learning_curve: List[float]) -> Dict[str, Any]:
    """
    Detect emergent behavior in maze solving.
    
    Args:
        solution: Solution metrics
        baseline_steps: Baseline performance (random, optimal)
        learning_curve: Reward per episode
        
    Returns:
        Dictionary with emergence assessment
    """
    # Calculate efficiency relative to optimal
    efficiency = solution.efficiency_score
    
    # Detect performance discontinuity
    if len(learning_curve) > 10:
        # Look for sudden jumps in performance
        window = 5
        for i in range(window, len(learning_curve) - window):
            before = np.mean(learning_curve[i-window:i])
            after = np.mean(learning_curve[i:i+window])
            if after - before > 30:  # Significant jump
                solution.performance_discontinuity = True
                solution.convergence_episode = i
                break
    
    # Emergence criteria
    is_emergent = False
    emergence_type = "none"
    emergence_score = 0.0
    
    # Perfect or near-perfect efficiency
    if efficiency >= 0.95:
        is_emergent = True
        emergence_type = "perfect_navigation"
        emergence_score = 1.0
    elif efficiency >= 0.85:
        is_emergent = True
        emergence_type = "efficient_navigation"
        emergence_score = 0.8
    
    # Performance discontinuity indicates emergent insight
    if solution.performance_discontinuity:
        is_emergent = True
        if emergence_type == "none":
            emergence_type = "sudden_insight"
        emergence_score = max(emergence_score, 0.7)
    
    # Compare to baseline
    random_improvement = (baseline_steps["random"] - solution.steps_to_goal) / baseline_steps["random"]
    if random_improvement > 0.8:
        is_emergent = True
        if emergence_type == "none":
            emergence_type = "intelligent_navigation"
        emergence_score = max(emergence_score, 0.6)
    
    return {
        "is_emergent": is_emergent,
        "emergence_type": emergence_type,
        "emergence_score": emergence_score,
        "efficiency_score": efficiency,
        "performance_discontinuity": solution.performance_discontinuity,
        "convergence_episode": solution.convergence_episode
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_solution(maze: np.ndarray, path: List[Tuple[int, int]], 
                      maze_name: str, save_path: Optional[str] = None):
    """
    Visualize the solution path in the maze.
    
    Args:
        maze: Maze array
        path: Solution path
        maze_name: Name of the maze
        save_path: Optional path to save visualization
    """
    visual = maze.copy().astype(float)
    
    # Mark path
    for i, (row, col) in enumerate(path[1:-1], 1):
        visual[row, col] = 0.5 + (i / len(path)) * 0.3
    
    # Create text representation
    print(f"\n{maze_name} Solution:")
    print("-" * 15)
    
    for row in range(5):
        line = ""
        for col in range(5):
            if (row, col) in path[1:-1]:
                line += "o "
            elif visual[row, col] == START:
                line += "S "
            elif visual[row, col] == GOAL:
                line += "G "
            elif visual[row, col] == WALL:
                line += "█ "
            else:
                line += ". "
        print(line)
    
    print(f"Path length: {len(path) - 1} steps")

# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_experiments(config: Optional[Dict] = None):
    """
    Execute adaptive problem-solving experiments.
    
    Args:
        config: Experimental configuration dictionary
        
    Returns:
        Results and validation metrics
    """
    if config is None:
        config = {
            "episodes_per_maze": 150,
            "epsilon_start": 0.3,
            "epsilon_decay": 0.995,
            "learning_rate": 0.01,
            "gamma": 0.99,
            "use_replay": True,
            "batch_size": 32,
            "output_dir": "results",
            "visualize": True
        }
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    print("\n" + "=" * 60)
    print("VQNN ADAPTIVE PROBLEM-SOLVING EXPERIMENTS")
    print("=" * 60)
    print(f"Quantum Backend: {QUANTUM_BACKEND if PENNYLANE_AVAILABLE else 'Classical Fallback'}")
    print(f"GPU Acceleration: {'ENABLED (NVIDIA cuQuantum)' if GPU_AVAILABLE else 'DISABLED'}")
    print(f"Episodes per maze: {config['episodes_per_maze']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Experience replay: {'ENABLED' if config['use_replay'] else 'DISABLED'}")
    print(f"Epsilon: {config['epsilon_start']} (decay: {config['epsilon_decay']})")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Seed: {GLOBAL_SEED or 'True random'}")
    print("=" * 60)
    print()
    
    # Initialize components
    generator = MazeGenerator(seed=GLOBAL_SEED)
    mazes = generator.get_fixed_mazes()
    
    # Output setup
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], f"vqnn_results_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "maze_name", "maze_complexity",
        "episodes_trained", "steps_to_goal", "optimal_steps", "efficiency_score",
        "final_reward", "convergence_episode", "performance_discontinuity",
        "path_length", "path_lz_complexity", "path_shannon_entropy",
        "action_sequence_length", "action_lz_complexity", "action_shannon_entropy",
        "action_approximate_entropy", "is_emergent", "emergence_type", 
        "emergence_score", "solution_path", "action_sequence",
        "quantum_backend", "gpu_accelerated"
    ]
    
    results = []
    run_id = 0
    
    print("Running experiments...\n")
    
    # Test each maze
    for maze, maze_name in mazes:
        run_id += 1
        print(f"\nMaze {run_id}: {maze_name}")
        print("-" * 40)
        
        # Calculate optimal path length
        optimal_steps = generator.bfs_shortest_path(maze)
        print(f"  Optimal path: {optimal_steps} steps")
        
        # Calculate baseline (random walk)
        random_steps = optimal_steps * 5  # Rough estimate
        
        # Initialize VQNN and agent
        vqnn = VQNN(
            num_qubits=25,
            num_layers=2,
            learning_rate=config["learning_rate"]
        )
        
        agent = QLearningAgent(
            vqnn=vqnn,
            epsilon=config["epsilon_start"],
            gamma=config["gamma"],
            epsilon_decay=config["epsilon_decay"],
            use_replay=config["use_replay"],
            batch_size=config["batch_size"]
        )
        
        # Training
        env = MazeEnvironment(maze)
        learning_curve = []
        best_solution = None
        best_steps = float('inf')
        
        print("  Training...")
        for episode in range(config["episodes_per_maze"]):
            position = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action_idx, action = agent.select_action(maze, position, training=True)
                next_position, reward, done = env.step(action)
                
                # Train agent
                loss = agent.train_step(maze, position, action_idx, reward, 
                                       next_position, done)
                
                position = next_position
                episode_reward += reward
            
            learning_curve.append(episode_reward)
            
            # Track best solution
            if done and position == env.goal_pos and env.steps < best_steps:
                best_steps = env.steps
                best_solution = MazeSolution(
                    maze_id=maze_name,
                    steps_to_goal=env.steps,
                    optimal_steps=optimal_steps,
                    efficiency_score=optimal_steps / env.steps if env.steps > 0 else 0,
                    solution_path=env.path.copy(),
                    action_sequence=env.actions.copy(),
                    learning_curve=learning_curve.copy(),
                    final_reward=episode_reward,
                    convergence_episode=episode,
                    performance_discontinuity=False
                )
            
            if episode % 20 == 0:
                avg_reward = np.mean(learning_curve[-10:]) if len(learning_curve) >= 10 else episode_reward
                print(f"    Episode {episode}: Avg reward = {avg_reward:.1f}, Epsilon = {agent.epsilon:.3f}")
        
        # Final evaluation (deterministic)
        print("  Evaluating...")
        agent.epsilon = 0  # Disable exploration
        env.reset()
        position = env.start_pos
        done = False
        
        while not done:
            action_idx, action = agent.select_action(maze, position, training=False)
            position, reward, done = env.step(action)
        
        # Create final solution if needed
        if best_solution is None or env.steps < best_solution.steps_to_goal:
            best_solution = MazeSolution(
                maze_id=maze_name,
                steps_to_goal=env.steps,
                optimal_steps=optimal_steps,
                efficiency_score=optimal_steps / env.steps if env.steps > 0 else 0,
                solution_path=env.path,
                action_sequence=env.actions,
                learning_curve=learning_curve,
                final_reward=reward,
                convergence_episode=len(learning_curve),
                performance_discontinuity=False
            )
        
        # Complexity analysis
        path_str = ''.join([f"{r}{c}" for r, c in best_solution.solution_path])
        action_str = ''.join([a[0] for a in best_solution.action_sequence])
        
        path_lz = lz_complexity(path_str)
        path_entropy = shannon_entropy(path_str)
        action_lz = lz_complexity(action_str)
        action_entropy = shannon_entropy(action_str)
        
        # Convert actions to numeric for approximate entropy
        action_numeric = [ACTIONS.index(a) for a in best_solution.action_sequence]
        action_apen = approximate_entropy(action_numeric) if len(action_numeric) > 2 else 0
        
        # Emergence detection
        baselines = {"random": random_steps, "optimal": optimal_steps}
        emergence = detect_emergence(best_solution, baselines, learning_curve)
        
        print(f"  Final: {best_solution.steps_to_goal} steps, "
              f"Efficiency: {best_solution.efficiency_score:.2f}, "
              f"Emergent: {emergence['is_emergent']}")
        
        # Visualize solution
        if config.get("visualize", True):
            visualize_solution(maze, best_solution.solution_path, maze_name)
        
        # Save results
        results.append([
            f"VQNN_maze_{run_id}",
            datetime.now().isoformat(),
            maze_name,
            len(best_solution.solution_path),  # Maze complexity proxy
            config["episodes_per_maze"],
            best_solution.steps_to_goal,
            optimal_steps,
            best_solution.efficiency_score,
            best_solution.final_reward,
            best_solution.convergence_episode,
            best_solution.performance_discontinuity,
            len(best_solution.solution_path),
            path_lz,
            path_entropy,
            len(best_solution.action_sequence),
            action_lz,
            action_entropy,
            action_apen,
            emergence["is_emergent"],
            emergence["emergence_type"],
            emergence["emergence_score"],
            str(best_solution.solution_path),
            str(best_solution.action_sequence),
            QUANTUM_BACKEND if PENNYLANE_AVAILABLE else "classical",
            GPU_AVAILABLE
        ])
    
    # Save results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_emergent = sum(1 for r in results if r[18])  # is_emergent column
    avg_efficiency = np.mean([r[7] for r in results])  # efficiency_score column
    
    print(f"Total mazes: {len(results)}")
    print(f"Emergent solutions: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    print(f"Average efficiency: {avg_efficiency:.3f}")
    print(f"Quantum backend: {QUANTUM_BACKEND if PENNYLANE_AVAILABLE else 'Classical'}")
    print(f"GPU acceleration: {'Yes (NVIDIA cuQuantum)' if GPU_AVAILABLE else 'No'}")
    
    # Detailed breakdown by emergence type
    emergence_types = {}
    for r in results:
        if r[18]:  # is_emergent
            etype = r[19]  # emergence_type
            emergence_types[etype] = emergence_types.get(etype, 0) + 1
    
    if emergence_types:
        print("\nEmergence types:")
        for etype, count in emergence_types.items():
            print(f"  {etype}: {count}")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration for experiments
    config = {
        "episodes_per_maze": 150,   # Training episodes
        "epsilon_start": 0.3,        # Initial exploration rate
        "epsilon_decay": 0.995,      # Exploration decay
        "learning_rate": 0.01,       # VQNN learning rate
        "gamma": 0.99,               # Discount factor
        "use_replay": True,          # Enable experience replay
        "batch_size": 32,            # Batch size for processing
        "output_dir": "results",
        "visualize": True
    }
    
    results = run_experiments(config)
    
    print("\n✅ Experiment complete!")
    print("=" * 60)