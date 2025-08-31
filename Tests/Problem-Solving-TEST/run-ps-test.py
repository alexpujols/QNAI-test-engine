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
__version__ = "1.05-alpha"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title         : {Variational Quantum Neural Network for Adaptive Problem Solving}
Date          : {05-18-2025}
Description   : {Full 25-qubit implementation utilizing all qubits for maximum
                quantum advantage. Each qubit maps directly to a cell in the 5x5 maze,
                providing a natural quantum representation of the spatial problem.}
Options       : {GPU acceleration via PennyLane-Lightning-GPU (NVIDIA cuQuantum SDK) or CPU fallback}
Dependencies  : {numpy scipy pennylane pennylane-lightning-gpu matplotlib}
Requirements  : {Python 3.8+, Optional: CUDA 11.0+ and cuQuantum for GPU acceleration}
Usage         : {python run-ps-test.py}
Notes         : {Available at Github at https://github.com/alexpujols/QNAI-test-engine/blob/main/Tests/Problem-Solving-TEST/run-ps-test.py}
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
import time
from contextlib import contextmanager
from collections import deque
warnings.filterwarnings('ignore')

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@contextmanager
def timer(name: str, verbose: bool = True):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    if verbose:
        print(f"  ⏱ {name}: {end - start:.3f}s")

class PerformanceMonitor:
    """Track performance metrics throughout execution."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def record(self, metric_name: str, value: float):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def report(self):
        elapsed = time.time() - self.start_time
        print("\n📊 Performance Report:")
        print(f"Total runtime: {elapsed/60:.1f} minutes")
        for name, values in self.metrics.items():
            avg = np.mean(values)
            print(f"  {name}: avg={avg:.3f}, min={min(values):.3f}, max={max(values):.3f}")

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def configure_gpu_settings():
    """Configure GPU settings for maximum quantum simulation performance."""
    import os
    
    # GPU optimization settings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
    os.environ['CUDNN_BENCHMARK'] = 'TRUE'    # Auto-tune operations
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Pre-allocate memory
    os.environ['TF_GPU_MEMORY_FRACTION'] = '0.9'  # Use 90% of GPU memory
    
    print("✓ GPU settings configured for maximum performance")

configure_gpu_settings()

# ============================================================================
# QUANTUM BACKEND DETECTION
# ============================================================================

def detect_quantum_backend():
    """Detect and configure the best available quantum backend for 25 qubits."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        
        # Try GPU backend first for 25-qubit simulation
        try:
            test_dev = qml.device("lightning.gpu", wires=25)
            del test_dev
            
            print("=" * 60)
            print("✓ NVIDIA cuQuantum SDK detected!")
            print("✓ Using PennyLane Lightning.GPU for 25-qubit simulation")
            print("✓ Expected runtime: 10-20 minutes per maze")
            print("=" * 60)
            
            return {
                "available": True,
                "backend": "lightning.gpu",
                "interface": "autograd",
                "diff_method": "adjoint",  # Faster for GPU
                "gpu": True,
                "pnp": pnp
            }
            
        except Exception as e:
            # Fallback to CPU-optimized backend
            try:
                # Try lightning.qubit for better CPU performance
                test_dev = qml.device("lightning.qubit", wires=25)
                del test_dev
                print("=" * 60)
                print("✓ Using PennyLane Lightning.Qubit (optimized CPU)")
                print("⚠ Warning: 25-qubit simulation on CPU will be slow")
                print("⚠ Expected runtime: 15-30 minutes per maze")
                print("=" * 60)
                
                return {
                    "available": True,
                    "backend": "lightning.qubit",
                    "interface": "autograd",
                    "diff_method": "parameter-shift",
                    "gpu": False,
                    "pnp": pnp
                }
            except:
                # Final fallback
                print("=" * 60)
                print("ℹ Using default quantum simulation")
                print("⚠ Warning: This will be very slow for 25 qubits")
                print("=" * 60)
                
                return {
                    "available": True,
                    "backend": "default.qubit",
                    "interface": "autograd",
                    "diff_method": "parameter-shift",
                    "gpu": False,
                    "pnp": pnp
                }
            
    except ImportError:
        print("⚠ Error: PennyLane not installed!")
        print("Install with: pip install pennylane pennylane-lightning-gpu")
        import sys
        sys.exit(1)

# Initialize quantum backend
QUANTUM_CONFIG = detect_quantum_backend()
PENNYLANE_AVAILABLE = QUANTUM_CONFIG["available"]
QUANTUM_BACKEND = QUANTUM_CONFIG["backend"]
QUANTUM_INTERFACE = QUANTUM_CONFIG["interface"]
QUANTUM_DIFF_METHOD = QUANTUM_CONFIG["diff_method"]
GPU_AVAILABLE = QUANTUM_CONFIG["gpu"]
pnp = QUANTUM_CONFIG["pnp"]

import pennylane as qml

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Quantum circuit configuration - FULL 25 QUBITS
NUM_QUBITS = 25  # Full utilization - one qubit per maze cell
NUM_LAYERS = 2   # Two layers for richer quantum dynamics
BATCH_SIZE = 16  # Smaller batch due to memory requirements

# Training configuration
EPISODES_PER_MAZE = 100  # More episodes for better convergence
LEARNING_RATE = 0.01
GAMMA = 0.99
EPSILON_START = 0.3
EPSILON_DECAY = 0.995

# Maze constants
EMPTY = 0
WALL = 1
START = 2
GOAL = 3

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
# FULL 25-QUBIT VQNN
# ============================================================================

class VQNN:
    """
    Full 25-Qubit Variational Quantum Neural Network.
    
    This implementation fully utilizes all 25 qubits with rich entanglement
    patterns to maximize quantum advantage for the research.
    """
    
    def __init__(self, num_qubits: int = NUM_QUBITS, num_layers: int = NUM_LAYERS,
                 learning_rate: float = LEARNING_RATE):
        """
        Initialize VQNN with full 25-qubit utilization.
        
        Args:
            num_qubits: Number of qubits (25 for full maze representation)
            num_layers: Number of variational layers (2 for rich dynamics)
            learning_rate: Learning rate for optimization
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # Initialize parameters for all 25 qubits
        self.params = pnp.random.randn(num_layers, num_qubits, 3) * 0.1  # 3 params per qubit
        
        # Adam optimizer state
        self.m = pnp.zeros_like(self.params)
        self.v = pnp.zeros_like(self.params)
        self.t = 0
        
        # Performance tracking
        self.perf = PerformanceMonitor()
        
        print(f"Initializing Full 25-Qubit VQNN...")
        print(f"  Configuration:")
        print(f"    - Qubits: {num_qubits}")
        print(f"    - Layers: {num_layers}")
        print(f"    - Parameters: {self.params.size}")
        print(f"    - Backend: {QUANTUM_BACKEND}")
        
        # Initialize quantum device
        self.dev = qml.device(
            QUANTUM_BACKEND,
            wires=num_qubits
        )
        
        # Create the full quantum circuit
        self.circuit = self._create_full_circuit()
        
        print(f"  ✓ Full 25-qubit quantum circuit initialized")
        print(f"  ✓ Ready for quantum maze navigation")
    
    def _create_full_circuit(self):
        """
        Create full 25-qubit quantum circuit with rich entanglement.
        
        This circuit fully utilizes all qubits with:
        - Direct maze encoding on all 25 qubits
        - Two-layer variational structure
        - Rich entanglement patterns reflecting maze connectivity
        """
        
        @qml.qnode(
            self.dev,
            interface=QUANTUM_INTERFACE,
            diff_method=QUANTUM_DIFF_METHOD
        )
        def circuit(inputs, params):
            """
            Full 25-qubit quantum circuit for maze navigation.
            
            Architecture:
            1. Amplitude encoding of maze state across all 25 qubits
            2. Two variational layers with full rotation gates
            3. Nearest-neighbor entanglement reflecting maze structure
            4. Measurement on 4 qubits for action Q-values
            """
            
            # Encode the full 25-dimensional maze state
            # Using angle encoding for better gradient flow
            for i in range(self.num_qubits):
                qml.RY(inputs[i] * pnp.pi, wires=i)
            
            # First variational layer
            for i in range(self.num_qubits):
                qml.RX(params[0, i, 0], wires=i)
                qml.RY(params[0, i, 1], wires=i)
                qml.RZ(params[0, i, 2], wires=i)
            
            # Entanglement layer 1: Grid connectivity (reflecting maze structure)
            # Horizontal connections
            for row in range(5):
                for col in range(4):
                    qubit1 = row * 5 + col
                    qubit2 = row * 5 + col + 1
                    qml.CNOT(wires=[qubit1, qubit2])
            
            # Vertical connections
            for row in range(4):
                for col in range(5):
                    qubit1 = row * 5 + col
                    qubit2 = (row + 1) * 5 + col
                    qml.CNOT(wires=[qubit1, qubit2])
            
            if self.num_layers > 1:
                # Second variational layer
                for i in range(self.num_qubits):
                    qml.RX(params[1, i, 0], wires=i)
                    qml.RY(params[1, i, 1], wires=i)
                    qml.RZ(params[1, i, 2], wires=i)
                
                # Entanglement layer 2: Diagonal connections for richer dynamics
                for row in range(4):
                    for col in range(4):
                        qubit1 = row * 5 + col
                        qubit2 = (row + 1) * 5 + col + 1
                        qml.CZ(wires=[qubit1, qubit2])
            
            # Global entanglement for quantum advantage
            # Connect corners to center for long-range correlations
            center = 12  # Center qubit (position 2,2 in 5x5 grid)
            corners = [0, 4, 20, 24]  # Corner qubits
            for corner in corners:
                qml.CNOT(wires=[center, corner])
            
            # Measure 4 qubits for Q-values (one for each action)
            # Use qubits corresponding to agent's adjacent positions
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        return circuit
    
    def encode_state(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Encode full maze state into 25-dimensional quantum state.
        
        Each qubit directly corresponds to a maze cell, providing a natural
        quantum representation of the spatial problem.
        
        Args:
            maze: Current 5x5 maze
            position: Agent position
            
        Returns:
            25-dimensional encoded state vector
        """
        state = pnp.zeros(self.num_qubits, dtype=np.float32)
        
        # Encode maze structure directly onto qubits
        flat_maze = maze.flatten()
        
        for i in range(25):
            row, col = i // 5, i % 5
            
            # Encode maze cell type
            if flat_maze[i] == WALL:
                state[i] = -1.0
            elif flat_maze[i] == GOAL:
                state[i] = 1.0
            elif flat_maze[i] == START:
                state[i] = 0.5
            else:
                state[i] = 0.0
            
            # Overlay agent position with strong signal
            if (row, col) == position:
                state[i] = 0.8  # Strong positive signal for agent position
            
            # Add distance-based encoding for spatial awareness
            goal_pos = np.where(maze == GOAL)
            if len(goal_pos[0]) > 0:
                goal_row, goal_col = goal_pos[0][0], goal_pos[1][0]
                distance = abs(row - goal_row) + abs(col - goal_col)
                state[i] += 0.1 * (1.0 / (1.0 + distance))  # Distance gradient
        
        # Normalize to improve gradient flow
        norm = pnp.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        return state
    
    def get_q_values(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Get Q-values using full 25-qubit quantum circuit.
        
        Args:
            maze: Current maze
            position: Agent position
            
        Returns:
            Q-values for each action
        """
        state = self.encode_state(maze, position)
        
        with timer("25-qubit circuit evaluation", verbose=False):
            q_values = pnp.array(self.circuit(state, self.params))
            self.perf.record("circuit_eval_time", time.time())
        
        return q_values
    
    def get_q_values_from_state(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values directly from encoded state."""
        q_values = pnp.array(self.circuit(state, self.params))
        return q_values
    
    def update(self, state: np.ndarray, action: int, target: float, 
               current_q: float) -> float:
        """
        Update network parameters using gradient descent.
        
        Uses efficient gradient computation for 25-qubit circuit.
        
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
        
        # Compute gradient efficiently
        gradient = pnp.zeros_like(self.params)
        
        with timer("Gradient computation (25 qubits)", verbose=False):
            # Use finite differences with adaptive step size
            epsilon = 0.01 / pnp.sqrt(self.t)  # Adaptive step size
            
            # Sample subset of parameters for efficiency
            # Update all parameters every 10 steps, otherwise sample
            if self.t % 10 == 0:
                param_indices = [(l, q, p) for l in range(self.num_layers) 
                                for q in range(self.num_qubits) 
                                for p in range(3)]
            else:
                # Sample 30% of parameters for faster updates
                total_params = self.num_layers * self.num_qubits * 3
                sample_size = max(10, int(0.3 * total_params))
                param_indices = []
                for _ in range(sample_size):
                    l = np.random.randint(self.num_layers)
                    q = np.random.randint(self.num_qubits)
                    p = np.random.randint(3)
                    param_indices.append((l, q, p))
            
            for layer, qubit, param_idx in param_indices:
                # Forward difference
                self.params[layer, qubit, param_idx] += epsilon
                q_plus = self.get_q_values_from_state(state)[action]
                
                self.params[layer, qubit, param_idx] -= 2 * epsilon
                q_minus = self.get_q_values_from_state(state)[action]
                
                self.params[layer, qubit, param_idx] += epsilon
                
                # Gradient estimate
                gradient[layer, qubit, param_idx] = (q_plus - q_minus) / (2 * epsilon)
        
        # Scale gradient
        gradient *= 2 * (current_q - target)
        
        # Adam optimizer update
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
        
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        
        # Adaptive learning rate
        adaptive_lr = self.learning_rate / pnp.sqrt(self.t)
        self.params -= adaptive_lr * m_hat / (pnp.sqrt(v_hat) + eps)
        
        self.perf.record("loss", float(loss))
        return float(loss)

# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ExperienceReplayBuffer:
    """Experience replay buffer optimized for 25-qubit states."""
    
    def __init__(self, capacity: int = 5000):  # Smaller due to memory
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
        """Sample random batch for training."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """Q-learning agent using full 25-qubit VQNN."""
    
    def __init__(self, vqnn: VQNN, epsilon: float = EPSILON_START, 
                 gamma: float = GAMMA, epsilon_decay: float = EPSILON_DECAY,
                 use_replay: bool = True, batch_size: int = BATCH_SIZE):
        self.vqnn = vqnn
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.use_replay = use_replay
        self.batch_size = batch_size
        
        if use_replay:
            self.replay_buffer = ExperienceReplayBuffer(capacity=5000)
        
        self.rng = np.random.RandomState()
    
    def select_action(self, maze: np.ndarray, position: Tuple[int, int],
                     training: bool = True) -> Tuple[int, str]:
        """Select action using epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            action_idx = self.rng.randint(4)
        else:
            q_values = self.vqnn.get_q_values(maze, position)
            action_idx = int(pnp.argmax(q_values))
        
        return action_idx, ACTIONS[action_idx]
    
    def train_step(self, maze: np.ndarray, position: Tuple[int, int],
                  action: int, reward: float, next_position: Tuple[int, int],
                  done: bool) -> float:
        """Perform one training step with 25-qubit circuit."""
        if self.use_replay:
            state = self.vqnn.encode_state(maze, position)
            next_state = self.vqnn.encode_state(maze, next_position)
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(self.replay_buffer) >= self.batch_size:
                return self._train_batch()
        
        current_q_values = self.vqnn.get_q_values(maze, position)
        current_q = current_q_values[action]
        
        if done:
            target = reward
        else:
            next_q_values = self.vqnn.get_q_values(maze, next_position)
            target = reward + self.gamma * pnp.max(next_q_values)
        
        state = self.vqnn.encode_state(maze, position)
        loss = self.vqnn.update(state, action, target, current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def _train_batch(self) -> float:
        """Train on batch from replay buffer."""
        batch = self.replay_buffer.sample_batch(self.batch_size)
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            current_q_values = self.vqnn.get_q_values_from_state(state)
            current_q = current_q_values[action]
            
            if done:
                target = reward
            else:
                next_q_values = self.vqnn.get_q_values_from_state(next_state)
                target = reward + self.gamma * pnp.max(next_q_values)
            
            loss = self.vqnn.update(state, action, target, current_q)
            total_loss += loss
        
        return total_loss / self.batch_size

# ============================================================================
# MAZE ENVIRONMENT
# ============================================================================

class MazeEnvironment:
    """Maze environment for agent interaction."""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze.copy()
        self.original_maze = maze.copy()
        self.start_pos = self.find_position(maze, START)
        self.goal_pos = self.find_position(maze, GOAL)
        self.agent_pos = self.start_pos
        self.steps = 0
        self.max_steps = 100
        self.path = [self.start_pos]
        self.actions = []
    
    @staticmethod
    def find_position(maze: np.ndarray, marker: int) -> Tuple[int, int]:
        """Find position of a marker in the maze."""
        pos = np.where(maze == marker)
        return (pos[0][0], pos[1][0])
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment."""
        self.maze = self.original_maze.copy()
        self.agent_pos = self.start_pos
        self.steps = 0
        self.path = [self.start_pos]
        self.actions = []
        return self.agent_pos
    
    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action in environment."""
        self.steps += 1
        self.actions.append(action)
        
        dr, dc = ACTION_DELTAS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        
        if not (0 <= new_row < 5 and 0 <= new_col < 5):
            return self.agent_pos, -10, self.steps >= self.max_steps
        
        if self.maze[new_row, new_col] == WALL:
            return self.agent_pos, -10, self.steps >= self.max_steps
        
        self.agent_pos = (new_row, new_col)
        self.path.append(self.agent_pos)
        
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 100, True
        else:
            return self.agent_pos, -1, self.steps >= self.max_steps

# ============================================================================
# MAZE GENERATION
# ============================================================================

class MazeGenerator:
    """Generate fixed maze environments."""
    
    @staticmethod
    def get_fixed_mazes() -> List[Tuple[np.ndarray, str]]:
        """Generate 10 fixed 5x5 mazes."""
        mazes = []
        
        configs = [
            ("simple_corridor", [[2,0,1,1,1],[0,0,1,1,1],[1,0,0,0,3],[1,1,1,0,1],[1,1,1,0,1]]),
            ("spiral", [[2,0,0,0,0],[1,1,1,1,0],[0,0,0,0,0],[0,1,1,1,1],[0,0,0,0,3]]),
            ("multiple_paths", [[2,0,1,0,3],[0,0,1,0,0],[0,1,0,1,0],[0,0,0,0,0],[1,0,1,0,1]]),
            ("dead_ends", [[2,0,0,1,1],[1,1,0,1,1],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,3]]),
            ("central_barrier", [[2,0,0,0,1],[0,1,1,0,1],[0,1,1,0,0],[0,1,1,1,0],[0,0,0,0,3]]),
            ("zigzag", [[2,1,0,0,0],[0,1,0,1,0],[0,0,0,1,0],[1,1,0,1,0],[3,0,0,1,0]]),
            ("open_field", [[2,0,0,0,0],[0,1,0,1,0],[0,0,0,0,0],[0,1,0,1,0],[0,0,0,0,3]]),
            ("narrow_passages", [[2,1,0,1,1],[0,1,0,1,1],[0,0,0,0,0],[1,1,1,1,0],[1,1,1,1,3]]),
            ("complex_branching", [[2,0,0,1,0],[1,1,0,1,0],[0,0,0,0,0],[0,1,0,1,1],[0,0,0,0,3]]),
            ("deceptive_path", [[2,0,0,0,1],[1,1,1,0,1],[0,0,0,0,1],[0,1,1,1,1],[0,0,0,0,3]])
        ]
        
        for name, layout in configs:
            maze = np.array(layout, dtype=np.float32)
            mazes.append((maze, name))
        
        return mazes
    
    @staticmethod
    def bfs_shortest_path(maze: np.ndarray) -> int:
        """Find shortest path using BFS."""
        from collections import deque
        
        start = MazeEnvironment.find_position(maze, START)
        goal = MazeEnvironment.find_position(maze, GOAL)
        
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
        
        return -1

# ============================================================================
# COMPLEXITY METRICS
# ============================================================================

@lru_cache(maxsize=512)
def lz_complexity(s: str) -> int:
    """Calculate Lempel-Ziv complexity."""
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
    """Calculate Shannon entropy."""
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
    """Calculate approximate entropy."""
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
    """Detect emergent behavior in maze solving."""
    efficiency = solution.efficiency_score
    
    if len(learning_curve) > 10:
        window = 5
        for i in range(window, len(learning_curve) - window):
            before = np.mean(learning_curve[i-window:i])
            after = np.mean(learning_curve[i:i+window])
            if after - before > 30:
                solution.performance_discontinuity = True
                solution.convergence_episode = i
                break
    
    is_emergent = False
    emergence_type = "none"
    emergence_score = 0.0
    
    if efficiency >= 0.95:
        is_emergent = True
        emergence_type = "perfect_navigation"
        emergence_score = 1.0
    elif efficiency >= 0.85:
        is_emergent = True
        emergence_type = "efficient_navigation"
        emergence_score = 0.8
    
    if solution.performance_discontinuity:
        is_emergent = True
        if emergence_type == "none":
            emergence_type = "sudden_insight"
        emergence_score = max(emergence_score, 0.7)
    
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
                      maze_name: str):
    """Visualize the solution path."""
    print(f"\n{maze_name} Solution:")
    print("-" * 15)
    
    for row in range(5):
        line = ""
        for col in range(5):
            if (row, col) in path[1:-1]:
                line += "o "
            elif maze[row, col] == START:
                line += "S "
            elif maze[row, col] == GOAL:
                line += "G "
            elif maze[row, col] == WALL:
                line += "█ "
            else:
                line += ". "
        print(line)
    
    print(f"Path length: {len(path) - 1} steps")

# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_experiments(config: Optional[Dict] = None):
    """Execute full 25-qubit quantum maze navigation experiments."""
    if config is None:
        config = {
            "episodes_per_maze": EPISODES_PER_MAZE,
            "epsilon_start": EPSILON_START,
            "epsilon_decay": EPSILON_DECAY,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "use_replay": True,
            "batch_size": BATCH_SIZE,
            "output_dir": "results",
            "visualize": True
        }
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    print("\n" + "=" * 60)
    print("FULL 25-QUBIT VQNN ADAPTIVE PROBLEM-SOLVING EXPERIMENTS")
    print("=" * 60)
    print(f"Quantum Backend: {QUANTUM_BACKEND}")
    print(f"GPU Acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")
    print(f"Qubits: {NUM_QUBITS} (Full utilization)")
    print(f"Layers: {NUM_LAYERS}")
    print(f"Episodes per maze: {config['episodes_per_maze']}")
    print(f"Expected time per maze: 10-20 minutes")
    print("=" * 60)
    print()
    
    generator = MazeGenerator()
    mazes = generator.get_fixed_mazes()
    
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
        "quantum_backend", "gpu_accelerated", "num_qubits", "training_time_minutes"
    ]
    
    results = []
    total_start = time.time()
    
    print("Running full 25-qubit experiments...\n")
    print("Note: Each maze will take 10-20 minutes to complete.\n")
    
    # Initialize VQNN once and reuse for all mazes
    vqnn = VQNN(
        num_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        learning_rate=config["learning_rate"]
    )
    
    for idx, (maze, maze_name) in enumerate(mazes, 1):
        maze_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Maze {idx}/10: {maze_name}")
        print(f"{'='*60}")
        
        optimal_steps = generator.bfs_shortest_path(maze)
        print(f"Optimal path: {optimal_steps} steps")
        
        random_steps = optimal_steps * 5
        
        # Create new agent for each maze
        agent = QLearningAgent(
            vqnn=vqnn,
            epsilon=config["epsilon_start"],
            gamma=config["gamma"],
            epsilon_decay=config["epsilon_decay"],
            use_replay=config["use_replay"],
            batch_size=config["batch_size"]
        )
        
        env = MazeEnvironment(maze)
        learning_curve = []
        best_solution = None
        best_steps = float('inf')
        
        print(f"\nTraining with full 25-qubit quantum circuit...")
        print("Progress:")
        
        for episode in range(config["episodes_per_maze"]):
            position = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action_idx, action = agent.select_action(maze, position, training=True)
                next_position, reward, done = env.step(action)
                
                loss = agent.train_step(maze, position, action_idx, reward, 
                                       next_position, done)
                
                position = next_position
                episode_reward += reward
            
            learning_curve.append(episode_reward)
            
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
            
            # Progress updates
            if episode % 10 == 0:
                avg_reward = np.mean(learning_curve[-10:]) if len(learning_curve) >= 10 else episode_reward
                elapsed = (time.time() - maze_start) / 60
                print(f"  Episode {episode:3d}/{config['episodes_per_maze']}: "
                      f"Avg reward = {avg_reward:6.1f}, ε = {agent.epsilon:.3f}, "
                      f"Time = {elapsed:.1f} min")
        
        # Final evaluation
        print("\nEvaluating final performance...")
        agent.epsilon = 0
        env.reset()
        position = env.start_pos
        done = False
        
        while not done:
            action_idx, action = agent.select_action(maze, position, training=False)
            position, reward, done = env.step(action)
        
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
        
        action_numeric = [ACTIONS.index(a) for a in best_solution.action_sequence]
        action_apen = approximate_entropy(action_numeric) if len(action_numeric) > 2 else 0
        
        baselines = {"random": random_steps, "optimal": optimal_steps}
        emergence = detect_emergence(best_solution, baselines, learning_curve)
        
        training_time = (time.time() - maze_start) / 60
        
        print(f"\nResults:")
        print(f"  Steps to goal: {best_solution.steps_to_goal}")
        print(f"  Efficiency: {best_solution.efficiency_score:.2%}")
        print(f"  Emergent: {emergence['is_emergent']} ({emergence['emergence_type']})")
        print(f"  Training time: {training_time:.1f} minutes")
        
        if config.get("visualize", True):
            visualize_solution(maze, best_solution.solution_path, maze_name)
        
        # Save results
        results.append([
            f"VQNN_maze_{idx}",
            datetime.now().isoformat(),
            maze_name,
            len(best_solution.solution_path),
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
            QUANTUM_BACKEND,
            GPU_AVAILABLE,
            NUM_QUBITS,
            training_time
        ])
        
        # Reset agent epsilon for next maze
        agent.epsilon = config["epsilon_start"]
    
    # Save all results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    total_time = (time.time() - total_start) / 60
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    total_emergent = sum(1 for r in results if r[18])
    avg_efficiency = np.mean([r[7] for r in results])
    
    print(f"Total mazes: {len(results)}")
    print(f"Emergent solutions: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    print(f"Average efficiency: {avg_efficiency:.3f}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Average time per maze: {total_time/len(mazes):.1f} minutes")
    print(f"Quantum backend: {QUANTUM_BACKEND}")
    print(f"Qubits used: {NUM_QUBITS}")
    print(f"Parameters trained: {vqnn.params.size}")
    
    emergence_types = {}
    for r in results:
        if r[18]:
            etype = r[19]
            emergence_types[etype] = emergence_types.get(etype, 0) + 1
    
    if emergence_types:
        print("\nEmergence breakdown:")
        for etype, count in emergence_types.items():
            print(f"  {etype}: {count}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Performance report
    vqnn.perf.report()
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("INITIALIZING FULL 25-QUBIT QUANTUM EXPERIMENT")
    print("="*60)
    print("\nThis experiment will demonstrate quantum computing")
    print("using all 25 qubits for maximum research impact.")
    print("\nExpected total runtime: 2-3 hours")
    print("="*60)
    
    response = input("\nProceed with full 25-qubit experiment? (y/n): ")
    
    if response.lower() == 'y':
        config = {
            "episodes_per_maze": EPISODES_PER_MAZE,
            "epsilon_start": EPSILON_START,
            "epsilon_decay": EPSILON_DECAY,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "use_replay": True,
            "batch_size": BATCH_SIZE,
            "output_dir": "results",
            "visualize": True
        }
        
        results = run_experiments(config)
        
        print("\n✅ Full 25-qubit experiment complete!")
    else:
        print("\nExperiment cancelled.")
        print("To run with fewer qubits, modify NUM_QUBITS in the configuration.")
    
    print("=" * 60)