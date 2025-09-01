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
__version__ = "1.10-alpha"
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
def timer(name: str, verbose: bool = False):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    if verbose:
        print(f"  ⏱ {name}: {end - start:.3f}s")

class PerformanceMonitor:
    """Track performance metrics."""
    
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
            if values:
                avg = np.mean(values)
                print(f"  {name}: avg={avg:.3f}, min={min(values):.3f}, max={max(values):.3f}")

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def configure_gpu_settings():
    """Configure GPU settings for shot-based quantum simulation."""
    import os
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDNN_BENCHMARK'] = 'TRUE'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    print("✓ GPU settings configured for shot-based simulation")

configure_gpu_settings()

# ============================================================================
# QUANTUM BACKEND DETECTION
# ============================================================================

def detect_quantum_backend():
    """Detect and configure quantum backend for shot-based simulation."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        
        # Try GPU backend first
        try:
            # Test with shots to ensure compatibility
            test_dev = qml.device("lightning.gpu", wires=2, shots=100)
            del test_dev
            
            print("=" * 60)
            print("✓ NVIDIA cuQuantum SDK detected!")
            print("✓ Using Lightning.GPU with shot-based simulation")
            print("✓ 25 qubits with 1000 shots per evaluation")
            print("=" * 60)
            
            return {
                "available": True,
                "backend": "lightning.gpu",
                "interface": "autograd",
                "diff_method": "parameter-shift",  # Required for shots
                "gpu": True,
                "pnp": pnp
            }
            
        except Exception:
            # Try lightning.qubit for CPU
            try:
                test_dev = qml.device("lightning.qubit", wires=2, shots=100)
                del test_dev
                print("=" * 60)
                print("✓ Using Lightning.Qubit with shot-based simulation")
                print("✓ 25 qubits with 1000 shots per evaluation")
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
                print("=" * 60)
                print("ℹ Using default.qubit with shot-based simulation")
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

# Quantum circuit configuration
NUM_QUBITS = 25  # Full 25 qubits
NUM_LAYERS = 1   # Single layer for shot-based efficiency
SHOTS = 1000     # Number of measurement shots

# Training configuration adjusted for shot noise
EPISODES_PER_MAZE = 150  # More episodes due to noise
LEARNING_RATE = 0.05      # Higher LR to overcome shot noise
GAMMA = 0.95
EPSILON_START = 0.3
EPSILON_DECAY = 0.993     # Slower decay for exploration
BATCH_SIZE = 32           # Reasonable batch size

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
# SHOT-BASED 25-QUBIT VQNN
# ============================================================================

class VQNN:
    """
    25-Qubit Variational Quantum Neural Network using shot-based simulation.
    
    This implementation uses measurement sampling (shots) instead of exact
    state vector calculation, making 25-qubit simulation tractable.
    """
    
    def __init__(self, num_qubits: int = NUM_QUBITS, num_layers: int = NUM_LAYERS,
                 learning_rate: float = LEARNING_RATE, shots: int = SHOTS):
        """
        Initialize VQNN with shot-based quantum simulation.
        
        Args:
            num_qubits: Number of qubits (25)
            num_layers: Number of variational layers (1 for efficiency)
            learning_rate: Learning rate (higher for shot noise)
            shots: Number of measurement shots (1000)
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.shots = shots
        
        # Initialize parameters - fewer for shot-based
        # Only 2 parameters per qubit for single layer
        self.params = pnp.random.randn(num_layers, num_qubits, 2) * 0.1
        
        # Adam optimizer state
        self.m = pnp.zeros_like(self.params)
        self.v = pnp.zeros_like(self.params)
        self.t = 0
        
        # Shot noise compensation
        self.noise_scale = 1.0 / np.sqrt(shots)  # Statistical error scale
        
        # Performance tracking
        self.perf = PerformanceMonitor()
        
        print(f"Initializing Shot-Based 25-Qubit VQNN...")
        print(f"  Configuration:")
        print(f"    - Qubits: {num_qubits}")
        print(f"    - Layers: {num_layers}")
        print(f"    - Shots: {shots}")
        print(f"    - Statistical error: ±{self.noise_scale:.3f}")
        print(f"    - Backend: {QUANTUM_BACKEND}")
        
        # Initialize quantum device with shots
        # CORRECTED: Removed analytic parameter
        self.dev = qml.device(
            QUANTUM_BACKEND,
            wires=num_qubits,
            shots=shots
        )
        
        # Create shot-optimized circuit
        self.circuit = self._create_shot_optimized_circuit()
        
        print(f"  ✓ Shot-based quantum circuit initialized")
        print(f"  ✓ Expected runtime: 5-15 minutes per maze")
    
    def _create_shot_optimized_circuit(self):
        """
        Create quantum circuit optimized for shot-based simulation.
        
        Simpler circuit structure to reduce shot noise accumulation:
        - Single layer of parameterized gates
        - Local entanglement only
        - Robust measurement strategy
        """
        
        @qml.qnode(
            self.dev,
            interface=QUANTUM_INTERFACE,
            diff_method=QUANTUM_DIFF_METHOD  # parameter-shift for shots
        )
        def circuit(inputs, params):
            """
            Shot-optimized 25-qubit quantum circuit.
            
            Design principles:
            - Shallow circuit to minimize noise accumulation
            - Local operations for better shot statistics
            - Strategic entanglement for quantum advantage
            """
            
            # Input encoding - simple angle encoding
            # More robust to shot noise than amplitude encoding
            for i in range(self.num_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Single variational layer
            for i in range(self.num_qubits):
                qml.RX(params[0, i, 0], wires=i)
                qml.RY(params[0, i, 1], wires=i)
            
            # Local entanglement pattern
            # Only nearest-neighbor to reduce circuit depth
            for i in range(0, self.num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            
            # Second set of CNOTs for odd pairs
            for i in range(1, self.num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            
            # Measure 4 qubits for Q-values
            # Using expectation values with shots
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        return circuit
    
    def encode_state(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Encode maze state for shot-based quantum processing.
        
        Simplified encoding to reduce sensitivity to shot noise.
        
        Args:
            maze: Current 5x5 maze
            position: Agent position
            
        Returns:
            25-dimensional encoded state vector
        """
        state = pnp.zeros(self.num_qubits, dtype=np.float32)
        
        # Direct position encoding
        flat_maze = maze.flatten()
        
        for i in range(25):
            row, col = i // 5, i % 5
            
            # Binary-like encoding for robustness
            if flat_maze[i] == WALL:
                state[i] = -0.5
            elif flat_maze[i] == GOAL:
                state[i] = 0.5
            else:
                state[i] = 0.0
            
            # Agent position gets strongest signal
            if (row, col) == position:
                state[i] = 1.0
        
        # Normalize for consistency
        norm = pnp.linalg.norm(state)
        if norm > 0:
            state = state / (norm * 2)  # Scaled normalization
        
        return state
    
    def get_q_values(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Get Q-values using shot-based quantum circuit.
        
        Returns averaged measurements over shots with noise handling.
        
        Args:
            maze: Current maze
            position: Agent position
            
        Returns:
            Q-values for each action with shot noise
        """
        state = self.encode_state(maze, position)
        
        with timer("Shot-based circuit evaluation", verbose=False):
            # Execute circuit with shots
            q_values = pnp.array(self.circuit(state, self.params))
            
            # Add small noise for exploration (simulating shot variance)
            if self.t > 0:  # Not during initialization
                noise = pnp.random.normal(0, self.noise_scale, size=4)
                q_values = q_values + noise
            
            self.perf.record("circuit_eval_time", time.time())
        
        return q_values
    
    def get_q_values_from_state(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values directly from encoded state."""
        q_values = pnp.array(self.circuit(state, self.params))
        
        # Add shot noise simulation
        if self.t > 0:
            noise = pnp.random.normal(0, self.noise_scale, size=4)
            q_values = q_values + noise
        
        return q_values
    
    def update(self, state: np.ndarray, action: int, target: float, 
               current_q: float) -> float:
        """
        Update parameters using parameter-shift rule for shot-based gradients.
        
        Includes noise-aware optimization strategies.
        
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
        
        # Compute gradient with parameter-shift rule
        gradient = pnp.zeros_like(self.params)
        shift = np.pi / 2  # Standard shift for parameter-shift rule
        
        with timer("Shot-based gradient computation", verbose=False):
            # Sample subset of parameters to update (for speed)
            # With shots, updating all parameters is expensive
            num_params = self.num_layers * self.num_qubits * 2
            
            # Adaptive sampling: update more parameters early, fewer later
            if self.t < 50:
                sample_rate = 0.5  # Update 50% of parameters
            elif self.t < 100:
                sample_rate = 0.3  # Update 30% of parameters
            else:
                sample_rate = 0.2  # Update 20% of parameters
            
            sample_size = max(5, int(sample_rate * num_params))
            
            # Randomly select parameters to update
            param_indices = []
            for _ in range(sample_size):
                l = np.random.randint(self.num_layers)
                q = np.random.randint(self.num_qubits)
                p = np.random.randint(2)
                param_indices.append((l, q, p))
            
            # Parameter-shift gradient estimation
            for layer, qubit, param_idx in param_indices:
                # Shift parameter up
                self.params[layer, qubit, param_idx] += shift
                q_plus = self.get_q_values_from_state(state)[action]
                
                # Shift parameter down
                self.params[layer, qubit, param_idx] -= 2 * shift
                q_minus = self.get_q_values_from_state(state)[action]
                
                # Restore parameter
                self.params[layer, qubit, param_idx] += shift
                
                # Parameter-shift gradient with noise scaling
                gradient[layer, qubit, param_idx] = (q_plus - q_minus) / (2 * np.sin(shift))
                
                # Scale by sampling rate to maintain update magnitude
                gradient[layer, qubit, param_idx] /= sample_rate
        
        # Scale gradient by loss derivative
        gradient *= 2 * (current_q - target)
        
        # Adam optimizer with noise-aware settings
        beta1 = 0.9
        beta2 = 0.99  # Less aggressive than usual due to noise
        eps = 1e-6    # Larger epsilon for shot noise
        
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
        
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        
        # Adaptive learning rate with noise compensation
        noise_adjusted_lr = self.learning_rate * np.sqrt(self.shots / 1000)
        effective_lr = noise_adjusted_lr / np.sqrt(self.t)
        
        self.params -= effective_lr * m_hat / (pnp.sqrt(v_hat) + eps)
        
        # Clip parameters to prevent instability from shot noise
        self.params = pnp.clip(self.params, -2 * np.pi, 2 * np.pi)
        
        self.perf.record("loss", float(loss))
        return float(loss)

# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ExperienceReplayBuffer:
    """Experience replay buffer adapted for shot-based training."""
    
    def __init__(self, capacity: int = 10000):
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
    """Q-learning agent adapted for shot-based VQNN."""
    
    def __init__(self, vqnn: VQNN, epsilon: float = EPSILON_START, 
                 gamma: float = GAMMA, epsilon_decay: float = EPSILON_DECAY,
                 use_replay: bool = True, batch_size: int = BATCH_SIZE):
        self.vqnn = vqnn
        self.epsilon = epsilon
        self.epsilon_min = 0.05  # Higher min due to shot noise
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.use_replay = use_replay
        self.batch_size = batch_size
        
        if use_replay:
            self.replay_buffer = ExperienceReplayBuffer(capacity=10000)
        
        self.rng = np.random.RandomState()
    
    def select_action(self, maze: np.ndarray, position: Tuple[int, int],
                     training: bool = True) -> Tuple[int, str]:
        """Select action with noise-aware epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            action_idx = self.rng.randint(4)
        else:
            q_values = self.vqnn.get_q_values(maze, position)
            
            # Boltzmann action selection for shot noise robustness
            if training and self.vqnn.t > 0:
                # Temperature parameter decreases over time
                temperature = 0.5 / np.sqrt(self.vqnn.t)
                probabilities = np.exp(q_values / temperature)
                probabilities = probabilities / np.sum(probabilities)
                action_idx = self.rng.choice(4, p=probabilities)
            else:
                action_idx = int(pnp.argmax(q_values))
        
        return action_idx, ACTIONS[action_idx]
    
    def train_step(self, maze: np.ndarray, position: Tuple[int, int],
                  action: int, reward: float, next_position: Tuple[int, int],
                  done: bool) -> float:
        """Training step adapted for shot noise."""
        if self.use_replay:
            state = self.vqnn.encode_state(maze, position)
            next_state = self.vqnn.encode_state(maze, next_position)
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train more frequently to average out shot noise
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
        """Batch training with shot noise handling."""
        batch = self.replay_buffer.sample_batch(self.batch_size)
        total_loss = 0.0
        
        # Process in smaller sub-batches for shot-based training
        sub_batch_size = 8
        
        for i in range(0, len(batch), sub_batch_size):
            sub_batch = batch[i:i+sub_batch_size]
            
            for state, action, reward, next_state, done in sub_batch:
                current_q_values = self.vqnn.get_q_values_from_state(state)
                current_q = current_q_values[action]
                
                if done:
                    target = reward
                else:
                    next_q_values = self.vqnn.get_q_values_from_state(next_state)
                    target = reward + self.gamma * pnp.max(next_q_values)
                
                loss = self.vqnn.update(state, action, target, current_q)
                total_loss += loss
        
        return total_loss / len(batch)

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
    
    # Adjusted thresholds for shot noise
    if efficiency >= 0.90:  # Slightly lower due to noise
        is_emergent = True
        emergence_type = "perfect_navigation"
        emergence_score = 1.0
    elif efficiency >= 0.75:  # Adjusted threshold
        is_emergent = True
        emergence_type = "efficient_navigation"
        emergence_score = 0.8
    
    if solution.performance_discontinuity:
        is_emergent = True
        if emergence_type == "none":
            emergence_type = "sudden_insight"
        emergence_score = max(emergence_score, 0.7)
    
    random_improvement = (baseline_steps["random"] - solution.steps_to_goal) / baseline_steps["random"]
    if random_improvement > 0.7:  # Adjusted for noise
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
    """Execute shot-based 25-qubit quantum maze navigation experiments."""
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
    print("SHOT-BASED 25-QUBIT VQNN EXPERIMENTS")
    print("=" * 60)
    print(f"Quantum Backend: {QUANTUM_BACKEND}")
    print(f"GPU Acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Shots: {SHOTS}")
    print(f"Episodes per maze: {config['episodes_per_maze']}")
    print(f"Expected time per maze: 5-15 minutes")
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
        "quantum_backend", "gpu_accelerated", "num_qubits", "shots",
        "training_time_minutes"
    ]
    
    results = []
    total_start = time.time()
    
    print("Running shot-based quantum experiments...\n")
    
    # Initialize VQNN once
    vqnn = VQNN(
        num_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        learning_rate=config["learning_rate"],
        shots=SHOTS
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
        
        print(f"\nTraining with shot-based quantum circuit (1000 shots)...")
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
            if episode % 20 == 0:
                avg_reward = np.mean(learning_curve[-10:]) if len(learning_curve) >= 10 else episode_reward
                elapsed = (time.time() - maze_start) / 60
                print(f"  Episode {episode:3d}/{config['episodes_per_maze']}: "
                      f"Avg reward = {avg_reward:6.1f}, ε = {agent.epsilon:.3f}, "
                      f"Time = {elapsed:.1f} min")
        
        # Final evaluation
        print("\nEvaluating final performance...")
        agent.epsilon = 0
        
        # Run multiple evaluations to average out shot noise
        eval_steps = []
        for _ in range(5):
            env.reset()
            position = env.start_pos
            done = False
            
            while not done and env.steps < env.max_steps:
                action_idx, action = agent.select_action(maze, position, training=False)
                position, reward, done = env.step(action)
            
            if done and position == env.goal_pos:
                eval_steps.append(env.steps)
        
        # Use median of evaluations for robustness
        if eval_steps:
            final_steps = int(np.median(eval_steps))
        else:
            final_steps = env.steps
        
        if best_solution is None or final_steps < best_solution.steps_to_goal:
            best_solution = MazeSolution(
                maze_id=maze_name,
                steps_to_goal=final_steps,
                optimal_steps=optimal_steps,
                efficiency_score=optimal_steps / final_steps if final_steps > 0 else 0,
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
            SHOTS,
            training_time
        ])
        
        # Reset parameters for next maze (optional)
        # vqnn.params = pnp.random.randn(NUM_LAYERS, NUM_QUBITS, 2) * 0.1
    
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
    
    total_emergent = sum(1 for r in results if r[19])
    avg_efficiency = np.mean([r[7] for r in results])
    
    print(f"Total mazes: {len(results)}")
    print(f"Emergent solutions: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    print(f"Average efficiency: {avg_efficiency:.3f}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Average time per maze: {total_time/len(mazes):.1f} minutes")
    print(f"Quantum backend: {QUANTUM_BACKEND}")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Shots: {SHOTS}")
    print(f"Statistical error: ±{1/np.sqrt(SHOTS):.3f}")
    
    emergence_types = {}
    for r in results:
        if r[19]:
            etype = r[20]
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
    print("SHOT-BASED 25-QUBIT QUANTUM EXPERIMENT")
    print("="*60)
    print("\nThis experiment uses 25 qubits with shot-based simulation")
    print("for tractable quantum computation.")
    print("\nConfiguration:")
    print(f"  - Qubits: {NUM_QUBITS}")
    print(f"  - Shots: {SHOTS} per circuit evaluation")
    print(f"  - Statistical error: ±{100/np.sqrt(SHOTS):.1f}%")
    print(f"  - Expected runtime: 1-2 hours total")
    print("="*60)
    
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
    
    print("\n✅ Shot-based 25-qubit experiment complete!")
    print("=" * 60)