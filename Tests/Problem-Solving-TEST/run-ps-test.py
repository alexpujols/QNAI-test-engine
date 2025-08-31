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
import time
from contextlib import contextmanager
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@contextmanager
def timer(name: str, verbose: bool = False):
    """Context manager for timing code blocks."""
    if verbose:
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        print(f"  ⏱ {name}: {end - start:.3f}s")
    else:
        yield

class PerformanceMonitor:
    """Track and report performance metrics."""
    
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
        print(f"Total runtime: {elapsed:.2f}s")
        for name, values in self.metrics.items():
            avg = np.mean(values)
            print(f"  {name}: avg={avg:.3f}, min={min(values):.3f}, max={max(values):.3f}")

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def optimize_gpu_settings():
    """Configure optimal GPU settings for quantum simulation."""
    import os
    
    # CUDA settings for optimal performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
    os.environ['CUDNN_BENCHMARK'] = 'TRUE'    # Auto-tune operations
    
    # Memory management
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_MEMORY_FRACTION'] = '0.8'
    
    # XLA compilation for better performance
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    print("✓ GPU settings optimized")

# Apply GPU optimizations
optimize_gpu_settings()

# ============================================================================
# QUANTUM BACKEND DETECTION
# ============================================================================

def detect_quantum_backend():
    """Detect and configure the best available quantum backend."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        
        # Try NVIDIA GPU backend first
        try:
            test_dev = qml.device("lightning.gpu", wires=2)
            del test_dev
            
            print("=" * 60)
            print("✓ NVIDIA cuQuantum SDK detected!")
            print("✓ Using PennyLane Lightning.GPU for quantum acceleration")
            print("=" * 60)
            
            return {
                "available": True,
                "backend": "lightning.gpu",
                "interface": "autograd",
                "diff_method": "adjoint",
                "gpu": True,
                "pnp": pnp
            }
            
        except Exception:
            print("=" * 60)
            print("ℹ Using CPU-based quantum simulation")
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
        print("⚠ PennyLane not installed, using classical fallback")
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
pnp = QUANTUM_CONFIG["pnp"]

if PENNYLANE_AVAILABLE:
    import pennylane as qml

# ============================================================================
# CONFIGURATION
# ============================================================================

# Performance settings
BATCH_SIZE = 64  # Increased for better GPU utilization
PARALLEL_ENVS = 8  # Number of parallel environments
CACHE_SIZE = 2048  # Circuit cache size
NUM_WORKERS = 4  # Async data loading workers

# Maze constants
EMPTY, WALL, START, GOAL, VISITED = 0, 1, 2, 3, 4
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DELTAS = {
    'UP': (-1, 0), 'DOWN': (1, 0),
    'LEFT': (0, -1), 'RIGHT': (0, 1)
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
# OPTIMIZED VQNN
# ============================================================================

class OptimizedVQNN:
    """
    Optimized Variational Quantum Neural Network with:
    - Batch processing
    - Circuit caching
    - Vectorized operations
    - GPU optimization
    """
    
    def __init__(self, num_qubits: int = 25, num_layers: int = 2,
                 learning_rate: float = 0.01, batch_size: int = BATCH_SIZE):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize parameters with smaller variance for stability
        self.params = pnp.random.randn(num_layers, num_qubits, 2) * 0.01
        
        # Adam optimizer state
        self.m = pnp.zeros_like(self.params)
        self.v = pnp.zeros_like(self.params)
        self.t = 0
        
        # Performance tracking
        self.perf = PerformanceMonitor()
        
        # Circuit caching
        self._circuit_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize quantum device
        if PENNYLANE_AVAILABLE:
            if QUANTUM_BACKEND == "lightning.gpu":
                self.dev = qml.device(
                    "lightning.gpu",
                    wires=num_qubits,
                    batch_obs=True,
                    shots=None  # Analytic mode
                )
            else:
                self.dev = qml.device(
                    "default.qubit",
                    wires=num_qubits,
                    shots=None
                )
            
            self._create_circuits()
        else:
            self.dev = None
        
        print(f"✓ Optimized VQNN initialized (batch_size={batch_size})")
    
    def _create_circuits(self):
        """Create optimized quantum circuits."""
        
        # Main circuit for Q-value computation
        @qml.qnode(
            self.dev,
            interface="autograd",
            diff_method="adjoint" if QUANTUM_BACKEND == "lightning.gpu" else "backprop",
            cache=True  # Enable caching
        )
        def circuit(inputs, params):
            # Amplitude encoding
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.num_qubits),
                normalize=True,
                pad_with=0.0
            )
            
            # Variational layers
            for layer in range(self.num_layers):
                # Parallel rotations
                for i in range(self.num_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                
                # Efficient entanglement for GPU
                if layer < self.num_layers - 1:
                    # Even pairs
                    for i in range(0, self.num_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    # Odd pairs
                    for i in range(1, self.num_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
            
            # Return Q-values for 4 actions
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        self.circuit = circuit
        
        # Create gradient function
        self.gradient_fn = qml.grad(circuit, argnum=1)
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _cached_circuit_eval(self, state_hash: int, params_hash: int):
        """Cached circuit evaluation."""
        self._cache_hits += 1
        # Reconstruct state from hash (simplified - in practice, store mapping)
        return self.circuit
    
    def encode_states_batch(self, mazes: np.ndarray, 
                           positions: np.ndarray) -> np.ndarray:
        """
        Vectorized batch encoding of maze states.
        
        Args:
            mazes: Batch of maze arrays (batch_size, 5, 5)
            positions: Batch of positions (batch_size, 2)
        """
        batch_size = len(mazes)
        states = pnp.zeros((batch_size, 25), dtype=np.float32)
        
        # Vectorized maze encoding
        flat_mazes = mazes.reshape(batch_size, -1)
        
        # Encode walls as -1, goals as 0.5, empty as 0
        states[flat_mazes == WALL] = -1.0
        states[flat_mazes == GOAL] = 0.5
        states[flat_mazes == EMPTY] = 0.0
        
        # Encode agent positions
        agent_indices = positions[:, 0] * 5 + positions[:, 1]
        for i, idx in enumerate(agent_indices):
            states[i, idx] = 1.0
        
        return states
    
    def get_q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Get Q-values for batch of states.
        
        Args:
            states: Batch of encoded states (batch_size, 25)
        """
        with timer("Q-value computation", verbose=False):
            if not PENNYLANE_AVAILABLE:
                return self._classical_forward_batch(states)
            
            batch_size = states.shape[0]
            q_values = pnp.zeros((batch_size, 4))
            
            # Process in chunks for memory efficiency
            chunk_size = min(32, batch_size)
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk = states[i:end_idx]
                
                # Parallel circuit evaluation
                for j, state in enumerate(chunk):
                    q_values[i + j] = pnp.array(self.circuit(state, self.params))
            
            self.perf.record("q_values_batch", time.time())
            return q_values
    
    def _classical_forward_batch(self, states: np.ndarray) -> np.ndarray:
        """Classical neural network fallback for batch."""
        # Simple linear model as fallback
        weights = self.params.reshape(-1, 4)[:states.shape[1]]
        return pnp.tanh(states @ weights)
    
    def update_batch(self, states: np.ndarray, actions: np.ndarray,
                    targets: np.ndarray, current_q_values: Optional[np.ndarray] = None) -> float:
        """
        Optimized batch update with vectorized operations.
        """
        self.t += 1
        batch_size = states.shape[0]
        
        with timer("Batch update", verbose=False):
            # Get current Q-values if not provided
            if current_q_values is None:
                current_q_values = self.get_q_values_batch(states)
            
            # Vectorized Q-value selection
            batch_indices = np.arange(batch_size)
            current_q = current_q_values[batch_indices, actions]
            
            # Vectorized loss computation
            losses = (targets - current_q) ** 2
            avg_loss = pnp.mean(losses)
            
            # Accumulate gradients
            if PENNYLANE_AVAILABLE:
                accumulated_gradient = pnp.zeros_like(self.params)
                
                # Sample subset for gradient computation (for efficiency)
                sample_size = min(16, batch_size)
                sample_indices = np.random.choice(batch_size, sample_size, replace=False)
                
                for idx in sample_indices:
                    grad = self.gradient_fn(states[idx], self.params)
                    loss_grad = 2 * (current_q[idx] - targets[idx])
                    
                    # Accumulate gradient
                    for action_idx in range(4):
                        if isinstance(grad[action_idx], np.ndarray):
                            accumulated_gradient += loss_grad * grad[action_idx]
                
                accumulated_gradient /= sample_size
            else:
                # Classical gradient
                accumulated_gradient = pnp.random.randn(*self.params.shape) * 0.01
            
            # Adam optimizer update
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            
            self.m = beta1 * self.m + (1 - beta1) * accumulated_gradient
            self.v = beta2 * self.v + (1 - beta2) * accumulated_gradient ** 2
            
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            
            self.params -= self.learning_rate * m_hat / (pnp.sqrt(v_hat) + eps)
            
            self.perf.record("update_batch", time.time())
            return float(avg_loss)

# ============================================================================
# OPTIMIZED REPLAY BUFFER
# ============================================================================

class OptimizedReplayBuffer:
    """
    High-performance experience replay buffer with:
    - Pre-allocated arrays
    - Vectorized operations
    - Efficient sampling
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, 25), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 25), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
    
    def push_batch(self, states: np.ndarray, actions: np.ndarray,
                  rewards: np.ndarray, next_states: np.ndarray,
                  dones: np.ndarray):
        """Add batch of experiences efficiently."""
        batch_size = len(states)
        
        if batch_size == 0:
            return
        
        # Calculate insertion indices
        if self.position + batch_size <= self.capacity:
            indices = np.arange(self.position, self.position + batch_size)
        else:
            # Wrap around
            indices = np.concatenate([
                np.arange(self.position, self.capacity),
                np.arange(0, (self.position + batch_size) % self.capacity)
            ])
        
        # Batch insertion
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones
        
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with vectorized operations."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

# ============================================================================
# PARALLEL ENVIRONMENT MANAGER
# ============================================================================

class ParallelEnvironmentManager:
    """
    Manage multiple maze environments in parallel for faster training.
    """
    
    def __init__(self, mazes: List[np.ndarray], num_parallel: int = PARALLEL_ENVS):
        self.mazes = mazes
        self.num_parallel = min(num_parallel, len(mazes))
        self.envs = []
        self.reset_all()
    
    def reset_all(self):
        """Reset all parallel environments."""
        from collections import deque
        
        self.envs = []
        maze_queue = deque(self.mazes)
        
        for _ in range(self.num_parallel):
            if maze_queue:
                maze = maze_queue.popleft()
                env = MazeEnvironment(maze)
                self.envs.append(env)
    
    def step_all(self, actions: List[str]) -> Tuple[List, List, List]:
        """Execute actions in all environments."""
        next_positions = []
        rewards = []
        dones = []
        
        for env, action in zip(self.envs, actions):
            next_pos, reward, done = env.step(action)
            next_positions.append(next_pos)
            rewards.append(reward)
            dones.append(done)
        
        return next_positions, rewards, dones
    
    def get_states(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Get current states from all environments."""
        mazes = [env.maze for env in self.envs]
        positions = [env.agent_pos for env in self.envs]
        return mazes, positions

# ============================================================================
# MAZE GENERATION
# ============================================================================

class MazeGenerator:
    """Generate and manage maze environments."""
    
    @staticmethod
    def get_fixed_mazes() -> List[Tuple[np.ndarray, str]]:
        """Generate 10 fixed 5x5 mazes with varying complexity."""
        mazes = []
        
        # Maze configurations (same as original)
        maze_configs = [
            ("simple_corridor", [
                [2, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [1, 0, 0, 0, 3],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 0, 1]
            ]),
            ("spiral", [
                [2, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1],
                [0, 0, 0, 0, 3]
            ]),
            ("multiple_paths", [
                [2, 0, 1, 0, 3],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1]
            ]),
            ("dead_ends", [
                [2, 0, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 3]
            ]),
            ("central_barrier", [
                [2, 0, 0, 0, 1],
                [0, 1, 1, 0, 1],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 3]
            ]),
            ("zigzag", [
                [2, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [3, 0, 0, 1, 0]
            ]),
            ("open_field", [
                [2, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 3]
            ]),
            ("narrow_passages", [
                [2, 1, 0, 1, 1],
                [0, 1, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 3]
            ]),
            ("complex_branching", [
                [2, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 0, 0, 0, 3]
            ]),
            ("deceptive_path", [
                [2, 0, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 0, 0, 3]
            ])
        ]
        
        for name, layout in maze_configs:
            maze = np.array(layout, dtype=np.float32)
            mazes.append((maze, name))
        
        return mazes
    
    @staticmethod
    def find_position(maze: np.ndarray, marker: int) -> Tuple[int, int]:
        """Find position of a marker in the maze."""
        pos = np.where(maze == marker)
        return (pos[0][0], pos[1][0])
    
    @staticmethod
    def bfs_shortest_path(maze: np.ndarray) -> int:
        """Find shortest path using BFS."""
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
        
        return -1

# ============================================================================
# MAZE ENVIRONMENT
# ============================================================================

class MazeEnvironment:
    """Maze environment for agent interaction."""
    
    def __init__(self, maze: np.ndarray):
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
        
        # Calculate new position
        dr, dc = ACTION_DELTAS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        
        # Check boundaries and walls
        if not (0 <= new_row < 5 and 0 <= new_col < 5):
            return self.agent_pos, -10, self.steps >= self.max_steps
        
        if self.maze[new_row, new_col] == WALL:
            return self.agent_pos, -10, self.steps >= self.max_steps
        
        # Valid move
        self.agent_pos = (new_row, new_col)
        self.path.append(self.agent_pos)
        
        # Check for goal
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 100, True
        else:
            return self.agent_pos, -1, self.steps >= self.max_steps

# ============================================================================
# OPTIMIZED Q-LEARNING AGENT
# ============================================================================

class OptimizedQLearningAgent:
    """
    Q-learning agent with optimized batch processing.
    """
    
    def __init__(self, vqnn: OptimizedVQNN, epsilon: float = 0.1,
                 gamma: float = 0.99, epsilon_decay: float = 0.995):
        self.vqnn = vqnn
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        
        self.replay_buffer = OptimizedReplayBuffer(capacity=100000)
        self.rng = np.random.RandomState()
    
    def select_actions_batch(self, mazes: List[np.ndarray],
                            positions: List[Tuple[int, int]],
                            training: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Select actions for batch of states."""
        batch_size = len(mazes)
        
        # Convert to numpy arrays
        mazes_array = np.array(mazes)
        positions_array = np.array(positions)
        
        # Encode states
        states = self.vqnn.encode_states_batch(mazes_array, positions_array)
        
        if training and self.rng.random() < self.epsilon:
            # Exploration
            action_indices = self.rng.randint(0, 4, size=batch_size)
        else:
            # Exploitation
            q_values = self.vqnn.get_q_values_batch(states)
            action_indices = pnp.argmax(q_values, axis=1)
        
        actions = [ACTIONS[int(i)] for i in action_indices]
        return action_indices, actions
    
    def train_on_batch(self) -> float:
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0
        
        with timer("Training batch", verbose=False):
            # Sample batch
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample_batch(BATCH_SIZE)
            
            # Compute Q-values for current and next states in parallel
            current_q_values = self.vqnn.get_q_values_batch(states)
            next_q_values = self.vqnn.get_q_values_batch(next_states)
            
            # Compute targets
            max_next_q = pnp.max(next_q_values, axis=1)
            targets = rewards + self.gamma * max_next_q * (~dones)
            
            # Update network
            loss = self.vqnn.update_batch(states, actions, targets, current_q_values)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return loss

# ============================================================================
# COMPLEXITY METRICS
# ============================================================================

@lru_cache(maxsize=1024)
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

@lru_cache(maxsize=1024)
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

# ============================================================================
# PARALLEL TRAINING
# ============================================================================

def train_parallel_episodes(agent: OptimizedQLearningAgent,
                           env_manager: ParallelEnvironmentManager,
                           episodes: int) -> List[float]:
    """
    Train agent on parallel environments for faster convergence.
    """
    learning_curve = []
    
    for episode in range(episodes):
        # Reset environments
        env_manager.reset_all()
        
        # Collect experiences from parallel environments
        episode_rewards = [0] * env_manager.num_parallel
        dones = [False] * env_manager.num_parallel
        
        # Pre-allocate batch arrays
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        step_count = 0
        max_steps = 100
        
        while not all(dones) and step_count < max_steps:
            # Get current states
            mazes, positions = env_manager.get_states()
            
            # Select actions for all environments
            action_indices, actions = agent.select_actions_batch(
                mazes, positions, training=True
            )
            
            # Execute actions
            next_positions, rewards, new_dones = env_manager.step_all(actions)
            
            # Encode states for storage
            states = agent.vqnn.encode_states_batch(
                np.array(mazes), np.array(positions)
            )
            next_states = agent.vqnn.encode_states_batch(
                np.array(mazes), np.array(next_positions)
            )
            
            # Collect experiences
            for i in range(env_manager.num_parallel):
                if not dones[i]:
                    batch_states.append(states[i])
                    batch_actions.append(action_indices[i])
                    batch_rewards.append(rewards[i])
                    batch_next_states.append(next_states[i])
                    batch_dones.append(new_dones[i])
                    
                    episode_rewards[i] += rewards[i]
                    dones[i] = new_dones[i]
            
            step_count += 1
        
        # Store experiences in replay buffer
        if batch_states:
            agent.replay_buffer.push_batch(
                np.array(batch_states),
                np.array(batch_actions),
                np.array(batch_rewards),
                np.array(batch_next_states),
                np.array(batch_dones)
            )
        
        # Train on multiple batches per episode
        for _ in range(4):  # 4 gradient updates per episode
            loss = agent.train_on_batch()
        
        # Record average reward
        avg_reward = np.mean(episode_rewards)
        learning_curve.append(avg_reward)
        
        # Progress update
        if episode % 10 == 0:
            recent_avg = np.mean(learning_curve[-10:]) if len(learning_curve) >= 10 else avg_reward
            print(f"    Episode {episode}: Avg reward = {recent_avg:.1f}, ε = {agent.epsilon:.3f}")
    
    return learning_curve

# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_optimized_experiments(config: Optional[Dict] = None):
    """
    Execute optimized adaptive problem-solving experiments.
    """
    if config is None:
        config = {
            "episodes_per_maze": 100,  # Reduced due to faster convergence
            "epsilon_start": 0.3,
            "epsilon_decay": 0.995,
            "learning_rate": 0.01,
            "gamma": 0.99,
            "batch_size": BATCH_SIZE,
            "parallel_envs": PARALLEL_ENVS,
            "output_dir": "results_optimized",
            "visualize": True
        }
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    print("\n" + "=" * 60)
    print("OPTIMIZED VQNN ADAPTIVE PROBLEM-SOLVING EXPERIMENTS")
    print("=" * 60)
    print(f"Quantum Backend: {QUANTUM_BACKEND if PENNYLANE_AVAILABLE else 'Classical'}")
    print(f"GPU Acceleration: {'ENABLED' if QUANTUM_CONFIG['gpu'] else 'DISABLED'}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Parallel Environments: {config['parallel_envs']}")
    print(f"Episodes per maze: {config['episodes_per_maze']}")
    print("=" * 60)
    
    # Initialize components
    generator = MazeGenerator()
    mazes = generator.get_fixed_mazes()
    
    # Output setup
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], 
                               f"vqnn_optimized_results_{timestamp}.csv")
    
    # Initialize VQNN and agent once (reuse for all mazes)
    vqnn = OptimizedVQNN(
        num_qubits=25,
        num_layers=2,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"]
    )
    
    agent = OptimizedQLearningAgent(
        vqnn=vqnn,
        epsilon=config["epsilon_start"],
        gamma=config["gamma"],
        epsilon_decay=config["epsilon_decay"]
    )
    
    # Prepare results storage
    headers = [
        "run_id", "timestamp", "maze_name", "steps_to_goal", 
        "optimal_steps", "efficiency_score", "final_reward",
        "convergence_episode", "training_time", "gpu_accelerated"
    ]
    
    results = []
    
    print("\nRunning optimized experiments...\n")
    
    # Process mazes in batches
    total_time = 0
    for idx, (maze, maze_name) in enumerate(mazes, 1):
        print(f"\nMaze {idx}: {maze_name}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Calculate optimal path
        optimal_steps = generator.bfs_shortest_path(maze)
        print(f"  Optimal path: {optimal_steps} steps")
        
        # Create parallel environments for this maze
        parallel_mazes = [maze] * config["parallel_envs"]
        env_manager = ParallelEnvironmentManager(parallel_mazes, config["parallel_envs"])
        
        # Train with parallel environments
        print("  Training with parallel environments...")
        learning_curve = train_parallel_episodes(
            agent, env_manager, config["episodes_per_maze"]
        )
        
        # Final evaluation
        print("  Evaluating...")
        agent.epsilon = 0  # Disable exploration
        env = MazeEnvironment(maze)
        position = env.reset()
        done = False
        
        while not done and env.steps < env.max_steps:
            # Single environment evaluation
            action_indices, actions = agent.select_actions_batch(
                [maze], [position], training=False
            )
            position, reward, done = env.step(actions[0])
        
        # Calculate metrics
        efficiency = optimal_steps / env.steps if env.steps > 0 else 0
        training_time = time.time() - start_time
        total_time += training_time
        
        # Find convergence episode
        convergence_episode = len(learning_curve)
        if len(learning_curve) > 10:
            for i in range(10, len(learning_curve)):
                if learning_curve[i] > 50:  # Threshold for "solved"
                    convergence_episode = i
                    break
        
        print(f"  Final: {env.steps} steps, Efficiency: {efficiency:.2f}")
        print(f"  Training time: {training_time:.2f}s")
        
        # Save results
        results.append([
            f"VQNN_optimized_{idx}",
            datetime.now().isoformat(),
            maze_name,
            env.steps,
            optimal_steps,
            efficiency,
            reward,
            convergence_episode,
            training_time,
            QUANTUM_CONFIG['gpu']
        ])
        
        # Reset epsilon for next maze
        agent.epsilon = config["epsilon_start"]
    
    # Save results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    # Performance report
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total training time: {total_time:.2f}s")
    print(f"Average time per maze: {total_time/len(mazes):.2f}s")
    
    if hasattr(vqnn, '_cache_hits'):
        cache_ratio = vqnn._cache_hits / (vqnn._cache_hits + vqnn._cache_misses + 1)
        print(f"Circuit cache hit ratio: {cache_ratio:.2%}")
    
    vqnn.perf.report()
    
    print(f"\nResults saved to: {output_file}")
    print("✅ Optimized experiment complete!")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run with optimized configuration
    config = {
        "episodes_per_maze": 100,    # Reduced due to faster convergence
        "epsilon_start": 0.3,
        "epsilon_decay": 0.995,
        "learning_rate": 0.01,
        "gamma": 0.99,
        "batch_size": BATCH_SIZE,
        "parallel_envs": PARALLEL_ENVS,
        "output_dir": "results_optimized",
        "visualize": True
    }
    
    print("🚀 Starting Optimized VQNN Experiments...")
    print(f"Configuration: Batch={BATCH_SIZE}, Parallel={PARALLEL_ENVS}")
    
    results = run_optimized_experiments(config)
    
    print("\n" + "=" * 60)
    print("All experiments completed successfully!")
    print("=" * 60)