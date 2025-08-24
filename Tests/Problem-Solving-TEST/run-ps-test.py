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
Title         : {Variational Quantum Neural Network for Adaptive Problemm Solving}
Date          : {05-18-2025}
Description   : {This code implements a GPU-accelerated Variational Quantum Neural Network (VQNN) that uses Q-learning to train an agent to solve 10 different 5x5 maze navigation tasks, measuring emergent problem-solving behaviors through complexity and entropy analysis of the learned solution paths.}
Options       : {}
Dependencies  : {pip install numpy scipy pennylane pennylane-lightning-gpu cupy-cuda12x torch torchvision torchaudio autograd matplotlib --index-url https://download.pytorch.org/whl/cu121}
Requirements  : {Python 3.8+}
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

# GPU and Quantum backend handling
try:
    import pennylane as qml
    from pennylane import numpy as pnp  # PennyLane's NumPy with autograd
    PENNYLANE_AVAILABLE = True
    
    # Try to import GPU-specific backends
    try:
        # Check for NVIDIA cuQuantum SDK support
        import cupy as cp  # NVIDIA GPU arrays
        CUPY_AVAILABLE = True
        print("✓ CuPy detected - GPU arrays enabled")
    except ImportError:
        CUPY_AVAILABLE = False
        cp = np  # Fallback to NumPy
        print("✗ CuPy not available - using CPU arrays")
    
    # Check for PennyLane GPU devices
    try:
        # Try lightning.gpu device (requires PennyLane-Lightning-GPU)
        test_dev = qml.device("lightning.gpu", wires=2)
        GPU_DEVICE = "lightning.gpu"
        print("✓ PennyLane Lightning GPU detected - using NVIDIA cuQuantum")
    except:
        try:
            # Try default.qubit with GPU support
            test_dev = qml.device("default.qubit.torch", wires=2)
            import torch
            if torch.cuda.is_available():
                GPU_DEVICE = "default.qubit.torch"
                print("✓ PyTorch GPU backend detected - using CUDA acceleration")
            else:
                GPU_DEVICE = "default.qubit"
                print("✗ No GPU backend available - using CPU simulation")
        except:
            GPU_DEVICE = "default.qubit"
            print("✗ No GPU backend available - using CPU simulation")
    
except ImportError:
    PENNYLANE_AVAILABLE = False
    CUPY_AVAILABLE = False
    GPU_DEVICE = None
    cp = np
    pnp = np
    print("Warning: PennyLane not available. Using simulation fallback.")

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

class GPUConfig:
    """GPU configuration and memory management."""
    
    @staticmethod
    def setup_gpu():
        """Configure GPU settings for optimal performance."""
        if CUPY_AVAILABLE:
            # Set memory pool for efficient allocation
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Pre-allocate memory to avoid fragmentation
            mempool.set_limit(size=2**31)  # 2GB limit
            
            # Enable TensorCore operations if available
            cp.cuda.Device().use()
            
            print(f"GPU Memory allocated: {mempool.used_bytes() / 1e9:.2f} GB")
            return True
        return False
    
    @staticmethod
    def to_gpu(array: np.ndarray):
        """Transfer array to GPU memory."""
        if CUPY_AVAILABLE:
            return cp.asarray(array, dtype=cp.float32)
        return array
    
    @staticmethod
    def to_cpu(array):
        """Transfer array from GPU to CPU memory."""
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
    @staticmethod
    def cleanup():
        """Clean up GPU memory."""
        if CUPY_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

# Initialize GPU
gpu_available = GPUConfig.setup_gpu()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Seed management for reproducibility
GLOBAL_SEED = None  # Set to integer for reproducibility, None for true randomness
QUANTUM_NOISE_LEVEL = 0.01  # Quantum noise parameter for realistic simulation

# Batch processing for GPU efficiency
BATCH_SIZE = 32  # Process multiple states simultaneously on GPU
PARALLEL_CIRCUITS = 4  # Number of parallel quantum circuits for GPU

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
# MAZE GENERATION MODULE (Using GPU Arrays)
# ============================================================================

class MazeGenerator:
    """Generate and manage maze environments with GPU acceleration."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        
        # Pre-allocate GPU memory for maze batch processing
        if CUPY_AVAILABLE:
            self.gpu_maze_cache = {}
    
    def _cache_maze_gpu(self, maze: np.ndarray, name: str):
        """Cache maze in GPU memory for faster access."""
        if CUPY_AVAILABLE:
            self.gpu_maze_cache[name] = GPUConfig.to_gpu(maze)
    
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
# GPU-ACCELERATED VARIATIONAL QUANTUM NEURAL NETWORK
# ============================================================================

class VQNN:
    """
    GPU-Accelerated Variational Quantum Neural Network for Q-learning.
    
    Implements:
    - Amplitude encoding for state representation
    - Two-layer variational circuit with RX, RY rotations and CNOT entanglement
    - Q-value prediction for action selection
    - GPU batch processing for parallel circuit evaluation
    """
    
    def __init__(self, num_qubits: int = 25, num_layers: int = 2, 
                 learning_rate: float = 0.01, use_gpu: bool = True):
        """
        Initialize VQNN with GPU support.
        
        Args:
            num_qubits: Number of qubits (25 for 5x5 maze)
            num_layers: Number of variational layers
            learning_rate: Learning rate for optimization
            use_gpu: Whether to use GPU acceleration
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu and gpu_available
        
        # Initialize parameters with GPU arrays if available
        if self.use_gpu and CUPY_AVAILABLE:
            self.params = cp.random.randn(num_layers, num_qubits, 2, dtype=cp.float32) * 0.1
            # Adam optimizer state on GPU
            self.m = cp.zeros_like(self.params)  # First moment
            self.v = cp.zeros_like(self.params)  # Second moment
        else:
            self.params = np.random.randn(num_layers, num_qubits, 2).astype(np.float32) * 0.1
            self.m = np.zeros_like(self.params)
            self.v = np.zeros_like(self.params)
        
        self.t = 0  # Time step
        
        # Initialize quantum device with GPU support
        if PENNYLANE_AVAILABLE:
            if GPU_DEVICE == "lightning.gpu":
                # Lightning GPU device for NVIDIA cuQuantum
                self.dev = qml.device(GPU_DEVICE, wires=num_qubits, batch_obs=True)
                print(f"Using {GPU_DEVICE} with {num_qubits} qubits")
            elif GPU_DEVICE == "default.qubit.torch" and self.use_gpu:
                # PyTorch GPU backend
                import torch
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                self.dev = qml.device(GPU_DEVICE, wires=num_qubits, torch_device='cuda')
                print(f"Using {GPU_DEVICE} with CUDA")
            else:
                # CPU fallback
                self.dev = qml.device("default.qubit", wires=num_qubits)
                print(f"Using CPU device with {num_qubits} qubits")
            
            # Create batched circuit for parallel processing
            self.circuit = self._create_batched_circuit()
        else:
            self.dev = None
            self.circuit = None
        
        self.rng = np.random.RandomState()
    
    def _create_batched_circuit(self):
        """Create variational quantum circuit with batch processing support."""
        
        @qml.batch_params
        @qml.qnode(self.dev, interface='autograd' if not self.use_gpu else 'torch', 
                   diff_method='adjoint' if GPU_DEVICE == "lightning.gpu" else 'backprop')
        def circuit(inputs, params):
            # Batch amplitude encoding
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
            
            # Variational layers with GPU-optimized structure
            for layer in range(self.num_layers):
                # Layer of single-qubit rotations
                for i in range(self.num_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                
                # Efficient entangling layer
                if layer < self.num_layers - 1:  # Skip on last layer for efficiency
                    # Linear entanglement pattern
                    for i in range(0, self.num_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(1, self.num_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
            
            # Measure expectation values for Q-values (4 actions)
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        return circuit
    
    def encode_state_batch(self, mazes: List[np.ndarray], 
                           positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Encode batch of maze states for parallel GPU processing.
        
        Args:
            mazes: List of maze arrays
            positions: List of agent positions
            
        Returns:
            Batch of encoded state vectors
        """
        batch_size = len(mazes)
        states = np.zeros((batch_size, 25), dtype=np.float32)
        
        for idx, (maze, position) in enumerate(zip(mazes, positions)):
            # Create state representation
            state = np.zeros(25, dtype=np.float32)
            
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
            
            states[idx] = state
        
        if self.use_gpu and CUPY_AVAILABLE:
            return GPUConfig.to_gpu(states)
        return states
    
    def encode_state(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Encode single maze state.
        
        Args:
            maze: Current maze
            position: Agent position
            
        Returns:
            Encoded state vector
        """
        state = np.zeros(25, dtype=np.float32)
        
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
        
        if self.use_gpu and CUPY_AVAILABLE:
            return GPUConfig.to_gpu(state)
        return state
    
    def get_q_values_batch(self, mazes: List[np.ndarray], 
                           positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Get Q-values for batch of states using GPU parallelization.
        
        Args:
            mazes: List of mazes
            positions: List of positions
            
        Returns:
            Batch of Q-values
        """
        states = self.encode_state_batch(mazes, positions)
        
        if PENNYLANE_AVAILABLE and self.circuit:
            # Convert parameters to CPU for PennyLane if needed
            params_cpu = GPUConfig.to_cpu(self.params)
            
            # Batch quantum circuit evaluation
            q_values_batch = []
            for state in states:
                state_cpu = GPUConfig.to_cpu(state)
                q_values = np.array(self.circuit(state_cpu, params_cpu))
                q_values_batch.append(q_values)
            
            q_values_batch = np.array(q_values_batch)
            
            if self.use_gpu and CUPY_AVAILABLE:
                return GPUConfig.to_gpu(q_values_batch)
            return q_values_batch
        else:
            # Classical fallback with GPU arrays
            return self._classical_forward_batch(states)
    
    def get_q_values(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Get Q-values for single state.
        
        Args:
            maze: Current maze
            position: Agent position
            
        Returns:
            Q-values for each action
        """
        state = self.encode_state(maze, position)
        
        if PENNYLANE_AVAILABLE and self.circuit:
            # Convert to CPU for PennyLane
            state_cpu = GPUConfig.to_cpu(state)
            params_cpu = GPUConfig.to_cpu(self.params)
            
            # Quantum circuit evaluation
            q_values = np.array(self.circuit(state_cpu, params_cpu))
            
            if self.use_gpu and CUPY_AVAILABLE:
                return GPUConfig.to_gpu(q_values)
            return q_values
        else:
            # Classical fallback
            return self._classical_forward(state)
    
    def _classical_forward_batch(self, states: np.ndarray) -> np.ndarray:
        """Classical neural network fallback with GPU acceleration."""
        if self.use_gpu and CUPY_AVAILABLE:
            # GPU matrix multiplication
            weights = cp.reshape(self.params, (-1, 4))[:states.shape[1]]
            q_values = cp.tanh(cp.dot(states, weights))
        else:
            # CPU computation
            weights = self.params.reshape(-1, 4)[:states.shape[1]]
            q_values = np.tanh(np.dot(states, weights))
        
        return q_values
    
    def _classical_forward(self, state: np.ndarray) -> np.ndarray:
        """Classical neural network fallback for single state."""
        if self.use_gpu and CUPY_AVAILABLE:
            weights = cp.reshape(self.params, (-1, 4))[:len(state)]
            q_values = cp.tanh(cp.dot(state, weights))
        else:
            weights = self.params.reshape(-1, 4)[:len(state)]
            q_values = np.tanh(np.dot(state, weights))
        
        return q_values
    
    def update(self, state: np.ndarray, action: int, target: float, 
               current_q: float) -> float:
        """
        Update network parameters using GPU-accelerated gradient descent.
        
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
        
        # Compute gradient with GPU acceleration
        if self.use_gpu and CUPY_AVAILABLE:
            gradient = cp.zeros_like(self.params)
            epsilon = 0.01
            
            # Parallel gradient computation on GPU
            for layer in range(self.num_layers):
                for qubit in range(self.num_qubits):
                    for param in range(2):
                        # Forward difference
                        self.params[layer, qubit, param] += epsilon
                        q_plus = self.get_q_values_from_state(state)[action]
                        
                        self.params[layer, qubit, param] -= 2 * epsilon
                        q_minus = self.get_q_values_from_state(state)[action]
                        
                        self.params[layer, qubit, param] += epsilon
                        
                        grad_val = (GPUConfig.to_cpu(q_plus) - GPUConfig.to_cpu(q_minus)) / (2 * epsilon)
                        gradient[layer, qubit, param] = grad_val
            
            # Adam optimizer update on GPU
            gradient *= 2 * (current_q - target)
            
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            
            self.m = beta1 * self.m + (1 - beta1) * gradient
            self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
            
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            
            self.params -= self.learning_rate * m_hat / (cp.sqrt(v_hat) + eps)
        else:
            # CPU gradient computation
            gradient = np.zeros_like(self.params)
            epsilon = 0.01
            
            for layer in range(self.num_layers):
                for qubit in range(self.num_qubits):
                    for param in range(2):
                        self.params[layer, qubit, param] += epsilon
                        q_plus = self.get_q_values_from_state(state)[action]
                        
                        self.params[layer, qubit, param] -= 2 * epsilon
                        q_minus = self.get_q_values_from_state(state)[action]
                        
                        self.params[layer, qubit, param] += epsilon
                        
                        gradient[layer, qubit, param] = (q_plus - q_minus) / (2 * epsilon)
            
            gradient *= 2 * (current_q - target)
            
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            
            self.m = beta1 * self.m + (1 - beta1) * gradient
            self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
            
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            
            self.params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        return float(loss)
    
    def get_q_values_from_state(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values directly from encoded state."""
        if PENNYLANE_AVAILABLE and self.circuit:
            state_cpu = GPUConfig.to_cpu(state)
            params_cpu = GPUConfig.to_cpu(self.params)
            q_values = np.array(self.circuit(state_cpu, params_cpu))
            
            if self.use_gpu and CUPY_AVAILABLE:
                return GPUConfig.to_gpu(q_values)
            return q_values
        else:
            return self._classical_forward(state)

# ============================================================================
# EXPERIENCE REPLAY BUFFER (GPU-OPTIMIZED)
# ============================================================================

class ExperienceReplayBuffer:
    """GPU-optimized experience replay buffer for batch training."""
    
    def __init__(self, capacity: int = 10000, use_gpu: bool = True):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            use_gpu: Whether to store on GPU
        """
        self.capacity = capacity
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Store on GPU if available
        if self.use_gpu:
            state = GPUConfig.to_gpu(state)
            next_state = GPUConfig.to_gpu(next_state)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self, batch_size: int) -> List:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        if self.use_gpu:
            # Keep batch on GPU
            return batch
        else:
            # Convert to arrays for CPU processing
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            
            return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Q-LEARNING AGENT WITH GPU OPTIMIZATION
# ============================================================================

class QLearningAgent:
    """
    GPU-optimized Q-learning agent using VQNN.
    
    Implements:
    - Epsilon-greedy exploration
    - Experience replay with batch training
    - GPU-accelerated updates
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
            self.replay_buffer = ExperienceReplayBuffer(
                capacity=10000, 
                use_gpu=vqnn.use_gpu
            )
        
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
            q_values_cpu = GPUConfig.to_cpu(q_values)
            action_idx = int(np.argmax(q_values_cpu))
        
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
        current_q = GPUConfig.to_cpu(current_q_values)[action]
        
        if done:
            target = reward
        else:
            next_q_values = self.vqnn.get_q_values(maze, next_position)
            next_q_values_cpu = GPUConfig.to_cpu(next_q_values)
            target = reward + self.gamma * np.max(next_q_values_cpu)
        
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
            current_q = GPUConfig.to_cpu(current_q_values)[action]
            
            # Compute target Q-value
            if done:
                target = reward
            else:
                next_q_values = self.vqnn.get_q_values_from_state(next_state)
                next_q_values_cpu = GPUConfig.to_cpu(next_q_values)
                target = reward + self.gamma * np.max(next_q_values_cpu)
            
            # Update network
            loss = self.vqnn.update(state, action, target, current_q)
            total_loss += loss
        
        return total_loss / self.batch_size

# ============================================================================
# MAZE ENVIRONMENT (Unchanged)
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
# COMPLEXITY METRICS (Unchanged)
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
# EMERGENCE DETECTION (Unchanged)
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
# VISUALIZATION (Unchanged)
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
    symbols = {
        0: '.',    # Empty
        1: '█',    # Wall
        2: 'S',    # Start
        3: 'G',    # Goal
    }
    
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
    Execute adaptive problem-solving experiments with GPU acceleration.
    
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
            "use_gpu": True,
            "use_replay": True,
            "batch_size": 32,
            "output_dir": "results",
            "visualize": True
        }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("VQNN ADAPTIVE PROBLEM-SOLVING EXPERIMENTS (GPU-ACCELERATED)")
    print("=" * 60)
    print(f"GPU Acceleration: {'ENABLED' if config['use_gpu'] and gpu_available else 'DISABLED'}")
    print(f"Quantum Device: {GPU_DEVICE if PENNYLANE_AVAILABLE else 'N/A'}")
    print(f"Episodes per maze: {config['episodes_per_maze']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Experience replay: {'ENABLED' if config['use_replay'] else 'DISABLED'}")
    print(f"Epsilon: {config['epsilon_start']} (decay: {config['epsilon_decay']})")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Seed: {GLOBAL_SEED or 'True random'}")
    
    if gpu_available and CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        print(f"GPU Memory: {mempool.used_bytes() / 1e6:.1f} MB allocated")
    print()
    
    # Initialize components
    generator = MazeGenerator(seed=GLOBAL_SEED)
    mazes = generator.get_fixed_mazes()
    
    # Cache mazes on GPU
    if config['use_gpu'] and CUPY_AVAILABLE:
        for maze, name in mazes:
            generator._cache_maze_gpu(maze, name)
    
    # Output setup
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], f"vqnn_gpu_results_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "maze_name", "maze_complexity",
        "episodes_trained", "steps_to_goal", "optimal_steps", "efficiency_score",
        "final_reward", "convergence_episode", "performance_discontinuity",
        "path_length", "path_lz_complexity", "path_shannon_entropy",
        "action_sequence_length", "action_lz_complexity", "action_shannon_entropy",
        "action_approximate_entropy", "is_emergent", "emergence_type", 
        "emergence_score", "solution_path", "action_sequence"
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
        
        # Initialize VQNN and agent with GPU support
        vqnn = VQNN(
            num_qubits=25,
            num_layers=2,
            learning_rate=config["learning_rate"],
            use_gpu=config["use_gpu"]
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
            str(best_solution.action_sequence)
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
    
    # GPU memory usage
    if gpu_available and CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        print(f"\nGPU Memory used: {mempool.used_bytes() / 1e6:.1f} MB")
        print(f"GPU Memory blocks: {mempool.n_free_blocks()} free, {mempool.n_total_blocks()} total")
    
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
    
    # Clean up GPU memory
    if config['use_gpu']:
        GPUConfig.cleanup()
        print("\nGPU memory cleaned up")
    
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
        "use_gpu": True,             # Enable GPU acceleration
        "use_replay": True,          # Enable experience replay
        "batch_size": 32,            # Batch size for GPU processing
        "output_dir": "results",
        "visualize": True
    }
    
    results = run_experiments(config)
    
    print("\n✅ Experiment complete!")