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
__version__ = "3.0-beta"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title         : {Scalable Variational Quantum Neural Network for Maze Navigation}
Date          : {05-18-2025}
Description   : {Scientifically rigorous implementation with proper problem-size scaling.
                Supports 9-qubit (3x3 mazes), 16-qubit (4x4 mazes), and 25-qubit (5x5 mazes)
                configurations. Uses fixed local observation window for consistency.}
Options       : {GPU acceleration via PennyLane-Lightning-GPU (NVIDIA cuQuantum SDK) or CPU fallback}
Dependencies  : {numpy scipy pennylane pennylane-lightning-gpu matplotlib}
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
import sys
from contextlib import contextmanager
from collections import deque
warnings.filterwarnings('ignore')

# ============================================================================
# USER INPUT FOR MAZE SIZE CONFIGURATION
# ============================================================================

def get_maze_configuration():
    """Get the maze size from user input with validation."""
    print("\n" + "=" * 60)
    print("QUANTUM MAZE NAVIGATION CONFIGURATION")
    print("=" * 60)
    
    while True:
        try:
            print("\nSelect maze size for the experiment:")
            print("\nAvailable configurations:")
            print("  1. 3x3 mazes (9 qubits)")
            print("     - Compact mazes with 9 cells")
            print("     - Fastest execution (~5-10 minutes)")
            print("     - Good for initial testing")
            print()
            print("  2. 4x4 mazes (16 qubits)")
            print("     - Medium mazes with 16 cells")
            print("     - Moderate runtime (~15-30 minutes)")
            print("     - Balanced complexity")
            print()
            print("  3. 5x5 mazes (25 qubits)")
            print("     - Large mazes with 25 cells")
            print("     - Longer runtime (~30-60 minutes)")
            print("     - Maximum complexity")
            print()
            print("Each configuration properly scales both maze size and qubits")
            print("Using fixed 3x3 local observation window for all sizes")
            
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            
            if choice == '1':
                maze_size = 3
                num_qubits = 9
                est_time = "5-10"
            elif choice == '2':
                maze_size = 4
                num_qubits = 16
                est_time = "15-30"
            elif choice == '3':
                maze_size = 5
                num_qubits = 25
                est_time = "30-60"
            else:
                print("⚠ Please enter 1, 2, or 3")
                continue
            
            # Confirm with user
            print(f"\nYou selected: {maze_size}x{maze_size} mazes")
            print(f"Configuration details:")
            print(f"  - Maze size: {maze_size}x{maze_size}")
            print(f"  - Qubits: {num_qubits}")
            print(f"  - Local observation: 3x3 window")
            print(f"  - Estimated runtime: {est_time} minutes")
            
            confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
            if confirm == 'y' or confirm == 'yes':
                return maze_size, num_qubits
            elif confirm == 'n' or confirm == 'no':
                continue
            else:
                print("Please enter 'y' for yes or 'n' for no.")
                
        except KeyboardInterrupt:
            print("\n\nExiting configuration...")
            sys.exit(0)
        except ValueError:
            print("⚠ Please enter a valid choice (1, 2, or 3).")

# Get user configuration
MAZE_SIZE, NUM_QUBITS = get_maze_configuration()

# Fixed observation window size (always 3x3 for consistency)
OBSERVATION_SIZE = 3
OBSERVATION_QUBITS = 9  # Always use 9 qubits for observation

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
    """Configure GPU settings for quantum simulation."""
    import os
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDNN_BENCHMARK'] = 'TRUE'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    print(f"✔ GPU settings configured for {NUM_QUBITS}-qubit simulation")

configure_gpu_settings()

# ============================================================================
# QUANTUM BACKEND DETECTION
# ============================================================================

def detect_quantum_backend():
    """Detect and configure quantum backend for simulation."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        
        # Try GPU backend first
        try:
            test_dev = qml.device("lightning.gpu", wires=2, shots=100)
            del test_dev
            
            print("=" * 60)
            print("✔ NVIDIA cuQuantum SDK detected!")
            print(f"✔ Using Lightning.GPU for {NUM_QUBITS}-qubit simulation")
            print("=" * 60)
            
            return {
                "available": True,
                "backend": "lightning.gpu",
                "interface": "autograd",
                "diff_method": "parameter-shift",
                "gpu": True,
                "pnp": pnp
            }
            
        except Exception:
            # Try lightning.qubit for CPU
            try:
                test_dev = qml.device("lightning.qubit", wires=2, shots=100)
                del test_dev
                print("=" * 60)
                print(f"✔ Using Lightning.Qubit (CPU) for {NUM_QUBITS}-qubit simulation")
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
                print(f"ℹ Using default.qubit for {NUM_QUBITS}-qubit simulation")
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
# CONFIGURATION CONSTANTS - ADJUSTED FOR MAZE SIZE
# ============================================================================

# Quantum circuit configuration
NUM_LAYERS = 2  # Two layers for better expressivity

# Adjust shots based on maze size
if MAZE_SIZE == 3:
    SHOTS = 250
    EPISODES_PER_MAZE = 60
elif MAZE_SIZE == 4:
    SHOTS = 200
    EPISODES_PER_MAZE = 80
else:  # MAZE_SIZE == 5
    SHOTS = 150
    EPISODES_PER_MAZE = 100

# Training configuration
LEARNING_RATE = 0.08
GAMMA = 0.95
EPSILON_START = 0.3
EPSILON_DECAY = 0.99
BATCH_SIZE = 16

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
# SCALABLE VQNN WITH FIXED OBSERVATION
# ============================================================================

class ScalableVQNN:
    """
    Scalable Variational Quantum Neural Network with fixed observation window.
    
    Uses a consistent 3x3 observation window regardless of maze size,
    ensuring fair comparison across different configurations.
    """
    
    def __init__(self, maze_size: int, num_qubits: int, num_layers: int = NUM_LAYERS,
                 learning_rate: float = LEARNING_RATE, shots: int = SHOTS):
        """
        Initialize VQNN with proper scaling.
        
        Args:
            maze_size: Size of the maze (3, 4, or 5)
            num_qubits: Total qubits available (9, 16, or 25)
            num_layers: Number of variational layers
            learning_rate: Learning rate for optimization
            shots: Number of measurement shots
        """
        self.maze_size = maze_size
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.shots = shots
        
        # Use ALL available qubits
        self.used_qubits = num_qubits  # Use all qubits available
        self.observation_qubits = min(9, num_qubits)  # 3x3 window encoding
        self.position_qubits = num_qubits - self.observation_qubits  # Remaining for position/context
        
        # Initialize parameters for used qubits only
        self.params = pnp.random.randn(num_layers, self.used_qubits, 2) * 0.1
        
        # Adam optimizer state
        self.m = pnp.zeros_like(self.params)
        self.v = pnp.zeros_like(self.params)
        self.t = 0
        
        # Shot noise compensation
        self.noise_scale = 1.0 / np.sqrt(shots)
        
        # Performance tracking
        self.perf = PerformanceMonitor()
        
        print(f"\nInitializing Scalable VQNN for {maze_size}x{maze_size} mazes...")
        print(f"  Configuration:")
        print(f"    - Maze size: {maze_size}x{maze_size}")
        print(f"    - Total qubits available: {num_qubits}")
        print(f"    - Observation qubits: {self.observation_qubits} (3x3 window)")
        print(f"    - Position encoding qubits: {self.position_qubits}")
        print(f"    - Active qubits: {self.used_qubits}")
        print(f"    - Layers: {num_layers}")
        print(f"    - Shots: {shots}")
        print(f"    - Backend: {QUANTUM_BACKEND}")
        
        # Initialize quantum device
        self.dev = qml.device(
            QUANTUM_BACKEND,
            wires=self.used_qubits,
            shots=shots
        )
        
        # Create circuit
        self.circuit = self._create_circuit()
        
        print(f"  ✔ Quantum circuit initialized with {self.used_qubits} active qubits")
    
    def _create_circuit(self):
        """
        Create quantum circuit with proper entanglement for the active qubits.
        """
        
        @qml.qnode(
            self.dev,
            interface=QUANTUM_INTERFACE,
            diff_method=QUANTUM_DIFF_METHOD
        )
        def circuit(inputs, params):
            """
            Quantum circuit with fixed observation encoding.
            """
            
            # Angle encoding
            for i in range(self.used_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Variational layers
            for layer in range(self.num_layers):
                # Rotation gates
                for i in range(self.used_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                
                # Entanglement layer
                # Linear chain
                for i in range(self.used_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Wrap-around for circular entanglement
                if self.used_qubits > 2:
                    qml.CNOT(wires=[self.used_qubits - 1, 0])
                
                # Additional entanglement for more qubits
                if self.used_qubits >= 9:
                    # Cross connections
                    for i in range(0, self.used_qubits - 3, 2):
                        qml.CNOT(wires=[i, i + 3])
            
            # Measure first 4 qubits for Q-values (4 actions)
            return [qml.expval(qml.PauliZ(i)) for i in range(min(4, self.used_qubits))]
        
        return circuit
    
    def get_local_observation(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Extract 3x3 local observation window around agent position.
        
        Args:
            maze: Full maze
            position: Agent position
            
        Returns:
            3x3 observation window (flattened)
        """
        row, col = position
        observation = np.zeros((3, 3), dtype=np.float32)
        
        # Extract 3x3 window centered on agent
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = row + dr, col + dc
                
                if 0 <= r < self.maze_size and 0 <= c < self.maze_size:
                    if maze[r, c] == WALL:
                        observation[dr + 1, dc + 1] = -1.0
                    elif maze[r, c] == GOAL:
                        observation[dr + 1, dc + 1] = 1.0
                    else:
                        observation[dr + 1, dc + 1] = 0.0
                else:
                    # Out of bounds = wall
                    observation[dr + 1, dc + 1] = -1.0
        
        # Agent is always at center
        observation[1, 1] = 0.5
        
        return observation.flatten()
    
    def encode_state(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Encode maze state into quantum state with fixed observation window.
        
        Args:
            maze: Full maze
            position: Agent position
            
        Returns:
            Encoded quantum state
        """
        state = pnp.zeros(self.used_qubits, dtype=np.float32)
        
        # Get 3x3 observation window (9 values)
        observation = self.get_local_observation(maze, position)
        
        # Encode observation into first 9 qubits
        for i in range(min(self.observation_qubits, len(observation))):
            state[i] = observation[i]
        
        # Use ALL remaining qubits for additional encoding
        if self.position_qubits > 0:
            row, col = position
            idx = self.observation_qubits
            
            # Basic position (2 qubits)
            if idx < self.used_qubits:
                state[idx] = (row / (self.maze_size - 1)) * 2 - 1
                idx += 1
            
            if idx < self.used_qubits:
                state[idx] = (col / (self.maze_size - 1)) * 2 - 1
                idx += 1
            
            # Goal-related features
            goal_pos = np.where(maze == GOAL)
            if len(goal_pos[0]) > 0:
                goal_row, goal_col = goal_pos[0][0], goal_pos[1][0]
                
                # Manhattan distance (1 qubit)
                if idx < self.used_qubits:
                    manhattan_dist = abs(goal_row - row) + abs(goal_col - col)
                    state[idx] = manhattan_dist / (2 * self.maze_size)
                    idx += 1
                
                # Euclidean distance (1 qubit)
                if idx < self.used_qubits:
                    euclidean_dist = np.sqrt((goal_row - row)**2 + (goal_col - col)**2)
                    state[idx] = euclidean_dist / (self.maze_size * np.sqrt(2))
                    idx += 1
                
                # Direction to goal (1 qubit)
                if idx < self.used_qubits:
                    angle_to_goal = np.arctan2(goal_row - row, goal_col - col) / np.pi
                    state[idx] = angle_to_goal
                    idx += 1
                
                # Goal direction components (2 qubits)
                if idx < self.used_qubits:
                    state[idx] = (goal_row - row) / self.maze_size
                    idx += 1
                
                if idx < self.used_qubits:
                    state[idx] = (goal_col - col) / self.maze_size
                    idx += 1
            
            # Spatial features for remaining qubits
            if idx < self.used_qubits:
                # Quadrant encoding
                state[idx] = 1.0 if row < self.maze_size/2 else -1.0
                idx += 1
            
            if idx < self.used_qubits:
                state[idx] = 1.0 if col < self.maze_size/2 else -1.0
                idx += 1
            
            # Distance from walls/boundaries
            if idx < self.used_qubits:
                state[idx] = min(row, self.maze_size - 1 - row) / (self.maze_size/2)
                idx += 1
            
            if idx < self.used_qubits:
                state[idx] = min(col, self.maze_size - 1 - col) / (self.maze_size/2)
                idx += 1
            
            # Fill remaining qubits with spatial patterns
            while idx < self.used_qubits:
                # Create diverse spatial encodings
                pattern_idx = idx - self.observation_qubits
                
                if pattern_idx % 4 == 0:
                    # Diagonal position
                    state[idx] = (row + col) / (2 * self.maze_size) - 0.5
                elif pattern_idx % 4 == 1:
                    # Anti-diagonal position
                    state[idx] = (row - col) / (2 * self.maze_size)
                elif pattern_idx % 4 == 2:
                    # Radial distance from center
                    center = self.maze_size / 2
                    radial = np.sqrt((row - center)**2 + (col - center)**2)
                    state[idx] = radial / (self.maze_size * np.sqrt(2))
                else:
                    # Sinusoidal spatial encoding
                    state[idx] = np.sin(2 * np.pi * (row + col * pattern_idx) / self.maze_size)
                
                idx += 1
        
        # Normalize
        norm = pnp.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        return state
    
    def get_q_values(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Get Q-values using quantum circuit."""
        state = self.encode_state(maze, position)
        
        with timer(f"Circuit evaluation", verbose=False):
            q_values = pnp.array(self.circuit(state, self.params))
            
            # Add small noise for exploration
            if self.t > 0:
                noise = pnp.random.normal(0, self.noise_scale * 0.5, size=4)
                q_values = q_values + noise
            
            self.perf.record("circuit_eval_time", time.time())
        
        return q_values
    
    def get_q_values_from_state(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values directly from encoded state."""
        q_values = pnp.array(self.circuit(state, self.params))
        
        if self.t > 0:
            noise = pnp.random.normal(0, self.noise_scale * 0.5, size=4)
            q_values = q_values + noise
        
        return q_values
    
    def update(self, state: np.ndarray, action: int, target: float, 
               current_q: float) -> float:
        """Update parameters using gradient estimation."""
        self.t += 1
        loss = (target - current_q) ** 2
        
        # Update frequency
        update_freq = 3
        if self.t % update_freq != 0:
            return float(loss)
        
        gradient = pnp.zeros_like(self.params)
        shift = np.pi / 2
        
        with timer("Gradient computation", verbose=False):
            # Sample a subset of parameters for efficiency
            sample_rate = 0.3
            num_params = self.num_layers * self.used_qubits * 2
            sample_size = max(4, int(sample_rate * num_params))
            
            param_indices = []
            for _ in range(sample_size):
                l = np.random.randint(self.num_layers)
                q = np.random.randint(self.used_qubits)
                p = np.random.randint(2)
                param_indices.append((l, q, p))
            
            for layer, qubit, param_idx in param_indices:
                # Parameter-shift gradient estimation
                self.params[layer, qubit, param_idx] += shift
                q_plus = self.get_q_values_from_state(state)[action]
                
                self.params[layer, qubit, param_idx] -= 2 * shift
                q_minus = self.get_q_values_from_state(state)[action]
                
                self.params[layer, qubit, param_idx] += shift
                
                gradient[layer, qubit, param_idx] = (q_plus - q_minus) / 2.0
                gradient[layer, qubit, param_idx] /= sample_rate
        
        gradient *= 2 * (current_q - target)
        
        # Adam optimizer
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
        
        m_hat = self.m / (1 - beta1 ** (self.t // update_freq))
        v_hat = self.v / (1 - beta2 ** (self.t // update_freq))
        
        self.params -= self.learning_rate * m_hat / (pnp.sqrt(v_hat) + eps)
        
        # Clip parameters
        self.params = pnp.clip(self.params, -np.pi, np.pi)
        
        self.perf.record("loss", float(loss))
        return float(loss)

# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ExperienceReplayBuffer:
    """Lightweight experience replay buffer."""
    
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self, batch_size: int) -> List:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """Q-learning agent for scalable VQNN."""
    
    def __init__(self, vqnn: ScalableVQNN, epsilon: float = EPSILON_START, 
                 gamma: float = GAMMA, epsilon_decay: float = EPSILON_DECAY,
                 use_replay: bool = True, batch_size: int = BATCH_SIZE):
        self.vqnn = vqnn
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.use_replay = use_replay
        self.batch_size = batch_size
        
        if use_replay:
            self.replay_buffer = ExperienceReplayBuffer(capacity=5000)
        
        self.rng = np.random.RandomState()
    
    def select_action(self, maze: np.ndarray, position: Tuple[int, int],
                     training: bool = True) -> Tuple[int, str]:
        if training and self.rng.random() < self.epsilon:
            action_idx = self.rng.randint(4)
        else:
            q_values = self.vqnn.get_q_values(maze, position)
            action_idx = int(pnp.argmax(q_values))
        
        return action_idx, ACTIONS[action_idx]
    
    def train_step(self, maze: np.ndarray, position: Tuple[int, int],
                  action: int, reward: float, next_position: Tuple[int, int],
                  done: bool) -> float:
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
        batch = self.replay_buffer.sample_batch(self.batch_size)
        total_loss = 0.0
        
        sub_batch_size = 4
        
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
# SCALABLE MAZE ENVIRONMENT
# ============================================================================

class ScalableMazeEnvironment:
    """Scalable maze environment for different sizes."""
    
    def __init__(self, maze: np.ndarray, maze_size: int):
        self.maze = maze.copy()
        self.original_maze = maze.copy()
        self.maze_size = maze_size
        self.start_pos = self.find_position(maze, START)
        self.goal_pos = self.find_position(maze, GOAL)
        self.agent_pos = self.start_pos
        self.steps = 0
        self.max_steps = maze_size * maze_size * 4  # Scale with maze size
        self.path = [self.start_pos]
        self.actions = []
    
    @staticmethod
    def find_position(maze: np.ndarray, marker: int) -> Tuple[int, int]:
        pos = np.where(maze == marker)
        if len(pos[0]) > 0:
            return (pos[0][0], pos[1][0])
        return (0, 0)
    
    def reset(self) -> Tuple[int, int]:
        self.maze = self.original_maze.copy()
        self.agent_pos = self.start_pos
        self.steps = 0
        self.path = [self.start_pos]
        self.actions = []
        return self.agent_pos
    
    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        self.steps += 1
        self.actions.append(action)
        
        dr, dc = ACTION_DELTAS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        
        # Check bounds
        if not (0 <= new_row < self.maze_size and 0 <= new_col < self.maze_size):
            return self.agent_pos, -10, self.steps >= self.max_steps
        
        # Check wall
        if self.maze[new_row, new_col] == WALL:
            return self.agent_pos, -10, self.steps >= self.max_steps
        
        # Move agent
        self.agent_pos = (new_row, new_col)
        self.path.append(self.agent_pos)
        
        # Check goal
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 100, True
        else:
            return self.agent_pos, -1, self.steps >= self.max_steps

# ============================================================================
# SCALABLE MAZE GENERATION
# ============================================================================

class ScalableMazeGenerator:
    """Generate mazes of different sizes."""
    
    @staticmethod
    def get_fixed_mazes(maze_size: int) -> List[Tuple[np.ndarray, str]]:
        """Generate fixed mazes for the specified size."""
        if maze_size == 3:
            return ScalableMazeGenerator._get_3x3_mazes()
        elif maze_size == 4:
            return ScalableMazeGenerator._get_4x4_mazes()
        else:  # maze_size == 5
            return ScalableMazeGenerator._get_5x5_mazes()
    
    @staticmethod
    def _get_3x3_mazes() -> List[Tuple[np.ndarray, str]]:
        """Generate 10 fixed 3x3 mazes."""
        mazes = []
        
        configs = [
            ("simple_3x3", [[2,0,1],[0,0,3],[1,0,0]]),
            ("corner_3x3", [[2,0,0],[1,1,0],[0,0,3]]),
            ("spiral_3x3", [[2,0,0],[1,1,0],[3,0,0]]),
            ("center_3x3", [[2,0,1],[0,1,0],[0,0,3]]),
            ("zigzag_3x3", [[2,1,0],[0,1,0],[0,0,3]]),
            ("edge_3x3", [[2,0,0],[0,1,0],[0,0,3]]),
            ("diag_3x3", [[2,0,1],[0,0,1],[1,0,3]]),
            ("open_3x3", [[2,0,0],[0,0,0],[0,0,3]]),
            ("barrier_3x3", [[2,0,0],[1,1,0],[0,0,3]]),
            ("complex_3x3", [[2,1,0],[0,1,0],[0,0,3]])
        ]
        
        for name, layout in configs:
            maze = np.array(layout, dtype=np.float32)
            mazes.append((maze, name))
        
        return mazes
    
    @staticmethod
    def _get_4x4_mazes() -> List[Tuple[np.ndarray, str]]:
        """Generate 10 fixed 4x4 mazes."""
        mazes = []
        
        configs = [
            ("simple_4x4", [[2,0,1,1],[0,0,1,1],[1,0,0,0],[1,1,0,3]]),
            ("spiral_4x4", [[2,0,0,0],[1,1,1,0],[0,0,0,0],[0,1,1,3]]),
            ("rooms_4x4", [[2,0,1,0],[0,0,1,0],[1,0,0,0],[0,0,1,3]]),
            ("zigzag_4x4", [[2,1,0,0],[0,1,0,1],[0,0,0,1],[1,1,0,3]]),
            ("cross_4x4", [[2,0,1,0],[0,0,1,0],[1,1,0,1],[0,0,0,3]]),
            ("corners_4x4", [[2,0,0,1],[1,0,0,0],[0,0,0,1],[0,1,0,3]]),
            ("maze_4x4", [[2,0,1,0],[1,0,1,0],[0,0,0,0],[0,1,1,3]]),
            ("open_4x4", [[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,3]]),
            ("dense_4x4", [[2,1,0,1],[0,1,0,1],[0,0,0,0],[1,0,1,3]]),
            ("path_4x4", [[2,0,0,1],[1,1,0,1],[0,0,0,0],[0,1,1,3]])
        ]
        
        for name, layout in configs:
            maze = np.array(layout, dtype=np.float32)
            mazes.append((maze, name))
        
        return mazes
    
    @staticmethod
    def _get_5x5_mazes() -> List[Tuple[np.ndarray, str]]:
        """Generate 10 fixed 5x5 mazes (original mazes)."""
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
    def bfs_shortest_path(maze: np.ndarray, maze_size: int) -> int:
        """Find shortest path using BFS."""
        from collections import deque
        
        start = ScalableMazeEnvironment.find_position(maze, START)
        goal = ScalableMazeEnvironment.find_position(maze, GOAL)
        
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (row, col), dist = queue.popleft()
            
            if (row, col) == goal:
                return dist
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < maze_size and 0 <= new_col < maze_size and
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

# ============================================================================
# SIZE-AWARE EMERGENCE DETECTION
# ============================================================================

def detect_emergence(solution: MazeSolution, optimal_steps: int,
                    learning_curve: List[float], maze_size: int) -> Dict[str, Any]:
    """
    Detect emergent behavior with size-appropriate criteria.
    """
    efficiency = solution.efficiency_score
    
    # Detect performance discontinuity
    solution.performance_discontinuity = False
    solution.convergence_episode = len(learning_curve)
    
    if len(learning_curve) > 20:
        window = 10
        for i in range(window, len(learning_curve) - window):
            before = np.mean(learning_curve[i-window:i])
            after = np.mean(learning_curve[i:i+window])
            if after - before > 30:  # Significant jump
                solution.performance_discontinuity = True
                solution.convergence_episode = i
                break
    
    # Size-adjusted thresholds
    if maze_size == 3:
        perfect_threshold = 0.90
        good_threshold = 0.75
        convergence_threshold = 30
    elif maze_size == 4:
        perfect_threshold = 0.85
        good_threshold = 0.70
        convergence_threshold = 40
    else:  # maze_size == 5
        perfect_threshold = 0.80
        good_threshold = 0.65
        convergence_threshold = 50
    
    is_emergent = False
    emergence_type = "none"
    emergence_score = 0.0
    
    # Evaluate emergence criteria
    criteria_met = 0
    
    if efficiency >= perfect_threshold:
        criteria_met += 2
        if solution.convergence_episode < convergence_threshold:
            emergence_type = "perfect_navigation"
            emergence_score = 1.0
            is_emergent = True
    elif efficiency >= good_threshold:
        criteria_met += 1
        if solution.performance_discontinuity:
            emergence_type = "sudden_insight"
            emergence_score = 0.8
            is_emergent = True
        elif solution.convergence_episode < convergence_threshold:
            emergence_type = "efficient_navigation"
            emergence_score = 0.7
            is_emergent = True
    
    # Check for consistent performance
    if len(learning_curve) >= 20:
        final_rewards = learning_curve[-20:]
        if np.std(final_rewards) < 15 and np.mean(final_rewards) > 70:
            criteria_met += 0.5
            if not is_emergent and efficiency >= 0.6:
                emergence_type = "stable_navigation"
                emergence_score = 0.5
                is_emergent = True
    
    return {
        "is_emergent": is_emergent,
        "emergence_type": emergence_type,
        "emergence_score": emergence_score,
        "efficiency_score": efficiency,
        "convergence_episode": solution.convergence_episode,
        "criteria_met": criteria_met
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_solution(maze: np.ndarray, path: List[Tuple[int, int]], 
                      maze_name: str, maze_size: int):
    """Visualize the solution path."""
    print(f"\n{maze_name} Solution:")
    print("-" * (maze_size * 2 + 3))
    
    for row in range(maze_size):
        line = ""
        for col in range(maze_size):
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
    """Execute scalable quantum maze navigation experiments."""
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
    print(f"SCALABLE QUANTUM MAZE NAVIGATION EXPERIMENTS")
    print("=" * 60)
    print(f"Maze size: {MAZE_SIZE}x{MAZE_SIZE}")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Observation window: 3x3 (fixed)")
    print(f"Quantum Backend: {QUANTUM_BACKEND}")
    print(f"GPU Acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")
    print(f"Shots: {SHOTS}")
    print(f"Episodes per maze: {config['episodes_per_maze']}")
    print("=" * 60)
    print()
    
    generator = ScalableMazeGenerator()
    mazes = generator.get_fixed_mazes(MAZE_SIZE)
    
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], 
                              f"vqnn_results_{NUM_QUBITS}q_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "maze_name", "maze_size", "maze_complexity",
        "episodes_trained", "steps_to_goal", "optimal_steps", "efficiency_score",
        "final_reward", "convergence_episode", "performance_discontinuity",
        "path_length", "path_lz_complexity", "path_shannon_entropy",
        "action_sequence_length", "action_lz_complexity", "action_shannon_entropy",
        "is_emergent", "emergence_type", "emergence_score",
        "solution_path", "action_sequence",
        "quantum_backend", "gpu_accelerated", "num_qubits", "observation_qubits",
        "shots", "training_time_minutes", "criteria_met"
    ]
    
    results = []
    total_start = time.time()
    
    print(f"Running experiments on {MAZE_SIZE}x{MAZE_SIZE} mazes...\n")
    
    # Initialize VQNN once
    vqnn = ScalableVQNN(
        maze_size=MAZE_SIZE,
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
        
        optimal_steps = generator.bfs_shortest_path(maze, MAZE_SIZE)
        print(f"Optimal path: {optimal_steps} steps")
        
        # Create new agent for each maze
        agent = QLearningAgent(
            vqnn=vqnn,
            epsilon=config["epsilon_start"],
            gamma=config["gamma"],
            epsilon_decay=config["epsilon_decay"],
            use_replay=config["use_replay"],
            batch_size=config["batch_size"]
        )
        
        env = ScalableMazeEnvironment(maze, MAZE_SIZE)
        learning_curve = []
        best_solution = None
        best_steps = float('inf')
        
        print(f"Training with {vqnn.used_qubits}-qubit quantum circuit...")
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
        
        eval_steps = []
        for _ in range(3):
            env.reset()
            position = env.start_pos
            done = False
            
            while not done and env.steps < env.max_steps:
                action_idx, action = agent.select_action(maze, position, training=False)
                position, reward, done = env.step(action)
            
            if done and position == env.goal_pos:
                eval_steps.append(env.steps)
        
        if eval_steps:
            final_steps = int(np.median(eval_steps))
        else:
            final_steps = env.steps if best_solution is None else best_solution.steps_to_goal
        
        if best_solution is None or final_steps < best_solution.steps_to_goal:
            best_solution = MazeSolution(
                maze_id=maze_name,
                steps_to_goal=final_steps,
                optimal_steps=optimal_steps,
                efficiency_score=optimal_steps / final_steps if final_steps > 0 else 0,
                solution_path=env.path,
                action_sequence=env.actions,
                learning_curve=learning_curve,
                final_reward=learning_curve[-1] if learning_curve else 0,
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
        
        # Emergence detection
        emergence = detect_emergence(best_solution, optimal_steps, learning_curve, MAZE_SIZE)
        
        training_time = (time.time() - maze_start) / 60
        
        print(f"\nResults:")
        print(f"  Steps to goal: {best_solution.steps_to_goal}")
        print(f"  Optimal steps: {optimal_steps}")
        print(f"  Efficiency: {best_solution.efficiency_score:.2%}")
        print(f"  Emergent: {emergence['is_emergent']} ({emergence['emergence_type']})")
        print(f"  Training time: {training_time:.1f} minutes")
        
        if config.get("visualize", True) and emergence['is_emergent']:
            visualize_solution(maze, best_solution.solution_path, maze_name, MAZE_SIZE)
        
        # Save results
        results.append([
            f"VQNN_{MAZE_SIZE}x{MAZE_SIZE}_maze_{idx}",
            datetime.now().isoformat(),
            maze_name,
            f"{MAZE_SIZE}x{MAZE_SIZE}",
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
            emergence["is_emergent"],
            emergence["emergence_type"],
            emergence["emergence_score"],
            str(best_solution.solution_path),
            str(best_solution.action_sequence),
            QUANTUM_BACKEND,
            GPU_AVAILABLE,
            NUM_QUBITS,
            vqnn.observation_qubits,
            SHOTS,
            training_time,
            emergence["criteria_met"]
        ])
    
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
    avg_efficiency = np.mean([r[8] for r in results])
    
    print(f"Maze size: {MAZE_SIZE}x{MAZE_SIZE}")
    print(f"Total mazes: {len(results)}")
    print(f"Emergent solutions: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    print(f"Average efficiency: {avg_efficiency:.3f}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Average time per maze: {total_time/len(mazes):.1f} minutes")
    print(f"Quantum backend: {QUANTUM_BACKEND}")
    print(f"Qubits used: {vqnn.used_qubits}/{NUM_QUBITS}")
    print(f"Observation window: 3x3")
    
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
    print("SCALABLE QUANTUM MAZE NAVIGATION EXPERIMENT")
    print("="*60)
    print("\nThis experiment tests a Variational Quantum Neural Network")
    print("with properly scaled maze sizes and qubit allocations.")
    print("\nEach configuration uses:")
    print("  • N² cells for NxN mazes")
    print("  • N² qubits for quantum processing")
    print("  • Fixed 3x3 observation window for consistency")
    print("\nThis ensures scientifically valid comparisons across scales.")
    
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
    
    print(f"\n✅ {MAZE_SIZE}x{MAZE_SIZE} maze experiment complete!")
    print("=" * 60)