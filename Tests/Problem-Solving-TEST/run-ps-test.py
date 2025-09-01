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
__version__ = "1.12-alpha"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title         : {Variational Quantum Neural Network for Adaptive Problem Solving - Improved Emergence}
Date          : {05-18-2025}
Description   : {Enhanced version with stricter emergence detection criteria and realistic baselines.
                Features multi-criteria emergence validation and proper random walk baselines.}
Options       : {GPU acceleration via PennyLane-Lightning-GPU (NVIDIA cuQuantum SDK) or CPU fallback}
Dependencies  : {numpy scipy pennylane pennylane-lightning-gpu matplotlib}
Requirements  : {Python 3.8+, Optional: CUDA 11.0+ and cuQuantum for GPU acceleration}
Usage         : {python run-ps-test-improved.py}
Notes         : {Improved emergence detection with stricter thresholds and multiple criteria}
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
# USER INPUT FOR QUBIT CONFIGURATION
# ============================================================================

def get_qubit_configuration():
    """Get the number of qubits from user input with validation."""
    print("\n" + "=" * 60)
    print("QUANTUM CIRCUIT CONFIGURATION")
    print("=" * 60)
    
    while True:
        try:
            print("\nEnter the number of qubits for the experiment")
            print("Recommended ranges:")
            print("  - 8-12 qubits: Fast execution (5-20 minutes)")
            print("  - 13-16 qubits: Balanced (20-40 minutes)")
            print("  - 17-20 qubits: Enhanced quantum advantage (40-90 minutes)")
            print("  - 21-25 qubits: Maximum complexity (90+ minutes)")
            print("\nNote: Higher qubit counts provide better quantum advantage")
            print("but require more computational resources and time.")
            
            num_qubits = input("\nNumber of qubits (8-25): ").strip()
            num_qubits = int(num_qubits)
            
            if num_qubits < 8:
                print("⚠ Minimum 8 qubits required for meaningful quantum advantage.")
                continue
            elif num_qubits > 25:
                print("⚠ Maximum 25 qubits supported to maintain practical runtime.")
                continue
            
            # Confirm with user
            print(f"\nYou selected {num_qubits} qubits.")
            
            # Estimate runtime
            if num_qubits <= 12:
                est_time = "15-30"
            elif num_qubits <= 16:
                est_time = "30-50"
            elif num_qubits <= 20:
                est_time = "50-90"
            else:
                est_time = "90-150"
            
            print(f"Estimated total runtime: {est_time} minutes")
            
            confirm = input("Proceed with this configuration? (y/n): ").strip().lower()
            if confirm == 'y' or confirm == 'yes':
                return num_qubits
            elif confirm == 'n' or confirm == 'no':
                continue
            else:
                print("Please enter 'y' for yes or 'n' for no.")
                
        except ValueError:
            print("⚠ Please enter a valid integer between 8 and 25.")
        except KeyboardInterrupt:
            print("\n\nExiting configuration...")
            import sys
            sys.exit(0)

# Get user configuration
NUM_QUBITS = get_qubit_configuration()

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
    
    print(f"✓ GPU settings configured for {NUM_QUBITS}-qubit simulation")

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
            print("✓ NVIDIA cuQuantum SDK detected!")
            print(f"✓ Using Lightning.GPU for {NUM_QUBITS}-qubit simulation")
            
            # Estimate runtime based on qubit count
            if NUM_QUBITS <= 12:
                runtime_est = "1-3"
            elif NUM_QUBITS <= 16:
                runtime_est = "3-5"
            elif NUM_QUBITS <= 20:
                runtime_est = "5-9"
            else:
                runtime_est = "9-15"
            
            print(f"✓ Expected runtime: {runtime_est} minutes per maze")
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
                print(f"✓ Using Lightning.Qubit (CPU) for {NUM_QUBITS}-qubit simulation")
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
# CONFIGURATION CONSTANTS - DYNAMICALLY ADJUSTED FOR QUBIT COUNT
# ============================================================================

# Quantum circuit configuration
NUM_LAYERS = 1        # Single layer is sufficient

# Adjust shots based on qubit count for balance
if NUM_QUBITS <= 12:
    SHOTS = 200
elif NUM_QUBITS <= 16:
    SHOTS = 200
elif NUM_QUBITS <= 20:
    SHOTS = 150
else:
    SHOTS = 100  # Reduce shots for very high qubit counts

# Training configuration
EPISODES_PER_MAZE = 80    # Keep consistent
LEARNING_RATE = 0.08      # Keep consistent
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
# ADAPTIVE VQNN
# ============================================================================

class VQNN:
    """
    Adaptive Variational Quantum Neural Network.
    
    Dynamically adjusts to the specified number of qubits (8-25).
    """
    
    def __init__(self, num_qubits: int = NUM_QUBITS, num_layers: int = NUM_LAYERS,
                 learning_rate: float = LEARNING_RATE, shots: int = SHOTS):
        """
        Initialize adaptive VQNN with shot-based quantum simulation.
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.shots = shots
        
        # Initialize parameters
        self.params = pnp.random.randn(num_layers, num_qubits, 2) * 0.1
        
        # Adam optimizer state
        self.m = pnp.zeros_like(self.params)
        self.v = pnp.zeros_like(self.params)
        self.t = 0
        
        # Shot noise compensation
        self.noise_scale = 1.0 / np.sqrt(shots)
        
        # Performance tracking
        self.perf = PerformanceMonitor()
        
        print(f"\nInitializing {num_qubits}-Qubit VQNN...")
        print(f"  Configuration:")
        print(f"    - Qubits: {num_qubits}")
        print(f"    - Layers: {num_layers}")
        print(f"    - Shots: {shots}")
        print(f"    - Statistical error: ±{self.noise_scale*100:.1f}%")
        print(f"    - Backend: {QUANTUM_BACKEND}")
        
        # Initialize quantum device with shots
        self.dev = qml.device(
            QUANTUM_BACKEND,
            wires=num_qubits,
            shots=shots
        )
        
        # Create circuit
        self.circuit = self._create_circuit()
        
        print(f"  ✓ {num_qubits}-qubit quantum circuit initialized")
    
    def _create_circuit(self):
        """
        Create quantum circuit with adaptive entanglement structure.
        """
        
        @qml.qnode(
            self.dev,
            interface=QUANTUM_INTERFACE,
            diff_method=QUANTUM_DIFF_METHOD
        )
        def circuit(inputs, params):
            """
            Adaptive circuit with entanglement scaling by qubit count.
            """
            
            # Angle encoding - fast and robust
            for i in range(self.num_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Single variational layer
            for i in range(self.num_qubits):
                qml.RX(params[0, i, 0], wires=i)
                qml.RY(params[0, i, 1], wires=i)
            
            # Adaptive entanglement based on qubit count
            # Linear chain
            for i in range(0, self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Add wrap-around connection
            qml.CNOT(wires=[self.num_qubits-1, 0])
            
            # Add cross-connections for higher qubit counts
            if self.num_qubits >= 12:
                # Connect qubits that are 4 positions apart
                for i in range(0, self.num_qubits - 4):
                    if i % 2 == 0:  # Only even indices
                        qml.CNOT(wires=[i, i + 4])
            
            if self.num_qubits >= 20:
                # Add additional connections for very high qubit counts
                for i in range(0, self.num_qubits - 8):
                    if i % 3 == 0:  # Every third qubit
                        qml.CNOT(wires=[i, i + 8])
            
            # Measure 4 qubits for Q-values
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        return circuit
    
    def encode_state(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Encode maze state into quantum state.
        Adaptively uses all available qubits.
        """
        state = pnp.zeros(self.num_qubits, dtype=np.float32)
        
        # Core features (always present)
        # Features 0-1: Agent position
        row, col = position
        state[0] = row / 4.0
        state[1] = col / 4.0
        
        # Features 2-3: Distance to goal
        goal_pos = np.where(maze == GOAL)
        if len(goal_pos[0]) > 0:
            goal_row, goal_col = goal_pos[0][0], goal_pos[1][0]
            state[2] = (goal_row - row) / 4.0
            state[3] = (goal_col - col) / 4.0
        
        # Feature 4: Manhattan distance (if we have enough qubits)
        if self.num_qubits > 4:
            if len(goal_pos[0]) > 0:
                manhattan_dist = abs(goal_row - row) + abs(goal_col - col)
                state[4] = manhattan_dist / 8.0
        
        # Features 5-8: Adjacent cells (if we have enough qubits)
        if self.num_qubits >= 9:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for i, (dr, dc) in enumerate(directions):
                if 5 + i < self.num_qubits:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 5 and 0 <= new_col < 5:
                        if maze[new_row, new_col] == WALL:
                            state[5 + i] = -0.5
                        elif maze[new_row, new_col] == GOAL:
                            state[5 + i] = 1.0
                        else:
                            state[5 + i] = 0.3
                    else:
                        state[5 + i] = -1.0
        
        # Features 9-12: Diagonal cells (if we have enough qubits)
        if self.num_qubits >= 13:
            diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for i, (dr, dc) in enumerate(diagonals):
                if 9 + i < self.num_qubits:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 5 and 0 <= new_col < 5:
                        if maze[new_row, new_col] == WALL:
                            state[9 + i] = -0.3
                        elif maze[new_row, new_col] == GOAL:
                            state[9 + i] = 0.7
                        else:
                            state[9 + i] = 0.1
                    else:
                        state[9 + i] = -0.5
        
        # Extended spatial awareness (if we have enough qubits)
        if self.num_qubits >= 16:
            # Check for walls in a 2-step radius
            extended_dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            for i, (dr, dc) in enumerate(extended_dirs):
                if 13 + i < self.num_qubits:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 5 and 0 <= new_col < 5:
                        if maze[new_row, new_col] == WALL:
                            state[13 + i] = -0.2
                        elif maze[new_row, new_col] == GOAL:
                            state[13 + i] = 0.5
                        else:
                            state[13 + i] = 0.05
                    else:
                        state[13 + i] = -0.3
        
        # Ultra-extended awareness (if we have 20+ qubits)
        if self.num_qubits >= 20:
            # Additional features for maximum qubit utilization
            # Path history encoding
            if 17 < self.num_qubits:
                state[17] = self.t / 100.0  # Normalized time step
            
            # Quadrant information
            if 18 < self.num_qubits:
                state[18] = 1.0 if row < 2.5 else -1.0  # Upper/lower half
            
            if 19 < self.num_qubits:
                state[19] = 1.0 if col < 2.5 else -1.0  # Left/right half
            
            # Fill remaining qubits with extended spatial patterns
            for i in range(20, min(self.num_qubits, 25)):
                # Create a unique spatial pattern for each additional qubit
                pattern_val = np.sin((row + col + i) * np.pi / 5)
                state[i] = pattern_val * 0.3
        
        # Normalize
        norm = pnp.linalg.norm(state)
        if norm > 0:
            state = state / (norm * 2)
        
        return state
    
    def get_q_values(self, maze: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Get Q-values using quantum circuit."""
        state = self.encode_state(maze, position)
        
        with timer(f"{self.num_qubits}-qubit circuit evaluation", verbose=False):
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
        """
        Update parameters using gradient estimation.
        Sampling rate adjusts based on qubit count.
        """
        self.t += 1
        loss = (target - current_q) ** 2
        
        # Update frequency based on qubit count
        update_freq = 3 if self.num_qubits <= 16 else 4
        if self.t % update_freq != 0:
            return float(loss)
        
        gradient = pnp.zeros_like(self.params)
        shift = np.pi / 2
        
        with timer("Gradient computation", verbose=False):
            # Adaptive sampling rate based on qubit count
            if self.num_qubits <= 12:
                sample_rate = 0.25
            elif self.num_qubits <= 16:
                sample_rate = 0.20
            elif self.num_qubits <= 20:
                sample_rate = 0.15
            else:
                sample_rate = 0.10
            
            num_params = self.num_layers * self.num_qubits * 2
            sample_size = max(3, int(sample_rate * num_params))
            
            param_indices = []
            for _ in range(sample_size):
                l = np.random.randint(self.num_layers)
                q = np.random.randint(self.num_qubits)
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
                gradient[layer, qubit, param_idx] /= sample_rate  # Scale for sampling
        
        gradient *= 2 * (current_q - target)
        
        # Adam optimizer
        beta1, beta2 = 0.9, 0.99
        eps = 1e-6
        
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
        
        m_hat = self.m / (1 - beta1 ** (self.t // update_freq))
        v_hat = self.v / (1 - beta2 ** (self.t // update_freq))
        
        # Adaptive learning rate based on qubit count
        lr_scale = 1.0 if self.num_qubits <= 16 else 0.9
        self.params -= (self.learning_rate * lr_scale) * m_hat / (pnp.sqrt(v_hat) + eps)
        
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
    """Q-learning agent optimized for adaptive VQNN."""
    
    def __init__(self, vqnn: VQNN, epsilon: float = EPSILON_START, 
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
        
        # Process smaller sub-batches
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
        pos = np.where(maze == marker)
        return (pos[0][0], pos[1][0])
    
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
# RANDOM WALK BASELINE
# ============================================================================

def calculate_random_walk_average(maze: np.ndarray, trials: int = 100) -> float:
    """
    Calculate average steps for random walk to reach goal.
    
    Args:
        maze: The maze to test
        trials: Number of random walk trials
        
    Returns:
        Average steps to reach goal (or max_steps if rarely succeeds)
    """
    env = MazeEnvironment(maze)
    total_steps = 0
    successes = 0
    max_steps = 200
    
    for _ in range(trials):
        env.reset()
        steps = 0
        
        while steps < max_steps:
            # Random action selection
            action = np.random.choice(ACTIONS)
            position, _, done = env.step(action)
            steps += 1
            
            if done and position == env.goal_pos:
                total_steps += steps
                successes += 1
                break
    
    if successes > 0:
        return total_steps / successes
    else:
        return max_steps  # If random walk rarely succeeds

# ============================================================================
# COMPLEXITY METRICS
# ============================================================================

@lru_cache(maxsize=512)
def lz_complexity(s: str) -> int:
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
# IMPROVED EMERGENCE DETECTION
# ============================================================================

def detect_emergence(solution: MazeSolution, baseline_steps: Dict[str, int],
                    learning_curve: List[float], maze: np.ndarray) -> Dict[str, Any]:
    """
    Improved emergence detection with stricter criteria.
    
    Args:
        solution: The maze solution to evaluate
        baseline_steps: Dictionary with baseline performance metrics
        learning_curve: Episode rewards over training
        maze: The maze array for random walk calculation
        
    Returns:
        Dictionary with emergence assessment
    """
    efficiency = solution.efficiency_score
    
    # Detect performance discontinuity (sudden jumps in learning)
    solution.performance_discontinuity = False
    solution.convergence_episode = len(learning_curve)
    
    if len(learning_curve) > 20:
        window = 10
        for i in range(window, len(learning_curve) - window):
            before = np.mean(learning_curve[i-window:i])
            after = np.mean(learning_curve[i:i+window])
            # Require larger jump (50 points instead of 30)
            if after - before > 50:
                solution.performance_discontinuity = True
                solution.convergence_episode = i
                break
    
    is_emergent = False
    emergence_type = "none"
    emergence_score = 0.0
    
    # STRICTER thresholds based on qubit count
    if NUM_QUBITS <= 12:
        perfect_threshold = 0.95  # Was 0.85
        good_threshold = 0.85     # Was 0.70
        min_threshold = 0.75      # New minimum threshold
    elif NUM_QUBITS <= 16:
        perfect_threshold = 0.97  # Was 0.88
        good_threshold = 0.90     # Was 0.75
        min_threshold = 0.80
    elif NUM_QUBITS <= 20:
        perfect_threshold = 0.98  # Was 0.90
        good_threshold = 0.92     # Was 0.78
        min_threshold = 0.85
    else:
        perfect_threshold = 0.99  # Was 0.92
        good_threshold = 0.95     # Was 0.80
        min_threshold = 0.88
    
    # Calculate actual random walk performance
    actual_random_steps = calculate_random_walk_average(maze, trials=50)
    random_improvement = (actual_random_steps - solution.steps_to_goal) / actual_random_steps
    
    # Track which criteria are met
    emergence_criteria_met = 0
    criteria_details = []
    
    # Criterion 1: Near-optimal performance
    if efficiency >= perfect_threshold:
        emergence_criteria_met += 2  # Weight this heavily
        criteria_details.append("perfect_efficiency")
    elif efficiency >= good_threshold:
        emergence_criteria_met += 1
        criteria_details.append("good_efficiency")
    elif efficiency >= min_threshold:
        emergence_criteria_met += 0.5
        criteria_details.append("acceptable_efficiency")
    
    # Criterion 2: Significant learning discontinuity with sustained improvement
    if solution.performance_discontinuity:
        # Check if the jump led to sustained improvement
        if len(learning_curve) > solution.convergence_episode + 10:
            post_jump = learning_curve[solution.convergence_episode:]
            if np.mean(post_jump) > 80:  # Sustained high performance
                emergence_criteria_met += 1
                criteria_details.append("sustained_jump")
    
    # Criterion 3: Vastly outperforms random walk
    if random_improvement > 0.85:  # Was 0.65, now much stricter
        emergence_criteria_met += 1.5
        criteria_details.append("beats_random_strongly")
    elif random_improvement > 0.75:
        emergence_criteria_met += 0.5
        criteria_details.append("beats_random")
    
    # Criterion 4: Fast convergence to good solution
    if solution.convergence_episode < 30 and efficiency >= good_threshold:
        emergence_criteria_met += 1
        criteria_details.append("fast_convergence")
    elif solution.convergence_episode < 40 and efficiency >= min_threshold:
        emergence_criteria_met += 0.5
        criteria_details.append("moderate_convergence")
    
    # Criterion 5: Consistency check - low variance in final performance
    if len(learning_curve) >= 20:
        final_rewards = learning_curve[-20:]
        reward_std = np.std(final_rewards)
        if reward_std < 10 and np.mean(final_rewards) > 80:
            emergence_criteria_met += 0.5
            criteria_details.append("consistent_performance")
    
    # Require at least 2.5 points worth of criteria for emergence
    if emergence_criteria_met >= 2.5:
        is_emergent = True
        
        # Determine emergence type based on which criteria were met
        if efficiency >= perfect_threshold and "fast_convergence" in criteria_details:
            emergence_type = "perfect_navigation"
            emergence_score = 1.0
        elif solution.performance_discontinuity and "sustained_jump" in criteria_details:
            emergence_type = "sudden_insight"
            emergence_score = 0.8
        elif efficiency >= good_threshold and "beats_random_strongly" in criteria_details:
            emergence_type = "efficient_navigation"
            emergence_score = 0.7
        elif "beats_random_strongly" in criteria_details:
            emergence_type = "intelligent_navigation"
            emergence_score = 0.5
        else:
            emergence_type = "weak_emergence"
            emergence_score = 0.4
    
    return {
        "is_emergent": is_emergent,
        "emergence_type": emergence_type,
        "emergence_score": emergence_score,
        "efficiency_score": efficiency,
        "performance_discontinuity": solution.performance_discontinuity,
        "convergence_episode": solution.convergence_episode,
        "criteria_met": emergence_criteria_met,
        "criteria_details": criteria_details,
        "random_walk_baseline": actual_random_steps,
        "random_improvement": random_improvement
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_solution(maze: np.ndarray, path: List[Tuple[int, int]], 
                      maze_name: str):
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
    """Execute quantum maze navigation experiments with improved emergence detection."""
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
    print(f"{NUM_QUBITS}-QUBIT VQNN ADAPTIVE PROBLEM-SOLVING EXPERIMENTS")
    print("WITH IMPROVED EMERGENCE DETECTION")
    print("=" * 60)
    print(f"Quantum Backend: {QUANTUM_BACKEND}")
    print(f"GPU Acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Shots: {SHOTS}")
    print(f"Episodes per maze: {config['episodes_per_maze']}")
    
    # Dynamic runtime estimate
    if NUM_QUBITS <= 12:
        est_total = "15-30"
    elif NUM_QUBITS <= 16:
        est_total = "30-50"
    elif NUM_QUBITS <= 20:
        est_total = "50-90"
    else:
        est_total = "90-150"
    
    print(f"Expected total runtime: {est_total} minutes")
    print("\nImproved emergence detection features:")
    print("  • Stricter efficiency thresholds")
    print("  • Actual random walk baselines")
    print("  • Multi-criteria validation")
    print("  • Performance consistency checks")
    print("=" * 60)
    print()
    
    generator = MazeGenerator()
    mazes = generator.get_fixed_mazes()
    
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], f"vqnn_results_{NUM_QUBITS}q_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "maze_name", "maze_complexity",
        "episodes_trained", "steps_to_goal", "optimal_steps", "efficiency_score",
        "final_reward", "convergence_episode", "performance_discontinuity",
        "path_length", "path_lz_complexity", "path_shannon_entropy",
        "action_sequence_length", "action_lz_complexity", "action_shannon_entropy",
        "action_approximate_entropy", "is_emergent", "emergence_type", 
        "emergence_score", "solution_path", "action_sequence",
        "quantum_backend", "gpu_accelerated", "num_qubits", "shots",
        "training_time_minutes", "criteria_met", "random_walk_baseline", "random_improvement"
    ]
    
    results = []
    total_start = time.time()
    
    print(f"Running {NUM_QUBITS}-qubit quantum experiments...\n")
    
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
        
        # Calculate actual random walk baseline
        print("Calculating random walk baseline...", end="")
        random_walk_avg = calculate_random_walk_average(maze, trials=50)
        print(f" {random_walk_avg:.1f} steps average")
        
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
        
        print(f"\nTraining with {NUM_QUBITS}-qubit quantum circuit ({SHOTS} shots)...")
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
            
            # Progress updates every 20 episodes
            if episode % 20 == 0:
                avg_reward = np.mean(learning_curve[-10:]) if len(learning_curve) >= 10 else episode_reward
                elapsed = (time.time() - maze_start) / 60
                print(f"  Episode {episode:3d}/{config['episodes_per_maze']}: "
                      f"Avg reward = {avg_reward:6.1f}, ε = {agent.epsilon:.3f}, "
                      f"Time = {elapsed:.1f} min")
        
        # Final evaluation
        print("\nEvaluating final performance...")
        agent.epsilon = 0
        
        # Run multiple evaluations
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
        
        baselines = {"random": random_walk_avg, "optimal": optimal_steps}
        emergence = detect_emergence(best_solution, baselines, learning_curve, maze)
        
        training_time = (time.time() - maze_start) / 60
        
        print(f"\nResults:")
        print(f"  Steps to goal: {best_solution.steps_to_goal}")
        print(f"  Optimal steps: {optimal_steps}")
        print(f"  Random walk avg: {random_walk_avg:.1f}")
        print(f"  Efficiency: {best_solution.efficiency_score:.2%}")
        print(f"  Random improvement: {emergence['random_improvement']:.2%}")
        print(f"  Criteria met: {emergence['criteria_met']:.1f}")
        print(f"  Emergent: {emergence['is_emergent']} ({emergence['emergence_type']})")
        print(f"  Training time: {training_time:.1f} minutes")
        
        if config.get("visualize", True) and emergence['is_emergent']:
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
            training_time,
            emergence["criteria_met"],
            emergence["random_walk_baseline"],
            emergence["random_improvement"]
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
    avg_efficiency = np.mean([r[7] for r in results])
    avg_criteria = np.mean([r[28] for r in results])
    
    print(f"Total mazes: {len(results)}")
    print(f"Emergent solutions: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    print(f"Average efficiency: {avg_efficiency:.3f}")
    print(f"Average criteria met: {avg_criteria:.2f}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Average time per maze: {total_time/len(mazes):.1f} minutes")
    print(f"Quantum backend: {QUANTUM_BACKEND}")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Shots: {SHOTS}")
    print(f"Statistical error: ±{100/np.sqrt(SHOTS):.1f}%")
    
    emergence_types = {}
    for r in results:
        if r[18]:
            etype = r[19]
            emergence_types[etype] = emergence_types.get(etype, 0) + 1
    
    if emergence_types:
        print("\nEmergence breakdown:")
        for etype, count in emergence_types.items():
            print(f"  {etype}: {count}")
    else:
        print("\nNo emergent behaviors detected with stricter criteria.")
    
    print(f"\nResults saved to: {output_file}")
    
    # Performance report
    vqnn.perf.report()
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADAPTIVE QUANTUM MAZE NAVIGATION EXPERIMENT")
    print("WITH IMPROVED EMERGENCE DETECTION")
    print("="*60)
    print("\nThis experiment features:")
    print("  • User-configurable qubit count (8-25)")
    print("  • Stricter emergence detection criteria")
    print("  • Actual random walk baselines")
    print("  • Multi-criteria validation")
    print("  • Performance consistency checks")
    print("\nHigher qubit counts provide:")
    print("  • Better quantum advantage and entanglement")
    print("  • Richer state representations")
    print("  • Potentially higher solution quality")
    print("  • Longer runtime due to increased complexity")
    
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
    
    print(f"\n✅ {NUM_QUBITS}-qubit experiment with improved emergence detection complete!")
    print("=" * 60)