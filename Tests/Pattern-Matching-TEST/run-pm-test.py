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
__version__ = "1.0-beta"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title         : {Scalable Quantum Hopfield Neural Network Pattern Matching}
Date          : {05-18-2025}
Description   : {This code implements a GPU-accelerated Quantum Hopfield Neural Network with 
                proper problem-size scaling. Tests 9-qubit (3x3), 16-qubit (4x4), and 
                25-qubit (5x5) configurations for scientifically valid emergence detection.}
Options       : {User-selectable pattern size with matched qubit allocation}
Dependencies  : {pip install numpy scipy pandas matplotlib seaborn statsmodels pennylane>=0.30.0 pennylane-lightning[gpu] torch torchvision torchaudio cupy-cuda118 scikit-learn plotly}
Requirements  : {Python 3.8+}
Usage         : {python run-pm-test.py}
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
import sys
import time
warnings.filterwarnings('ignore')

# ============================================================================
# USER INPUT FOR PATTERN SIZE CONFIGURATION
# ============================================================================

def get_pattern_configuration():
    """Get the pattern size from user input with validation."""
    print("\n" + "=" * 60)
    print("QUANTUM HOPFIELD NETWORK CONFIGURATION")
    print("=" * 60)
    
    while True:
        try:
            print("\nSelect pattern size for the Quantum Hopfield Network:")
            print("\nAvailable configurations:")
            print("  1. 3x3 patterns (9 qubits)")
            print("     - Simple patterns: I, L, T")
            print("     - Fastest execution (~2-5 minutes)")
            print("     - Good for initial testing")
            print()
            print("  2. 4x4 patterns (16 qubits)")
            print("     - Medium patterns: T, O, H")
            print("     - Moderate runtime (~5-10 minutes)")
            print("     - Balanced complexity")
            print()
            print("  3. 5x5 patterns (25 qubits)")
            print("     - Complex patterns: T, O, X")
            print("     - Longer runtime (~10-20 minutes)")
            print("     - Maximum pattern complexity")
            print()
            print("Each configuration uses N² qubits for NxN patterns")
            print("ensuring proper problem-solution scaling")
            
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            
            if choice == '1':
                pattern_size = 3
                num_qubits = 9
                pattern_names = "I, L, T"
                est_time = "2-5"
            elif choice == '2':
                pattern_size = 4
                num_qubits = 16
                pattern_names = "T, O, H"
                est_time = "5-10"
            elif choice == '3':
                pattern_size = 5
                num_qubits = 25
                pattern_names = "T, O, X"
                est_time = "10-20"
            else:
                print("⚠ Please enter 1, 2, or 3")
                continue
            
            # Confirm with user
            print(f"\nYou selected: {pattern_size}x{pattern_size} patterns")
            print(f"Configuration details:")
            print(f"  - Pattern size: {pattern_size}x{pattern_size}")
            print(f"  - Qubits required: {num_qubits}")
            print(f"  - Pattern types: {pattern_names}")
            print(f"  - Estimated runtime: {est_time} minutes")
            
            confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
            if confirm == 'y' or confirm == 'yes':
                return pattern_size, num_qubits
            elif confirm == 'n' or confirm == 'no':
                continue
            else:
                print("Please enter 'y' for yes or 'n' for no.")
                
        except KeyboardInterrupt:
            print("\n\nExiting configuration...")
            sys.exit(0)
        except Exception as e:
            print(f"⚠ An error occurred: {e}")
            continue

# Get user configuration
PATTERN_SIZE, NUM_QUBITS = get_pattern_configuration()
NUM_NEURONS = NUM_QUBITS  # Direct 1-to-1 mapping

# ============================================================================
# CONFIGURATION CONSTANTS - ADJUSTED FOR PATTERN SIZE
# ============================================================================

# Seed management for reproducibility
GLOBAL_SEED = None  # Set to integer for reproducibility, None for true randomness

# Quantum noise parameter - adjust based on pattern complexity
if PATTERN_SIZE == 3:
    QUANTUM_NOISE_LEVEL = 0.06
    DEFAULT_SHOTS = 250
    TEMPERATURE = 0.15
elif PATTERN_SIZE == 4:
    QUANTUM_NOISE_LEVEL = 0.05
    DEFAULT_SHOTS = 200
    TEMPERATURE = 0.12
else:  # PATTERN_SIZE == 5
    QUANTUM_NOISE_LEVEL = 0.04
    DEFAULT_SHOTS = 150
    TEMPERATURE = 0.10

# Quantum backend handling
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Using simulation fallback.")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(slots=True)
class RetrievalMetrics:
    """Container for retrieval metrics."""
    retrieved_pattern: np.ndarray
    prediction_accuracy: float
    hamming_distance: int
    confidence: float
    convergence_quality: float
    raw_output_bitstring: str
    cue_bitstring: str
    energy: float
    spurious_state: bool

# ============================================================================
# SCALABLE PATTERN GENERATION MODULE
# ============================================================================

class ScalablePatternGenerator:
    """Pattern generation for different grid sizes."""
    
    def __init__(self, pattern_size: int, seed: Optional[int] = None):
        """Initialize with pattern size and optional seed."""
        self.pattern_size = pattern_size
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
    
    def get_fixed_patterns(self) -> Tuple[List[np.ndarray], List[str]]:
        """Generate fixed patterns based on grid size."""
        if self.pattern_size == 3:
            return self._get_3x3_patterns()
        elif self.pattern_size == 4:
            return self._get_4x4_patterns()
        else:  # pattern_size == 5
            return self._get_5x5_patterns()
    
    def _get_3x3_patterns(self) -> Tuple[List[np.ndarray], List[str]]:
        """Generate 3x3 patterns: I, L, T"""
        patterns = {}
        
        # I pattern: vertical line
        I = -np.ones((3, 3), dtype=np.int8)
        I[:, 1] = 1
        patterns["I"] = I
        
        # L pattern: L shape
        L = -np.ones((3, 3), dtype=np.int8)
        L[:, 0] = 1
        L[2, :] = 1
        patterns["L"] = L
        
        # T pattern: T shape
        T = -np.ones((3, 3), dtype=np.int8)
        T[0, :] = 1
        T[:, 1] = 1
        patterns["T"] = T
        
        return list(patterns.values()), list(patterns.keys())
    
    def _get_4x4_patterns(self) -> Tuple[List[np.ndarray], List[str]]:
        """Generate 4x4 patterns: T, O, H"""
        patterns = {}
        
        # T pattern: top row + center column
        T = -np.ones((4, 4), dtype=np.int8)
        T[0, :] = 1
        T[:, 1:3] = 1  # Use two center columns for 4x4
        patterns["T"] = T
        
        # O pattern: border only
        O = -np.ones((4, 4), dtype=np.int8)
        O[0, :] = O[-1, :] = O[:, 0] = O[:, -1] = 1
        patterns["O"] = O
        
        # H pattern: two verticals connected
        H = -np.ones((4, 4), dtype=np.int8)
        H[:, 0] = H[:, 3] = 1
        H[1:3, :] = 1
        patterns["H"] = H
        
        return list(patterns.values()), list(patterns.keys())
    
    def _get_5x5_patterns(self) -> Tuple[List[np.ndarray], List[str]]:
        """Generate 5x5 patterns: T, O, X"""
        patterns = {}
        
        # T pattern: top row + center column
        T = -np.ones((5, 5), dtype=np.int8)
        T[0, :] = 1
        T[:, 2] = 1
        patterns["T"] = T
        
        # O pattern: border only
        O = -np.ones((5, 5), dtype=np.int8)
        O[0, :] = O[-1, :] = O[:, 0] = O[:, -1] = 1
        patterns["O"] = O
        
        # X pattern: diagonals
        X = -np.ones((5, 5), dtype=np.int8)
        for i in range(5):
            X[i, i] = X[i, 4-i] = 1
        patterns["X"] = X
        
        return list(patterns.values()), list(patterns.keys())
    
    def apply_noise(self, pattern: np.ndarray, noise_level: float, 
                   ensure_randomness: bool = True) -> np.ndarray:
        """Apply random bit-flip noise to pattern."""
        if ensure_randomness and self.seed is None:
            local_rng = np.random.RandomState()
        else:
            local_rng = self.rng
            
        flat = pattern.flatten().copy()
        n_flip = int(noise_level * len(flat))
        
        if n_flip > 0:
            flip_indices = local_rng.choice(len(flat), n_flip, replace=False)
            flat[flip_indices] *= -1
            
        return flat.reshape(pattern.shape)
    
    def apply_mask(self, pattern: np.ndarray, mask_level: float,
                  ensure_randomness: bool = True) -> np.ndarray:
        """Apply masking to pattern by setting bits to unknown (0)."""
        if ensure_randomness and self.seed is None:
            local_rng = np.random.RandomState()
        else:
            local_rng = self.rng
            
        flat = pattern.flatten().copy()
        n_mask = int(mask_level * len(flat))
        
        if n_mask > 0:
            mask_indices = local_rng.choice(len(flat), n_mask, replace=False)
            flat[mask_indices] = 0
            
        return flat.reshape(pattern.shape)

# ============================================================================
# SCALABLE QUANTUM HOPFIELD NETWORK
# ============================================================================

class ScalableQuantumHopfieldNetwork:
    """
    Quantum Hopfield Network with proper size scaling.
    
    Implements:
    - Hebbian learning for pattern storage
    - Size-appropriate quantum circuits
    - Stochastic field inference for masked bits
    - Quantum/classical evolution dynamics
    - Energy landscape analysis
    """
    
    def __init__(self, pattern_size: int, temperature: float = None):
        """
        Initialize network for specific pattern size.
        
        Args:
            pattern_size: Size of patterns (3, 4, or 5)
            temperature: Temperature parameter (uses default if None)
        """
        self.pattern_size = pattern_size
        self.num_neurons = pattern_size * pattern_size
        self.num_qubits = self.num_neurons  # Direct mapping
        self.temperature = temperature if temperature is not None else TEMPERATURE
        self.weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        self.stored_patterns = []
        self.pattern_names = []
        self.rng = np.random.RandomState()
        
        print(f"\nInitializing {pattern_size}x{pattern_size} Quantum Hopfield Network...")
        print(f"  Configuration:")
        print(f"    - Pattern size: {pattern_size}x{pattern_size}")
        print(f"    - Neurons/Qubits: {self.num_neurons}")
        print(f"    - Temperature: {self.temperature:.3f}")
        print(f"    - Quantum noise: {QUANTUM_NOISE_LEVEL:.3f}")
        print(f"    - Shots per trial: {DEFAULT_SHOTS}")
    
    def store_patterns(self, patterns: List[np.ndarray], names: List[str]):
        """Store patterns using Hebbian learning rule."""
        self.stored_patterns = [p.astype(np.int8) for p in patterns]
        self.pattern_names = names
        
        if not patterns:
            return
            
        # Standard Hebbian learning
        pattern_matrix = np.array([p.flatten() for p in patterns], dtype=np.float32)
        self.weights = np.dot(pattern_matrix.T, pattern_matrix) / len(patterns)
        np.fill_diagonal(self.weights, 0)
    
    def stochastic_field_inference(self, pattern_1d: np.ndarray) -> np.ndarray:
        """Stochastic inference for masked bits using local fields."""
        s = pattern_1d.astype(np.float32).copy()
        mask = (s == 0)
        
        if not mask.any():
            return s.astype(np.int8)
        
        # Calculate local fields from known bits
        s_known = s.copy()
        s_known[mask] = 0
        h = self.weights @ s_known
        
        # Stochastic assignment for masked positions
        for i in np.where(mask)[0]:
            if self.temperature > 0:
                # Boltzmann probability for spin assignment
                prob_plus = 1 / (1 + np.exp(-2 * h[i] / self.temperature))
                s[i] = 1 if self.rng.random() < prob_plus else -1
            else:
                # Zero temperature: deterministic sign
                s[i] = np.sign(h[i]) if h[i] != 0 else (1 if self.rng.random() > 0.5 else -1)
        
        return s.astype(np.int8)
    
    def calculate_energy(self, state: np.ndarray) -> float:
        """Calculate Hopfield energy for a state."""
        state_flat = state.flatten()
        return -0.5 * np.dot(state_flat, np.dot(self.weights, state_flat))
    
    def is_spurious_state(self, state: np.ndarray, threshold: float = 0.2) -> bool:
        """Check if state is spurious (not close to any stored pattern)."""
        state_flat = state.flatten()
        
        for pattern in self.stored_patterns:
            pattern_flat = pattern.flatten()
            overlap = np.dot(state_flat, pattern_flat) / len(state_flat)
            if abs(overlap) > (1 - threshold):
                return False
        
        return True
    
    def retrieve(self, cue_2d: np.ndarray, original_2d: np.ndarray,
                shots: int = None, add_quantum_noise: bool = True) -> Dict[str, Any]:
        """Retrieve pattern from degraded cue using stochastic dynamics."""
        if shots is None:
            shots = DEFAULT_SHOTS
            
        shape = cue_2d.shape
        cue_flat = cue_2d.flatten()
        original_flat = original_2d.flatten()
        
        # Run multiple independent retrievals for statistics
        outcomes = []
        energies = []
        
        # Adjust sample size based on pattern size
        if self.pattern_size == 3:
            max_samples = min(shots, 60)
        elif self.pattern_size == 4:
            max_samples = min(shots, 40)
        else:
            max_samples = min(shots, 30)
        
        for _ in range(max_samples):
            # Stochastic inference for masked bits
            cue_filled = self.stochastic_field_inference(cue_flat)
            
            if PENNYLANE_AVAILABLE and shots > 10:
                retrieved = self._quantum_evolution(cue_filled, add_quantum_noise)
            else:
                retrieved = self._classical_evolution(cue_filled)
            
            outcomes.append(tuple(retrieved))
            energies.append(self.calculate_energy(retrieved))
        
        # Statistical analysis of outcomes
        unique_outcomes, counts = np.unique(outcomes, axis=0, return_counts=True)
        best_idx = np.argmax(counts)
        best_outcome = unique_outcomes[best_idx]
        confidence = counts[best_idx] / len(outcomes)
        
        # Check for spurious state
        spurious = self.is_spurious_state(best_outcome.reshape(shape))
        
        # Calculate metrics
        accuracy = np.mean(best_outcome == original_flat)
        hamming = np.sum(best_outcome != original_flat)
        convergence = self._convergence_quality(best_outcome)
        avg_energy = np.mean(energies)
        
        # Create bitstring representations
        output_bits = ''.join('0' if x == 1 else '1' for x in best_outcome)
        cue_bits = ''.join('0' if x == 1 else '1' if x == -1 else '2' for x in cue_flat)
        
        return {
            "retrieved_pattern": best_outcome.reshape(shape),
            "prediction_accuracy": float(accuracy),
            "hamming_distance": int(hamming),
            "confidence": float(confidence),
            "convergence_quality": float(convergence),
            "raw_output_bitstring": output_bits,
            "cue_bitstring": cue_bits,
            "energy": float(avg_energy),
            "spurious_state": spurious
        }
    
    def _quantum_evolution(self, initial_state: np.ndarray, add_noise: bool) -> np.ndarray:
        """
        Quantum circuit evolution with size-appropriate architecture.
        """
        # GPU device selection with fallbacks
        try:
            dev = qml.device("lightning.gpu", wires=self.num_qubits, shots=1)
            device_name = f"GPU (lightning.gpu, {self.num_qubits} qubits)"
        except Exception:
            try:
                dev = qml.device("lightning.qubit", wires=self.num_qubits, shots=1)
                device_name = f"CPU (lightning.qubit, {self.num_qubits} qubits)"
            except Exception:
                dev = qml.device("default.qubit", wires=self.num_qubits, shots=1)
                device_name = f"CPU (default.qubit, {self.num_qubits} qubits)"
        
        # Print device selection once
        if not hasattr(self, '_device_logged'):
            print(f"🔧 Quantum device: {device_name}")
            self._device_logged = True
        
        @qml.qnode(dev)
        def circuit():
            # State preparation
            for i in range(self.num_neurons):
                if initial_state[i] == -1:
                    qml.PauliX(wires=i)
            
            # Add quantum noise for uncertainty
            if add_noise:
                for i in range(self.num_neurons):
                    qml.RY(QUANTUM_NOISE_LEVEL * self.rng.randn(), wires=i)
            
            # Hopfield evolution with Ising interactions
            # Adjust interaction density based on pattern size
            if self.pattern_size == 3:
                # For 3x3: Full connectivity feasible
                for i in range(self.num_neurons):
                    for j in range(i + 1, self.num_neurons):
                        if abs(self.weights[i, j]) > 1e-6:
                            coupling = self.weights[i, j]
                            if add_noise:
                                coupling += QUANTUM_NOISE_LEVEL * self.rng.randn() * 0.5
                            qml.IsingZZ(2.0 * coupling, wires=[i, j])
            
            elif self.pattern_size == 4:
                # For 4x4: Selective connectivity
                for i in range(self.num_neurons):
                    # Connect to neighbors and important pairs
                    for j in range(i + 1, min(i + 5, self.num_neurons)):
                        if abs(self.weights[i, j]) > 1e-5:
                            coupling = self.weights[i, j]
                            if add_noise:
                                coupling += QUANTUM_NOISE_LEVEL * self.rng.randn() * 0.3
                            qml.IsingZZ(1.5 * coupling, wires=[i, j])
            
            else:  # pattern_size == 5
                # For 5x5: Sparse connectivity for efficiency
                for i in range(self.num_neurons):
                    # Local neighborhood connections
                    neighbors = []
                    row, col = i // 5, i % 5
                    
                    # Adjacent cells in grid
                    for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 5 and 0 <= new_col < 5:
                            j = new_row * 5 + new_col
                            if j > i:
                                neighbors.append(j)
                    
                    for j in neighbors:
                        if abs(self.weights[i, j]) > 1e-5:
                            coupling = self.weights[i, j]
                            if add_noise:
                                coupling += QUANTUM_NOISE_LEVEL * self.rng.randn() * 0.2
                            qml.IsingZZ(coupling, wires=[i, j])
            
            # Measure all qubits
            return qml.sample(wires=range(self.num_neurons))
        
        # Execute circuit
        try:
            measurement = circuit().flatten()
            return np.where(measurement == 0, 1, -1).astype(np.int8)
        except Exception as e:
            print(f"⚠️ Quantum evolution failed: {e}")
            return self._classical_evolution(initial_state)
    
    def _classical_evolution(self, initial_state: np.ndarray) -> np.ndarray:
        """Classical stochastic Hopfield dynamics."""
        state = initial_state.copy()
        
        # Adjust update steps based on pattern size
        num_steps = 8 if self.pattern_size == 3 else 10 if self.pattern_size == 4 else 12
        
        for _ in range(num_steps):
            # Random asynchronous updates
            for i in self.rng.permutation(self.num_neurons):
                h = np.dot(self.weights[i], state)
                
                if self.temperature > 0:
                    # Stochastic update
                    prob_plus = 1 / (1 + np.exp(-2 * h / self.temperature))
                    state[i] = 1 if self.rng.random() < prob_plus else -1
                else:
                    # Deterministic update
                    state[i] = np.sign(h) if h != 0 else state[i]
        
        return state.astype(np.int8)
    
    def _convergence_quality(self, retrieved: np.ndarray) -> float:
        """Measure convergence quality to stored patterns."""
        if not self.stored_patterns:
            return 0.0
        
        max_overlap = 0.0
        for pattern in self.stored_patterns:
            pattern_flat = pattern.flatten()
            overlap = abs(np.dot(retrieved, pattern_flat)) / len(retrieved)
            max_overlap = max(max_overlap, overlap)
        
        return max_overlap

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

# ============================================================================
# SIZE-AWARE EMERGENCE DETECTION
# ============================================================================

def detect_emergence(degradation_level: float, degradation_type: str,
                    metrics: Dict, patterns: List[np.ndarray],
                    pattern_names: List[str], pattern_size: int) -> Dict[str, Any]:
    """
    Detect emergent behavior with size-appropriate criteria.
    
    Smaller patterns have more lenient criteria since they have
    less information content and fewer degrees of freedom.
    """
    retrieved = metrics["retrieved_pattern"].flatten()
    confidence = metrics["confidence"]
    spurious = metrics.get("spurious_state", False)
    
    # Calculate distances to all stored patterns
    distances = []
    for pattern, name in zip(patterns, pattern_names):
        dist = np.sum(retrieved != pattern.flatten())
        distances.append((dist, name))
    
    min_dist, closest_name = min(distances)
    
    # Size-adjusted thresholds
    if pattern_size == 3:
        # 3x3: More lenient (9 bits total)
        confidence_threshold_noise = 0.7
        confidence_threshold_mask = 0.75
        distance_threshold = 2
        min_degradation_noise = 0.10
        min_degradation_mask = 0.15
    elif pattern_size == 4:
        # 4x4: Moderate (16 bits total)
        confidence_threshold_noise = 0.75
        confidence_threshold_mask = 0.80
        distance_threshold = 3
        min_degradation_noise = 0.12
        min_degradation_mask = 0.18
    else:  # pattern_size == 5
        # 5x5: Stricter (25 bits total)
        confidence_threshold_noise = 0.80
        confidence_threshold_mask = 0.85
        distance_threshold = 4
        min_degradation_noise = 0.15
        min_degradation_mask = 0.20
    
    # Emergence criteria evaluation
    is_emergent = False
    emergence_type = "none"
    emergence_score = 0.0
    
    # Different criteria for different degradation types
    if degradation_type == "noise":
        if degradation_level >= min_degradation_noise:
            if min_dist <= distance_threshold and confidence > confidence_threshold_noise and not spurious:
                is_emergent = True
                if min_dist == 0:
                    emergence_type = "perfect_recall"
                    emergence_score = 1.0
                elif min_dist <= 1:
                    emergence_type = "near_perfect_recall"
                    emergence_score = 0.9
                else:
                    emergence_type = "robust_recall"
                    emergence_score = max(0, 0.8 - (min_dist - 1) * 0.1)
    
    elif degradation_type == "incompleteness":
        if degradation_level >= min_degradation_mask:
            if min_dist <= distance_threshold and confidence > confidence_threshold_mask and not spurious:
                is_emergent = True
                if min_dist == 0:
                    emergence_type = "perfect_reconstruction"
                    emergence_score = 0.95
                elif min_dist <= 1:
                    emergence_type = "near_perfect_reconstruction"
                    emergence_score = 0.85
                else:
                    emergence_type = "robust_reconstruction"
                    emergence_score = max(0, 0.7 - (min_dist - 1) * 0.15)
    
    # Error correction capability (normalized by pattern size)
    expected_errors = degradation_level * (pattern_size * pattern_size)
    actual_errors = min_dist
    error_correction = max(0, 1 - actual_errors / max(expected_errors, 1))
    
    # Special case: Exceptional emergence for high degradation
    if degradation_level >= 0.25 and min_dist == 0 and confidence > 0.9:
        emergence_type = "exceptional_recall"
        emergence_score = 1.0
        is_emergent = True
    
    return {
        "is_emergent": is_emergent,
        "emergence_type": emergence_type,
        "emergence_score": emergence_score,
        "closest_pattern": closest_name,
        "closest_distance": int(min_dist),
        "error_correction_capability": error_correction
    }

# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================

def validate_results(results: List[List], pattern_size: int) -> Dict[str, Any]:
    """Perform statistical validation of experimental results."""
    noise_results = [r for r in results if r[3] == "noise"]
    mask_results = [r for r in results if r[3] == "incompleteness"]
    
    noise_emergence = sum(1 for r in noise_results if r[17])
    mask_emergence = sum(1 for r in mask_results if r[17])
    
    validation = {
        "pattern_size": pattern_size,
        "noise_emergence_rate": noise_emergence / len(noise_results) if noise_results else 0,
        "mask_emergence_rate": mask_emergence / len(mask_results) if mask_results else 0,
        "anomaly_detected": False,
        "warnings": []
    }
    
    # Check for statistical anomalies
    if validation["mask_emergence_rate"] > 0.5 and validation["noise_emergence_rate"] < 0.1:
        validation["anomaly_detected"] = True
        validation["warnings"].append(
            "Asymmetry detected: masking shows significantly higher emergence than noise"
        )
    
    # Size-specific warnings
    if pattern_size == 3 and validation["noise_emergence_rate"] > 0.9:
        validation["warnings"].append(
            "Very high emergence rate for 3x3 patterns may indicate overly lenient criteria"
        )
    
    if pattern_size == 5 and validation["mask_emergence_rate"] < 0.1:
        validation["warnings"].append(
            "Very low emergence rate for 5x5 masking may indicate overly strict criteria"
        )
    
    return validation

# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_experiments(config: Optional[Dict] = None):
    """Execute pattern matching experiments with proper size scaling."""
    if config is None:
        config = {
            "noise_levels": [0.1, 0.2, 0.3],
            "masking_levels": [0.1, 0.2, 0.3],
            "repetitions": DEFAULT_SHOTS,
            "temperature": TEMPERATURE,
            "output_dir": "results",
            "validate": True
        }
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    print("\n" + "=" * 60)
    print(f"{PATTERN_SIZE}x{PATTERN_SIZE} QUANTUM HOPFIELD NETWORK EXPERIMENTS")
    print("=" * 60)
    print(f"Pattern size: {PATTERN_SIZE}x{PATTERN_SIZE}")
    print(f"Qubits: {NUM_QUBITS}")
    print(f"Temperature: {config.get('temperature', TEMPERATURE):.3f}")
    print(f"Quantum noise: {QUANTUM_NOISE_LEVEL:.3f}")
    print(f"Shots per retrieval: {config['repetitions']}")
    print(f"Seed: {GLOBAL_SEED or 'True random'}")
    print("=" * 60)
    print()
    
    # Initialize components
    generator = ScalablePatternGenerator(pattern_size=PATTERN_SIZE, seed=GLOBAL_SEED)
    patterns, names = generator.get_fixed_patterns()
    
    print(f"Patterns to test: {', '.join(names)}")
    
    qhnn = ScalableQuantumHopfieldNetwork(
        pattern_size=PATTERN_SIZE,
        temperature=config.get("temperature", TEMPERATURE)
    )
    qhnn.store_patterns([p.flatten() for p in patterns], names)
    
    # Output setup
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], 
                              f"qhnn_results_{NUM_QUBITS}q_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "pattern_name", "degradation_type", "degradation_level",
        "pattern_size", "num_patterns_stored", "prediction_accuracy", "hamming_distance",
        "confidence", "convergence_quality", "raw_output_bitstring", "cue_bitstring",
        "cue_lz_complexity", "cue_shannon_entropy", "output_lz_complexity", "output_shannon_entropy",
        "is_emergent", "emergence_type", "emergence_score", "closest_pattern",
        "closest_distance", "error_correction_capability", "energy", "spurious_state",
        "grid_size", "num_qubits"
    ]
    
    results = []
    run_id = 0
    start_time = time.time()
    
    print("\nRunning experiments...\n")
    
    # Test each pattern
    for pattern, name in zip(patterns, names):
        
        # Noise degradation tests
        for noise_level in config["noise_levels"]:
            run_id += 1
            print(f"  Run {run_id}: {name} + noise {noise_level:.1%}", end="")
            
            # Generate noisy pattern
            degraded = generator.apply_noise(pattern, noise_level, ensure_randomness=True)
            
            # Retrieve pattern
            metrics = qhnn.retrieve(
                degraded, pattern,
                shots=config["repetitions"],
                add_quantum_noise=True
            )
            
            # Complexity analysis
            cue_lz = lz_complexity(metrics["cue_bitstring"])
            cue_ent = shannon_entropy(metrics["cue_bitstring"])
            out_lz = lz_complexity(metrics["raw_output_bitstring"])
            out_ent = shannon_entropy(metrics["raw_output_bitstring"])
            
            # Emergence detection
            emergence = detect_emergence(
                noise_level, "noise", metrics, patterns, names, PATTERN_SIZE
            )
            
            print(f" -> Accuracy: {metrics['prediction_accuracy']:.2f}, "
                  f"Emergent: {emergence['is_emergent']}")
            
            results.append([
                f"QHNN_{PATTERN_SIZE}x{PATTERN_SIZE}_noise_{name}_{run_id}",
                datetime.now().isoformat(),
                name, "noise", noise_level,
                PATTERN_SIZE * PATTERN_SIZE, len(patterns),
                metrics["prediction_accuracy"],
                metrics["hamming_distance"],
                metrics["confidence"],
                metrics["convergence_quality"],
                metrics["raw_output_bitstring"],
                metrics["cue_bitstring"],
                cue_lz, cue_ent, out_lz, out_ent,
                emergence["is_emergent"],
                emergence["emergence_type"],
                emergence["emergence_score"],
                emergence["closest_pattern"],
                emergence["closest_distance"],
                emergence["error_correction_capability"],
                metrics["energy"],
                metrics["spurious_state"],
                f"{PATTERN_SIZE}x{PATTERN_SIZE}",
                NUM_QUBITS
            ])
        
        # Masking degradation tests
        for mask_level in config["masking_levels"]:
            run_id += 1
            print(f"  Run {run_id}: {name} + mask {mask_level:.1%}", end="")
            
            # Generate masked pattern
            degraded = generator.apply_mask(pattern, mask_level, ensure_randomness=True)
            
            # Retrieve pattern
            metrics = qhnn.retrieve(
                degraded, pattern,
                shots=config["repetitions"],
                add_quantum_noise=True
            )
            
            # Complexity analysis
            cue_lz = lz_complexity(metrics["cue_bitstring"])
            cue_ent = shannon_entropy(metrics["cue_bitstring"])
            out_lz = lz_complexity(metrics["raw_output_bitstring"])
            out_ent = shannon_entropy(metrics["raw_output_bitstring"])
            
            # Emergence detection
            emergence = detect_emergence(
                mask_level, "incompleteness", metrics, patterns, names, PATTERN_SIZE
            )
            
            print(f" -> Accuracy: {metrics['prediction_accuracy']:.2f}, "
                  f"Emergent: {emergence['is_emergent']}")
            
            results.append([
                f"QHNN_{PATTERN_SIZE}x{PATTERN_SIZE}_mask_{name}_{run_id}",
                datetime.now().isoformat(),
                name, "incompleteness", mask_level,
                PATTERN_SIZE * PATTERN_SIZE, len(patterns),
                metrics["prediction_accuracy"],
                metrics["hamming_distance"],
                metrics["confidence"],
                metrics["convergence_quality"],
                metrics["raw_output_bitstring"],
                metrics["cue_bitstring"],
                cue_lz, cue_ent, out_lz, out_ent,
                emergence["is_emergent"],
                emergence["emergence_type"],
                emergence["emergence_score"],
                emergence["closest_pattern"],
                emergence["closest_distance"],
                emergence["error_correction_capability"],
                metrics["energy"],
                metrics["spurious_state"],
                f"{PATTERN_SIZE}x{PATTERN_SIZE}",
                NUM_QUBITS
            ])
    
    # Save results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    
    # Runtime calculation
    runtime = (time.time() - start_time) / 60
    
    # Validation
    if config.get("validate", True):
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        validation = validate_results(results, PATTERN_SIZE)
        
        print(f"Pattern size: {PATTERN_SIZE}x{PATTERN_SIZE}")
        print(f"Noise emergence rate: {validation['noise_emergence_rate']:.1%}")
        print(f"Mask emergence rate: {validation['mask_emergence_rate']:.1%}")
        
        if validation["anomaly_detected"] or validation["warnings"]:
            print("\n⚠️ WARNINGS:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        else:
            print("\n✓ No anomalies detected")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    total_emergent = sum(1 for r in results if r[17])
    print(f"Pattern size: {PATTERN_SIZE}x{PATTERN_SIZE}")
    print(f"Total runs: {len(results)}")
    print(f"Emergent cases: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    print(f"Runtime: {runtime:.1f} minutes")
    print(f"Qubits used: {NUM_QUBITS}")
    
    # Detailed breakdown
    for dtype in ["noise", "incompleteness"]:
        dtype_results = [r for r in results if r[3] == dtype]
        dtype_emergent = sum(1 for r in dtype_results if r[17])
        if dtype_results:
            print(f"  {dtype}: {dtype_emergent}/{len(dtype_results)} "
                  f"({dtype_emergent/len(dtype_results)*100:.1f}%)")
    
    # Emergence type breakdown
    emergence_types = {}
    for r in results:
        if r[17]:  # is_emergent
            etype = r[18]  # emergence_type
            emergence_types[etype] = emergence_types.get(etype, 0) + 1
    
    if emergence_types:
        print("\nEmergence breakdown by type:")
        for etype, count in sorted(emergence_types.items()):
            print(f"  {etype}: {count}")
    
    # Pattern-specific performance
    print("\nPer-pattern performance:")
    for pname in names:
        pattern_results = [r for r in results if r[2] == pname]
        pattern_emergent = sum(1 for r in pattern_results if r[17])
        avg_accuracy = np.mean([r[7] for r in pattern_results])
        print(f"  {pname}: {pattern_emergent}/{len(pattern_results)} emergent, "
              f"avg accuracy: {avg_accuracy:.3f}")
    
    return results, validation if config.get("validate", True) else None

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SCALABLE QUANTUM HOPFIELD NETWORK EXPERIMENT")
    print("="*60)
    print("\nThis experiment tests a Quantum Hopfield Neural Network")
    print("with properly scaled pattern sizes and qubit allocations.")
    print("\nEach configuration uses N² qubits for NxN patterns,")
    print("ensuring scientifically valid problem-solution scaling.")
    print("\nSmaller patterns (3x3) run faster but have less complexity,")
    print("while larger patterns (5x5) provide richer dynamics.")
    
    # Configuration for experiments
    config = {
        "noise_levels": [0.1, 0.15, 0.2, 0.25, 0.3],  # Granular levels
        "masking_levels": [0.1, 0.15, 0.2, 0.25, 0.3],
        "repetitions": DEFAULT_SHOTS,  # Adjusted based on pattern size
        "temperature": TEMPERATURE,  # Size-specific temperature
        "output_dir": "results",
        "validate": True
    }
    
    results, validation = run_experiments(config)
    
    print(f"\n✅ {PATTERN_SIZE}x{PATTERN_SIZE} experiment complete!")
    print("=" * 60)