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
Title         : {Quantum Hopfield Neural Network Pattern Matching Implementation}
Date          : {05-18-2025}
Description   : {This code implements a GPU-accelerated Quantum Hopfield Neural Network that tests whether complexity and entropy measures can predict emergent pattern retrieval behaviors when the network reconstructs degraded 5×5 character patterns (T, O, X) under various levels of noise and masking.}
Options       : {}
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
warnings.filterwarnings('ignore')

# Quantum backend handling
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Using simulation fallback.")

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Seed management for reproducibility
GLOBAL_SEED = None  # Set to integer for reproducibility, None for true randomness
QUANTUM_NOISE_LEVEL = 0.05  # Quantum noise parameter for measurement uncertainty

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
# PATTERN GENERATION MODULE
# ============================================================================

class PatternGenerator:
    """Pattern generation with controlled randomness."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
    
    @staticmethod
    def get_fixed_patterns() -> Tuple[List[np.ndarray], List[str]]:
        """Generate fixed 5x5 character patterns."""
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
        """
        Apply random bit-flip noise to pattern.
        
        Args:
            pattern: Original pattern
            noise_level: Fraction of bits to flip
            ensure_randomness: Force new random state for each call
        """
        if ensure_randomness and self.seed is None:
            # Create new random state for true randomness
            local_rng = np.random.RandomState()
        else:
            local_rng = self.rng
            
        flat = pattern.flatten().copy()
        n_flip = int(noise_level * len(flat))
        
        if n_flip > 0:
            # Randomly select bits to flip
            flip_indices = local_rng.choice(len(flat), n_flip, replace=False)
            flat[flip_indices] *= -1
            
        return flat.reshape(pattern.shape)
    
    def apply_mask(self, pattern: np.ndarray, mask_level: float,
                  ensure_randomness: bool = True) -> np.ndarray:
        """
        Apply masking to pattern by setting bits to unknown (0).
        
        Args:
            pattern: Original pattern
            mask_level: Fraction of bits to mask
            ensure_randomness: Force new random state for each call
        """
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
# QUANTUM HOPFIELD NETWORK
# ============================================================================

class QuantumHopfieldNetwork:
    """
    Quantum Hopfield Network with stochastic dynamics.
    
    Implements:
    - Hebbian learning for pattern storage
    - Stochastic field inference for masked bits
    - Quantum/classical evolution dynamics
    - Energy landscape analysis
    """
    
    def __init__(self, num_neurons: int, temperature: float = 0.1):
        """
        Initialize network.
        
        Args:
            num_neurons: Number of neurons/qubits
            temperature: Temperature parameter for stochastic dynamics
        """
        self.num_neurons = num_neurons
        self.temperature = temperature
        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float32)
        self.stored_patterns = []
        self.pattern_names = []
        self.rng = np.random.RandomState()
    
    def store_patterns(self, patterns: List[np.ndarray], names: List[str]):
        """
        Store patterns using Hebbian learning rule.
        
        Args:
            patterns: List of patterns to store
            names: Names for the patterns
        """
        self.stored_patterns = [p.astype(np.int8) for p in patterns]
        self.pattern_names = names
        
        if not patterns:
            return
            
        # Standard Hebbian learning
        pattern_matrix = np.array([p.flatten() for p in patterns], dtype=np.float32)
        self.weights = np.dot(pattern_matrix.T, pattern_matrix) / len(patterns)
        np.fill_diagonal(self.weights, 0)
    
    def stochastic_field_inference(self, pattern_1d: np.ndarray) -> np.ndarray:
        """
        Stochastic inference for masked bits using local fields.
        
        Uses Boltzmann probability distribution for spin assignment
        based on local field strength and temperature.
        
        Args:
            pattern_1d: Input pattern with possible masked bits (0s)
            
        Returns:
            Pattern with all bits filled
        """
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
        """
        Calculate Hopfield energy for a state.
        
        Energy function: E = -0.5 * sum_ij W_ij * s_i * s_j
        
        Args:
            state: Network state
            
        Returns:
            Energy value
        """
        state_flat = state.flatten()
        return -0.5 * np.dot(state_flat, np.dot(self.weights, state_flat))
    
    def is_spurious_state(self, state: np.ndarray, threshold: float = 0.2) -> bool:
        """
        Check if state is spurious (not close to any stored pattern).
        
        Args:
            state: State to check
            threshold: Distance threshold for spurious classification
            
        Returns:
            True if state is spurious
        """
        state_flat = state.flatten()
        
        for pattern in self.stored_patterns:
            pattern_flat = pattern.flatten()
            overlap = np.dot(state_flat, pattern_flat) / len(state_flat)
            if abs(overlap) > (1 - threshold):  # Close to stored pattern
                return False
        
        return True  # Far from all stored patterns
    
    def retrieve(self, cue_2d: np.ndarray, original_2d: np.ndarray,
                shots: int = 200, add_quantum_noise: bool = True) -> Dict[str, Any]:
        """
        Retrieve pattern from degraded cue using stochastic dynamics.
        
        Args:
            cue_2d: Degraded input pattern
            original_2d: Original pattern for comparison
            shots: Number of independent retrieval attempts
            add_quantum_noise: Add quantum measurement noise
            
        Returns:
            Dictionary with retrieval metrics
        """
        shape = cue_2d.shape
        cue_flat = cue_2d.flatten()
        original_flat = original_2d.flatten()
        
        # Run multiple independent retrievals for statistics
        outcomes = []
        energies = []
        
        for _ in range(min(shots, 50)):  # Limit for performance
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
        GPU-optimized quantum circuit evolution with Ising dynamics.
        
        Args:
            initial_state: Initial state for circuit
            add_noise: Whether to add quantum noise
            
        Returns:
            Measured state after evolution
        """
        # GPU device selection with fallbacks
        try:
            # Try lightning.gpu first (GPU acceleration)
            dev = qml.device("lightning.gpu", wires=self.num_neurons, shots=1)
            device_name = "GPU (lightning.gpu)"
        except Exception as e:
            try:
                # Fallback to lightning.qubit (optimized CPU)
                dev = qml.device("lightning.qubit", wires=self.num_neurons, shots=1)
                device_name = "CPU (lightning.qubit)"
            except Exception as e2:
                # Final fallback to default.qubit
                dev = qml.device("default.qubit", wires=self.num_neurons, shots=1)
                device_name = "CPU (default.qubit)"
        
        # Print device selection (only for first few calls to avoid spam)
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
            for i in range(self.num_neurons):
                for j in range(i + 1, self.num_neurons):
                    if abs(self.weights[i, j]) > 1e-6:
                        coupling = self.weights[i, j]
                        if add_noise:
                            coupling += QUANTUM_NOISE_LEVEL * self.rng.randn()
                        qml.IsingZZ(2.0 * coupling, wires=[i, j])
            
            return qml.sample()
        
        # Execute circuit
        try:
            measurement = circuit().flatten()
            return np.where(measurement == 0, 1, -1).astype(np.int8)
        except Exception as e:
            print(f"⚠️  Quantum evolution failed: {e}")
            # Fallback to classical evolution
            return self._classical_evolution(initial_state)
        
    def _classical_evolution(self, initial_state: np.ndarray) -> np.ndarray:
        """
        Classical stochastic Hopfield dynamics.
        
        Args:
            initial_state: Initial state
            
        Returns:
            Evolved state
        """
        state = initial_state.copy()
        
        # Multiple update steps with Glauber dynamics
        for _ in range(10):
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
        """
        Measure convergence quality to stored patterns.
        
        Args:
            retrieved: Retrieved pattern
            
        Returns:
            Maximum overlap with stored patterns
        """
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
    """
    Calculate Lempel-Ziv complexity of a string.
    
    Args:
        s: Input string
        
    Returns:
        LZ complexity value
    """
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
    """
    Calculate Shannon entropy of a string.
    
    Args:
        s: Input string
        
    Returns:
        Shannon entropy in bits
    """
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
# EMERGENCE DETECTION
# ============================================================================

def detect_emergence(degradation_level: float, degradation_type: str,
                    metrics: Dict, patterns: List[np.ndarray],
                    pattern_names: List[str]) -> Dict[str, Any]:
    """
    Detect emergent behavior in pattern retrieval.
    
    Args:
        degradation_level: Level of pattern degradation
        degradation_type: Type of degradation (noise or incompleteness)
        metrics: Retrieval metrics
        patterns: Stored patterns
        pattern_names: Names of patterns
        
    Returns:
        Dictionary with emergence assessment
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
    
    # Emergence criteria evaluation
    is_emergent = False
    emergence_type = "none"
    emergence_score = 0.0
    
    # Different criteria for different degradation types
    if degradation_type == "noise":
        # Noise degradation criteria
        if degradation_level >= 0.15:
            if min_dist == 0 and confidence > 0.8 and not spurious:
                is_emergent = True
                emergence_type = "perfect_recall"
                emergence_score = 1.0
            elif min_dist <= 3 and confidence > 0.6 and not spurious:
                is_emergent = True
                emergence_type = "robust_recall"
                emergence_score = max(0, 0.8 - min_dist * 0.1)
    
    elif degradation_type == "incompleteness":
        # Masking degradation criteria
        if degradation_level >= 0.2:
            if min_dist == 0 and confidence > 0.9 and not spurious:
                is_emergent = True
                emergence_type = "perfect_recall"
                emergence_score = 0.9
            elif min_dist <= 2 and confidence > 0.8 and not spurious:
                is_emergent = True
                emergence_type = "robust_recall"
                emergence_score = max(0, 0.7 - min_dist * 0.15)
    
    # Error correction capability
    expected_errors = degradation_level * len(retrieved)
    actual_errors = min_dist
    error_correction = max(0, 1 - actual_errors / max(expected_errors, 1))
    
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

def validate_results(results: List[List]) -> Dict[str, Any]:
    """
    Perform statistical validation of experimental results.
    
    Args:
        results: List of experimental results
        
    Returns:
        Validation metrics and warnings
    """
    noise_results = [r for r in results if r[3] == "noise"]
    mask_results = [r for r in results if r[3] == "incompleteness"]
    
    noise_emergence = sum(1 for r in noise_results if r[17])
    mask_emergence = sum(1 for r in mask_results if r[17])
    
    validation = {
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
    
    if validation["mask_emergence_rate"] > 0.8:
        validation["warnings"].append(
            "High emergence rate for masking may indicate deterministic behavior"
        )
    
    return validation

# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_experiments(config: Optional[Dict] = None):
    """
    Execute pattern matching experiments.
    
    Args:
        config: Experimental configuration dictionary
        
    Returns:
        Results and validation metrics
    """
    if config is None:
        config = {
            "noise_levels": [0.1, 0.2, 0.3],
            "masking_levels": [0.1, 0.2, 0.3],
            "repetitions": 100,
            "temperature": 0.1,
            "output_dir": "results",
            "validate": True
        }
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    print("=" * 60)
    print("QUANTUM HOPFIELD NETWORK PATTERN MATCHING EXPERIMENTS")
    print("=" * 60)
    print(f"Temperature: {config.get('temperature', 0.1)}")
    print(f"Quantum noise: {QUANTUM_NOISE_LEVEL}")
    print(f"Seed: {GLOBAL_SEED or 'True random'}")
    print()
    
    # Initialize components
    generator = PatternGenerator(seed=GLOBAL_SEED)
    patterns, names = generator.get_fixed_patterns()
    
    qhnn = QuantumHopfieldNetwork(
        num_neurons=25,
        temperature=config.get("temperature", 0.1)
    )
    qhnn.store_patterns([p.flatten() for p in patterns], names)
    
    # Output setup
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], f"qhnn_results_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "pattern_name", "degradation_type", "degradation_level",
        "pattern_size", "num_patterns_stored", "prediction_accuracy", "hamming_distance",
        "confidence", "convergence_quality", "raw_output_bitstring", "cue_bitstring",
        "cue_lz_complexity", "cue_shannon_entropy", "output_lz_complexity", "output_shannon_entropy",
        "is_emergent", "emergence_type", "emergence_score", "closest_pattern",
        "closest_distance", "error_correction_capability", "energy", "spurious_state"
    ]
    
    results = []
    run_id = 0
    
    print("Running experiments...\n")
    
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
                noise_level, "noise", metrics, patterns, names
            )
            
            print(f" -> Accuracy: {metrics['prediction_accuracy']:.2f}, "
                  f"Emergent: {emergence['is_emergent']}")
            
            results.append([
                f"QHNN_noise_{name}_{run_id}",
                datetime.now().isoformat(),
                name, "noise", noise_level,
                25, 3,
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
                metrics["spurious_state"]
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
                mask_level, "incompleteness", metrics, patterns, names
            )
            
            print(f" -> Accuracy: {metrics['prediction_accuracy']:.2f}, "
                  f"Emergent: {emergence['is_emergent']}")
            
            results.append([
                f"QHNN_mask_{name}_{run_id}",
                datetime.now().isoformat(),
                name, "incompleteness", mask_level,
                25, 3,
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
                metrics["spurious_state"]
            ])
    
    # Save results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    
    # Validation
    if config.get("validate", True):
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        validation = validate_results(results)
        
        print(f"Noise emergence rate: {validation['noise_emergence_rate']:.1%}")
        print(f"Mask emergence rate: {validation['mask_emergence_rate']:.1%}")
        
        if validation["anomaly_detected"]:
            print("\n⚠️  ANOMALIES DETECTED:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        else:
            print("\n✓ No anomalies detected")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_emergent = sum(1 for r in results if r[17])
    print(f"Total runs: {len(results)}")
    print(f"Emergent cases: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    
    # Detailed breakdown
    for dtype in ["noise", "incompleteness"]:
        dtype_results = [r for r in results if r[3] == dtype]
        dtype_emergent = sum(1 for r in dtype_results if r[17])
        if dtype_results:
            print(f"  {dtype}: {dtype_emergent}/{len(dtype_results)} "
                  f"({dtype_emergent/len(dtype_results)*100:.1f}%)")
    
    return results, validation if config.get("validate", True) else None

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration for experiments
    config = {
        "noise_levels": [0.1, 0.15, 0.2, 0.25, 0.3],  # Granular levels
        "masking_levels": [0.1, 0.15, 0.2, 0.25, 0.3],
        "repetitions": 50,  # Number of measurement shots
        "temperature": 0.1,  # Stochastic dynamics temperature
        "output_dir": "results",
        "validate": True
    }
    
    results, validation = run_experiments(config)
    
    print("\n✅ Experiment complete!")
