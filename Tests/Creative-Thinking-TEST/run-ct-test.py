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
Title         : {Quantum Associative Memory Creative Thinking Implementation}
Date          : {05-18-2025}
Description   : {This code implements a GPU-accelerated Quantum Associative Memory network that tests whether complexity and entropy measures can predict emergent creative behaviors when the network generates novel conceptual associations from thematic prompts using a predefined semantic network.}
Options       : {}
Dependencies  : {pip install numpy scipy pandas matplotlib seaborn statsmodels pennylane>=0.30.0 pennylane-lightning[gpu] torch torchvision torchaudio cupy-cuda118 scikit-learn plotly openai anthropic google-generativeai python-dotenv}
Requirements  : {Python 3.8+}
Usage         : {python run-ct-test.py}
Notes         : {Available at Github at https://github.com/alexpujols/QNAI-test-engine}
'''

import json
import csv
import os
import sys
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
from dataclasses import dataclass
import warnings
import asyncio
import aiohttp
warnings.filterwarnings('ignore')

# ============================================================================
# USER INPUT FOR QUBIT CONFIGURATION
# ============================================================================

def get_qubit_configuration():
    """Get the number of qubits from user input with validation."""
    print("\n" + "=" * 60)
    print("QUANTUM CIRCUIT CONFIGURATION FOR CREATIVE THINKING")
    print("=" * 60)
    
    while True:
        try:
            print("\nEnter the number of qubits for the creative thinking experiment")
            print("Recommended ranges:")
            print("  - 8-10 qubits: Fast execution, basic creativity")
            print("  - 11-15 qubits: Balanced creativity and runtime")
            print("  - 16-20 qubits: Enhanced creative associations")
            print("  - 21-25 qubits: Maximum creative complexity")
            print("\nNote: Higher qubit counts enable richer creative associations")
            print("but require more computational resources.")
            
            num_qubits = input("\nNumber of qubits (8-25): ").strip()
            num_qubits = int(num_qubits)
            
            if num_qubits < 8:
                print("⚠  Minimum 8 qubits required for meaningful creative associations.")
                continue
            elif num_qubits > 25:
                print("⚠  Maximum 25 qubits supported to maintain practical runtime.")
                continue
            
            # Confirm with user
            print(f"\nYou selected {num_qubits} qubits.")
            
            # Estimate runtime based on qubit count
            if num_qubits <= 10:
                est_time = "5-10"
            elif num_qubits <= 15:
                est_time = "10-20"
            elif num_qubits <= 20:
                est_time = "20-30"
            else:
                est_time = "30-45"
            
            print(f"Estimated runtime: {est_time} minutes")
            print(f"Creative space dimensionality: {2**num_qubits} quantum states")
            
            confirm = input("Proceed with this configuration? (y/n): ").strip().lower()
            if confirm == 'y' or confirm == 'yes':
                return num_qubits
            elif confirm == 'n' or confirm == 'no':
                continue
            else:
                print("Please enter 'y' for yes or 'n' for no.")
                
        except ValueError:
            print("⚠  Please enter a valid integer between 8 and 25.")
        except KeyboardInterrupt:
            print("\n\nExiting configuration...")
            sys.exit(0)

# Get user configuration
NUM_QUBITS = get_qubit_configuration()

# ============================================================================
# CONFIGURATION CONSTANTS - DYNAMICALLY ADJUSTED FOR QUBIT COUNT
# ============================================================================

# Seed management for reproducibility
GLOBAL_SEED = None  # Set to integer for reproducibility, None for true randomness

# Quantum noise level scales with qubit count
QUANTUM_NOISE_LEVEL = 0.05 * (1 + (NUM_QUBITS - 10) / 40)  # Scale noise with qubits

# Adjust shots based on qubit count for balance
if NUM_QUBITS <= 10:
    SHOTS = 200
elif NUM_QUBITS <= 15:
    SHOTS = 150
elif NUM_QUBITS <= 20:
    SHOTS = 100
else:
    SHOTS = 75  # Reduce shots for very high qubit counts

print(f"\n✓ Quantum configuration set:")
print(f"  - Qubits: {NUM_QUBITS}")
print(f"  - Shots: {SHOTS}")
print(f"  - Quantum noise: {QUANTUM_NOISE_LEVEL:.3f}")
print("=" * 60)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads the .env file
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Using system environment variables only.")
    print("Install with: pip install python-dotenv")

# Quantum backend handling
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Using simulation fallback.")

# AI scoring libraries with updated imports
AI_SCORING_AVAILABLE = False
try:
    # Updated OpenAI import for v1.0+
    from openai import OpenAI
    
    # Anthropic import
    from anthropic import Anthropic
    
    # Google Gemini import
    import google.generativeai as genai
    
    AI_SCORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI scoring libraries not available: {e}")
    print("Using mock scoring.")

# API Keys from environment variables (loaded from .env file)
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
    "google": os.environ.get("GOOGLE_API_KEY")
}

# Check if API keys are properly set
def check_api_keys():
    """Check if API keys are set in environment variables."""
    missing_keys = []
    for service, key in API_KEYS.items():
        if not key:
            missing_keys.append(service.upper())
    
    if missing_keys and AI_SCORING_AVAILABLE:
        print("⚠️  Warning: Missing API keys for:", ", ".join(missing_keys))
        if not DOTENV_AVAILABLE:
            print("   Install python-dotenv to load from .env file: pip install python-dotenv")
        else:
            print("   Check your .env file contains:")
            for service in missing_keys:
                print(f"   {service}_API_KEY=your-api-key-here")
        print("   Using mock scoring as fallback.\n")
        return False
    elif AI_SCORING_AVAILABLE:
        print("✅ All API keys loaded successfully from environment")
    return True

# Creativity thresholds - adjust based on qubit count
BASE_CREATIVITY_THRESHOLD = 5.5
CREATIVITY_THRESHOLD = BASE_CREATIVITY_THRESHOLD - (NUM_QUBITS - 10) * 0.05  # Lower threshold for higher qubits
SEMANTIC_DISTANCE_THRESHOLD = 0.4 - (NUM_QUBITS - 10) * 0.01  # Adjust for qubit count

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(slots=True)
class CreativeOutput:
    """Container for creative output and metrics."""
    prompt_concept: str
    generated_concepts: List[str]
    output_vectors: List[np.ndarray]
    ai_scores: Dict[str, Dict[str, float]]  # {concept: {novelty, relevance, surprise}}
    semantic_distances: List[float]
    confidence_scores: List[float]
    raw_output_bitstrings: List[str]
    prompt_bitstring: str
    energy_values: List[float]

@dataclass(slots=True)
class SemanticNetwork:
    """Predefined semantic network structure."""
    concepts: List[str]
    concept_vectors: Dict[str, np.ndarray]
    associations: List[Tuple[str, str]]
    association_matrix: np.ndarray

# ============================================================================
# SEMANTIC NETWORK GENERATION
# ============================================================================

class SemanticNetworkGenerator:
    """Generate and manage the predefined semantic network."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self.network = self._create_adaptive_network()
    
    def _create_adaptive_network(self) -> SemanticNetwork:
        """
        Create semantic network that adapts to qubit count.
        More qubits = more concepts and richer associations.
        """
        # Base concepts (always present)
        base_concepts = [
            "animal", "machine", "emotion", "nature", "algorithm",
            "joy", "forest", "memory", "ocean", "dream"
        ]
        
        # Add more concepts for higher qubit counts
        extended_concepts = []
        if NUM_QUBITS >= 12:
            extended_concepts.extend(["wisdom", "chaos", "harmony", "energy"])
        if NUM_QUBITS >= 16:
            extended_concepts.extend(["infinity", "quantum", "consciousness", "emergence"])
        if NUM_QUBITS >= 20:
            extended_concepts.extend(["paradox", "beauty", "truth", "mystery"])
        if NUM_QUBITS >= 24:
            extended_concepts.extend(["transcendence", "void", "creation", "entropy"])
        
        concepts = base_concepts + extended_concepts
        num_concepts = len(concepts)
        
        # Create vectors with dimensionality matching available qubits
        vector_dim = min(NUM_QUBITS, num_concepts)
        concept_vectors = {}
        
        for i, concept in enumerate(concepts):
            vec = np.zeros(vector_dim, dtype=np.float32)
            if i < vector_dim:
                vec[i] = 1.0  # One-hot for first concepts
            else:
                # Distributed representation for additional concepts
                indices = self.rng.choice(vector_dim, size=3, replace=False)
                for idx in indices:
                    vec[idx] = self.rng.uniform(0.3, 0.7)
            vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
            concept_vectors[concept] = vec
        
        # Create associations - more for higher qubit counts
        base_associations = [
            ("animal", "nature"), ("animal", "forest"), ("machine", "algorithm"),
            ("machine", "memory"), ("emotion", "joy"), ("emotion", "dream"),
            ("nature", "forest"), ("nature", "ocean"), ("algorithm", "memory"),
            ("joy", "dream"), ("forest", "ocean"), ("memory", "dream"),
            ("animal", "emotion"), ("machine", "dream"), ("nature", "joy"),
            ("algorithm", "emotion"), ("forest", "dream"), ("ocean", "emotion"),
            ("memory", "nature"), ("joy", "animal")
        ]
        
        associations = base_associations.copy()
        
        # Add more complex associations for higher qubit counts
        if NUM_QUBITS >= 12:
            associations.extend([
                ("wisdom", "memory"), ("chaos", "energy"), 
                ("harmony", "nature"), ("energy", "machine")
            ])
        if NUM_QUBITS >= 16:
            associations.extend([
                ("quantum", "infinity"), ("consciousness", "dream"),
                ("emergence", "chaos"), ("quantum", "algorithm")
            ])
        if NUM_QUBITS >= 20:
            associations.extend([
                ("paradox", "truth"), ("beauty", "harmony"),
                ("mystery", "quantum"), ("paradox", "infinity")
            ])
        
        # Create association matrix
        association_matrix = np.zeros((num_concepts, num_concepts), dtype=np.float32)
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        
        for c1, c2 in associations:
            if c1 in concept_to_idx and c2 in concept_to_idx:
                idx1, idx2 = concept_to_idx[c1], concept_to_idx[c2]
                association_matrix[idx1, idx2] = 1.0
                association_matrix[idx2, idx1] = 1.0  # Bidirectional
        
        return SemanticNetwork(
            concepts=concepts,
            concept_vectors=concept_vectors,
            associations=associations,
            association_matrix=association_matrix
        )
    
    def get_prompt_themes(self) -> List[str]:
        """Get core themes for prompt generation."""
        themes = ["animal", "machine", "emotion", "nature", "algorithm"]
        if NUM_QUBITS >= 16:
            themes.extend(["quantum", "consciousness"])
        if NUM_QUBITS >= 20:
            themes.extend(["paradox", "beauty"])
        return themes
    
    def generate_prompt_vector(self, theme: str, add_noise: bool = True) -> np.ndarray:
        """
        Generate a prompt vector based on a theme.
        
        Args:
            theme: Core theme for the prompt
            add_noise: Add slight noise for variation
            
        Returns:
            Vector with dimensionality matching qubit count
        """
        if theme not in self.network.concept_vectors:
            # If theme not in concepts, use random concept
            theme = self.rng.choice(self.network.concepts)
        
        base_vec = self.network.concept_vectors[theme]
        
        # Extend or truncate to match qubit count
        if len(base_vec) < NUM_QUBITS:
            prompt_vec = np.zeros(NUM_QUBITS, dtype=np.float32)
            prompt_vec[:len(base_vec)] = base_vec
        else:
            prompt_vec = base_vec[:NUM_QUBITS].copy()
        
        if add_noise:
            # Add small noise while maintaining sparsity
            noise = self.rng.normal(0, QUANTUM_NOISE_LEVEL, size=NUM_QUBITS)
            prompt_vec = prompt_vec + noise
            prompt_vec = np.clip(prompt_vec, -1, 1)
            # Re-normalize to maintain unit energy
            norm = np.linalg.norm(prompt_vec)
            if norm > 0:
                prompt_vec = prompt_vec / norm
        
        return prompt_vec

# ============================================================================
# QUANTUM ASSOCIATIVE MEMORY
# ============================================================================

class QuantumAssociativeMemory:
    """
    Quantum Associative Memory for creative association generation.
    Dynamically adapts to the specified number of qubits.
    """
    
    def __init__(self, num_qubits: int = NUM_QUBITS, temperature: float = 0.5):
        """
        Initialize QAM with adaptive qubit count.
        
        Args:
            num_qubits: Number of qubits (8-25)
            temperature: Temperature for probabilistic dynamics
        """
        self.num_qubits = num_qubits
        self.num_dimensions = num_qubits  # Direct mapping
        self.temperature = temperature * (1 + (num_qubits - 10) / 20)  # Scale with qubits
        self.weights = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)
        self.stored_associations = []
        self.semantic_network = None
        self.rng = np.random.RandomState()
        
        print(f"Initializing {num_qubits}-qubit Quantum Associative Memory...")
        print(f"  Temperature: {self.temperature:.3f}")
        print(f"  State space: {2**num_qubits} dimensions")
    
    def store_semantic_network(self, network: SemanticNetwork):
        """
        Store the semantic network using one-shot Hebbian learning.
        
        Args:
            network: SemanticNetwork object to store
        """
        self.semantic_network = network
        
        # Build weight matrix from associations
        # Adapt to qubit count
        if network.association_matrix.shape[0] <= self.num_dimensions:
            # Pad if needed
            self.weights = np.zeros((self.num_dimensions, self.num_dimensions))
            size = network.association_matrix.shape[0]
            self.weights[:size, :size] = network.association_matrix.copy()
        else:
            # Truncate if needed
            self.weights = network.association_matrix[:self.num_dimensions, :self.num_dimensions].copy()
        
        # Add self-connections with reduced weight for stability
        np.fill_diagonal(self.weights, 0.5)
        
        # Normalize weights
        max_weight = np.max(np.abs(self.weights))
        if max_weight > 0:
            self.weights = self.weights / max_weight
    
    def calculate_energy(self, state: np.ndarray) -> float:
        """
        Calculate energy of a state in the associative memory.
        
        Args:
            state: Current state vector
            
        Returns:
            Energy value
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def quantum_association_search(self, prompt: np.ndarray, 
                                  num_outputs: int = 3,
                                  shots: int = None) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Quantum search for associated concepts.
        
        Args:
            prompt: Input prompt vector
            num_outputs: Number of top outputs to return
            shots: Number of quantum measurements (uses global SHOTS if None)
            
        Returns:
            Tuple of (output vectors, confidence scores, energies)
        """
        if shots is None:
            shots = SHOTS
            
        if PENNYLANE_AVAILABLE and shots > 10:
            return self._quantum_search(prompt, num_outputs, shots)
        else:
            return self._classical_search(prompt, num_outputs)
    
    def _quantum_search(self, prompt: np.ndarray, num_outputs: int, 
                       shots: int) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        GPU-accelerated quantum circuit for association search.
        Adapts circuit depth and entanglement to qubit count.
        """
        # Device selection with GPU preference
        try:
            dev = qml.device("lightning.gpu", wires=self.num_qubits, shots=shots)
            device_name = "GPU (lightning.gpu)"
        except:
            try:
                dev = qml.device("lightning.qubit", wires=self.num_qubits, shots=shots)
                device_name = "CPU (lightning.qubit)"
            except:
                dev = qml.device("default.qubit", wires=self.num_qubits, shots=shots)
                device_name = "CPU (default.qubit)"
        
        if not hasattr(self, '_device_logged'):
            print(f"🔧 Quantum device: {device_name}")
            self._device_logged = True
        
        @qml.qnode(dev)
        def association_circuit():
            # Encode prompt into quantum state
            prompt_normalized = prompt / np.linalg.norm(prompt)
            
            # Create full state vector for amplitude encoding
            state_vector = np.zeros(2**self.num_qubits, dtype=complex)
            
            # Map prompt to computational basis states
            for i in range(min(len(prompt_normalized), 2**self.num_qubits)):
                state_vector[i] = prompt_normalized[i % len(prompt_normalized)]
            
            # Normalize the full state vector
            norm = np.linalg.norm(state_vector)
            if norm > 0:
                state_vector = state_vector / norm
            
            # Prepare the quantum state
            qml.StatePrep(state_vector, wires=range(self.num_qubits))
            
            # Apply associative memory dynamics via controlled rotations
            # Adapt entanglement structure to qubit count
            
            # Base entanglement layer
            for i in range(self.num_qubits):
                for j in range(i + 1, min(i + 3, self.num_qubits)):
                    if abs(self.weights[i, j]) > 0.01:
                        # Entangling gates based on association strength
                        qml.CRZ(2.0 * self.weights[i, j] * np.pi, wires=[i, j])
                        if self.num_qubits <= 15:  # Extra entanglement for smaller circuits
                            qml.CNOT(wires=[i, j])
            
            # Additional entanglement layers for higher qubit counts
            if self.num_qubits >= 16:
                # Add long-range connections
                for i in range(0, self.num_qubits - 4, 2):
                    qml.CNOT(wires=[i, i + 4])
            
            if self.num_qubits >= 20:
                # Add all-to-all connections for subset of qubits
                for i in range(0, min(5, self.num_qubits)):
                    for j in range(i + 1, min(5, self.num_qubits)):
                        qml.CZ(wires=[i, j])
            
            # Add controlled noise for creativity
            for i in range(self.num_qubits):
                qml.RY(self.temperature * self.rng.randn() * 0.1, wires=i)
            
            # Measure in computational basis
            return qml.sample()
        
        # Execute circuit and collect measurements
        try:
            measurements = association_circuit()
            
            # Process measurements to extract top associations
            unique_states = {}
            for measurement in measurements:
                state_tuple = tuple(measurement)
                if state_tuple not in unique_states:
                    unique_states[state_tuple] = 0
                unique_states[state_tuple] += 1
            
            # Sort by frequency and convert to vectors
            sorted_states = sorted(unique_states.items(), key=lambda x: x[1], reverse=True)
            
            output_vectors = []
            confidence_scores = []
            energies = []
            
            for state, count in sorted_states[:num_outputs]:
                # Convert measurement to continuous vector
                vec = np.array([1.0 if bit == 1 else -1.0 for bit in state[:self.num_dimensions]])
                # Normalize
                vec = vec / np.linalg.norm(vec)
                
                output_vectors.append(vec)
                confidence_scores.append(count / shots)
                energies.append(self.calculate_energy(vec))
            
            # Pad if needed
            while len(output_vectors) < num_outputs:
                # Generate random creative output
                random_vec = self.rng.randn(self.num_dimensions)
                random_vec = random_vec / np.linalg.norm(random_vec)
                output_vectors.append(random_vec)
                confidence_scores.append(0.01)
                energies.append(self.calculate_energy(random_vec))
            
            return output_vectors, confidence_scores, energies
            
        except Exception as e:
            print(f"⚠️ Quantum search failed: {e}")
            return self._classical_search(prompt, num_outputs)
    
    def _classical_search(self, prompt: np.ndarray, 
                         num_outputs: int) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Classical associative search fallback.
        
        Args:
            prompt: Input prompt vector
            num_outputs: Number of outputs to generate
            
        Returns:
            Tuple of (output vectors, confidence scores, energies)
        """
        output_vectors = []
        confidence_scores = []
        energies = []
        
        # Start with prompt
        current_state = prompt.copy()
        
        for _ in range(num_outputs):
            # Apply associative dynamics
            field = np.dot(self.weights, current_state)
            
            # Add stochastic element for creativity
            field += self.temperature * self.rng.randn(self.num_dimensions)
            
            # Apply activation (tanh for bounded output)
            new_state = np.tanh(field)
            
            # Add noise for diversity
            new_state += 0.1 * self.rng.randn(self.num_dimensions)
            
            # Normalize
            new_state = new_state / np.linalg.norm(new_state)
            
            output_vectors.append(new_state)
            
            # Calculate confidence based on energy
            energy = self.calculate_energy(new_state)
            confidence = np.exp(-energy / self.temperature)
            confidence_scores.append(min(confidence, 1.0))
            energies.append(energy)
            
            # Perturb for next iteration
            current_state = new_state + 0.2 * self.rng.randn(self.num_dimensions)
            current_state = current_state / np.linalg.norm(current_state)
        
        return output_vectors, confidence_scores, energies
    
    def decode_vector_to_concept(self, vector: np.ndarray) -> str:
        """
        Decode a continuous vector to the nearest concept.
        
        Args:
            vector: Continuous vector to decode
            
        Returns:
            Closest concept name
        """
        if self.semantic_network is None:
            return "unknown"
        
        # Find closest concept by cosine similarity
        best_similarity = -1
        best_concept = "unknown"
        
        for concept, concept_vec in self.semantic_network.concept_vectors.items():
            # Ensure vectors are same size for comparison
            if len(concept_vec) != len(vector):
                if len(concept_vec) < len(vector):
                    padded_vec = np.zeros(len(vector))
                    padded_vec[:len(concept_vec)] = concept_vec
                    concept_vec = padded_vec
                else:
                    concept_vec = concept_vec[:len(vector)]
            
            similarity = np.dot(vector, concept_vec) / (np.linalg.norm(vector) * np.linalg.norm(concept_vec))
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = concept
        
        return best_concept

# ============================================================================
# AI SCORING MODULE (unchanged from original)
# ============================================================================

class AICreativityScorer:
    """
    AI-based creativity scoring using multiple LLM judges.
    """
    
    def __init__(self):
        """Initialize AI scoring clients."""
        self.initialized = False
        self.api_keys_valid = check_api_keys()
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_model = None
        
        if AI_SCORING_AVAILABLE and self.api_keys_valid:
            try:
                # Initialize OpenAI with new client interface
                if API_KEYS["openai"]:
                    self.openai_client = OpenAI(api_key=API_KEYS["openai"])
                
                # Initialize Anthropic
                if API_KEYS["anthropic"]:
                    self.anthropic_client = Anthropic(api_key=API_KEYS["anthropic"])
                
                # Initialize Google Gemini with updated model
                if API_KEYS["google"]:
                    genai.configure(api_key=API_KEYS["google"])
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
                self.initialized = True
                print("✅ AI scoring APIs initialized")
            except Exception as e:
                print(f"⚠️ AI scoring initialization failed: {e}")
                self.initialized = False
        else:
            if not self.api_keys_valid:
                print("⚠️ Using mock scoring (API keys not configured)")
            else:
                print("⚠️ Using mock scoring (AI libraries not available)")
    
    async def score_output(self, prompt: str, output: str, 
                          association_context: str) -> Dict[str, Dict[str, float]]:
        """
        Score creative output using three AI judges.
        
        Args:
            prompt: Original prompt/theme
            output: Generated concept
            association_context: Context about the semantic network
            
        Returns:
            Dictionary with scores from each judge
        """
        if not self.initialized:
            return self._mock_scoring(prompt, output)
        
        scoring_prompt = self._create_scoring_prompt(prompt, output, association_context)
        
        scores = {}
        
        # Get scores from each AI judge if available
        if self.openai_client:
            try:
                scores["openai"] = await self._score_with_openai(scoring_prompt)
            except Exception as e:
                print(f"OpenAI scoring failed: {e}")
                scores["openai"] = self._default_scores()
        else:
            scores["openai"] = self._default_scores()
        
        if self.anthropic_client:
            try:
                scores["anthropic"] = await self._score_with_anthropic(scoring_prompt)
            except Exception as e:
                print(f"Anthropic scoring failed: {e}")
                scores["anthropic"] = self._default_scores()
        else:
            scores["anthropic"] = self._default_scores()
        
        if self.gemini_model:
            try:
                scores["google"] = await self._score_with_gemini(scoring_prompt)
            except Exception as e:
                print(f"Gemini scoring failed: {e}")
                scores["google"] = self._default_scores()
        else:
            scores["google"] = self._default_scores()
        
        return scores
    
    def _create_scoring_prompt(self, prompt: str, output: str, context: str) -> str:
        """Create standardized scoring prompt for all AI judges."""
        return f"""
        You are evaluating the creativity of an AI-generated conceptual association.
        
        Context: {context}
        
        Input Prompt: {prompt}
        Generated Output: {output}
        
        Please rate the output on a scale of 1-7 for each dimension:
        
        1. NOVELTY: How original and unexpected is this association?
           (1 = completely predictable, 7 = highly original)
        
        2. RELEVANCE: How well does the output relate to the prompt?
           (1 = unrelated, 7 = perfectly relevant)
        
        3. SURPRISE: How surprising or thought-provoking is this association?
           (1 = mundane, 7 = very surprising)
        
        Respond ONLY with three numbers separated by commas in this format:
        novelty,relevance,surprise
        
        Example: 5,6,4
        """
    
    async def _score_with_openai(self, prompt: str) -> Dict[str, float]:
        """Score using OpenAI GPT with updated API."""
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20
            )
            
            scores_text = completion.choices[0].message.content.strip()
            return self._parse_scores(scores_text)
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self._default_scores()
    
    async def _score_with_anthropic(self, prompt: str) -> Dict[str, float]:
        """Score using Anthropic Claude with updated model."""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3
            )
            
            scores_text = response.content[0].text.strip()
            return self._parse_scores(scores_text)
        except Exception as e:
            print(f"Anthropic error: {e}")
            return self._default_scores()
    
    async def _score_with_gemini(self, prompt: str) -> Dict[str, float]:
        """Score using Google Gemini with updated API."""
        try:
            response = self.gemini_model.generate_content(prompt)
            scores_text = response.text.strip()
            return self._parse_scores(scores_text)
        except Exception as e:
            print(f"Gemini error: {e}")
            return self._default_scores()
    
    def _parse_scores(self, scores_text: str) -> Dict[str, float]:
        """Parse score string into dictionary."""
        try:
            parts = scores_text.split(',')
            return {
                "novelty": float(parts[0]),
                "relevance": float(parts[1]),
                "surprise": float(parts[2])
            }
        except:
            return self._default_scores()
    
    def _default_scores(self) -> Dict[str, float]:
        """Default scores when parsing fails."""
        return {"novelty": 4.0, "relevance": 4.0, "surprise": 4.0}
    
    def _mock_scoring(self, prompt: str, output: str) -> Dict[str, Dict[str, float]]:
        """Mock scoring for testing without API access."""
        # Simulate realistic scoring based on semantic distance
        base_score = 4.0 + np.random.randn() * 1.5
        
        scores = {}
        for judge in ["openai", "anthropic", "google"]:
            scores[judge] = {
                "novelty": np.clip(base_score + np.random.randn() * 0.5, 1, 7),
                "relevance": np.clip(base_score + np.random.randn() * 0.3, 1, 7),
                "surprise": np.clip(base_score + np.random.randn() * 0.4, 1, 7)
            }
        
        return scores

# ============================================================================
# COMPLEXITY METRICS (unchanged)
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

def calculate_semantic_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate semantic distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Distance measure (0 = identical, 1 = orthogonal)
    """
    # Ensure vectors are same size
    if len(vec1) != len(vec2):
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
    
    # Use cosine distance
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    distance = 1 - abs(similarity)
    return distance

# ============================================================================
# EMERGENCE DETECTION (adapted for qubit count)
# ============================================================================

def detect_creative_emergence(prompt: str, outputs: List[str], 
                            ai_scores: Dict[str, Dict[str, float]],
                            semantic_distances: List[float],
                            network: SemanticNetwork) -> Dict[str, Any]:
    """
    Detect emergent creative behavior.
    Adapts thresholds based on qubit count.
    """
    # Calculate average scores across judges and outputs
    all_scores = []
    for output_key, output_scores in ai_scores.items():
        for judge, judge_scores in output_scores.items():
            if isinstance(judge_scores, dict):
                avg_score = np.mean(list(judge_scores.values()))
                all_scores.append(avg_score)
    
    if not all_scores:
        return {
            "is_emergent": False,
            "emergence_type": "none",
            "emergence_score": 0.0,
            "creativity_score": 0.0,
            "avg_semantic_distance": 0.0
        }
    
    overall_creativity = np.mean(all_scores)
    avg_semantic_distance = np.mean(semantic_distances)
    
    # Check emergence criteria
    is_emergent = False
    emergence_type = "none"
    emergence_score = 0.0
    
    # Adjust thresholds based on qubit count
    creativity_threshold = CREATIVITY_THRESHOLD
    distance_threshold = SEMANTIC_DISTANCE_THRESHOLD
    
    # High creativity threshold
    if overall_creativity > creativity_threshold:
        # Check if outputs are not directly encoded
        novel_outputs = 0
        for output in outputs:
            if output not in network.concepts:
                novel_outputs += 1
            elif avg_semantic_distance > distance_threshold:
                novel_outputs += 0.5
        
        # More lenient for higher qubit counts
        required_novel = 2 if NUM_QUBITS <= 15 else 1.5
        
        if novel_outputs >= required_novel:
            is_emergent = True
            emergence_type = "creative_association"
            emergence_score = min(1.0, (overall_creativity - 4) / 3 * 
                                 (avg_semantic_distance / distance_threshold))
    
    # Alternative emergence: High semantic distance with relevance
    elif avg_semantic_distance > 0.6 and overall_creativity > 4.5:
        is_emergent = True
        emergence_type = "semantic_leap"
        emergence_score = min(1.0, avg_semantic_distance * (overall_creativity / 7))
    
    # Bonus for higher qubit counts achieving any creativity
    elif NUM_QUBITS >= 20 and overall_creativity > 5.0:
        is_emergent = True
        emergence_type = "quantum_creativity"
        emergence_score = min(1.0, (overall_creativity - 5.0) * (NUM_QUBITS / 25))
    
    return {
        "is_emergent": is_emergent,
        "emergence_type": emergence_type,
        "emergence_score": emergence_score,
        "creativity_score": overall_creativity,
        "avg_semantic_distance": avg_semantic_distance,
        "num_qubits": NUM_QUBITS
    }

# ============================================================================
# MAIN EXPERIMENTAL PIPELINE (adapted for user-specified qubits)
# ============================================================================

async def run_experiments(config: Optional[Dict] = None):
    """
    Execute creative thinking experiments with user-specified qubit count.
    
    Args:
        config: Experimental configuration dictionary
        
    Returns:
        Results and validation metrics
    """
    if config is None:
        config = {
            "num_prompts_per_theme": 3,
            "num_outputs": 3,
            "shots": SHOTS,
            "temperature": 0.5,
            "output_dir": "results",
            "use_ai_scoring": True
        }
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    print("\n" + "=" * 60)
    print(f"{NUM_QUBITS}-QUBIT QUANTUM CREATIVE THINKING EXPERIMENTS")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Qubits: {NUM_QUBITS}")
    print(f"  Shots: {config.get('shots', SHOTS)}")
    print(f"  Temperature: {config.get('temperature', 0.5)}")
    print(f"  Quantum noise: {QUANTUM_NOISE_LEVEL:.3f}")
    print(f"  Creativity threshold: {CREATIVITY_THRESHOLD:.2f}")
    print(f"  AI Scoring: {'Enabled' if config['use_ai_scoring'] else 'Mock'}")
    print(f"  Seed: {GLOBAL_SEED or 'True random'}")
    print(f"  .env file support: {'Enabled' if DOTENV_AVAILABLE else 'Disabled'}")
    print()
    
    # Initialize components
    network_gen = SemanticNetworkGenerator(seed=GLOBAL_SEED)
    network = network_gen.network
    
    print(f"Semantic network initialized with {len(network.concepts)} concepts")
    print(f"Association complexity: {len(network.associations)} connections")
    
    qam = QuantumAssociativeMemory(
        num_qubits=NUM_QUBITS,
        temperature=config.get("temperature", 0.5)
    )
    qam.store_semantic_network(network)
    
    # Initialize AI scorer
    scorer = AICreativityScorer() if config["use_ai_scoring"] else None
    
    # Output setup
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], 
                              f"qam_creative_results_{NUM_QUBITS}q_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "num_qubits", "prompt_theme", "prompt_concept",
        "output_1", "output_2", "output_3",
        "ai_score_novelty_avg", "ai_score_relevance_avg", "ai_score_surprise_avg",
        "semantic_distance_1", "semantic_distance_2", "semantic_distance_3",
        "confidence_1", "confidence_2", "confidence_3",
        "energy_1", "energy_2", "energy_3",
        "prompt_lz_complexity", "prompt_shannon_entropy",
        "output_lz_complexity_avg", "output_shannon_entropy_avg",
        "is_emergent", "emergence_type", "emergence_score",
        "creativity_score", "avg_semantic_distance", "shots", "temperature"
    ]
    
    results = []
    run_id = 0
    
    print("Running experiments...\n")
    
    # Test each theme
    themes = network_gen.get_prompt_themes()
    print(f"Testing {len(themes)} themes with {config['num_prompts_per_theme']} prompts each")
    
    for theme in themes:
        for prompt_iter in range(config["num_prompts_per_theme"]):
            run_id += 1
            print(f"  Run {run_id}: Theme '{theme}' (iteration {prompt_iter + 1})", end="")
            
            # Generate prompt vector
            prompt_vec = network_gen.generate_prompt_vector(theme, add_noise=True)
            
            # Generate creative associations
            output_vecs, confidences, energies = qam.quantum_association_search(
                prompt_vec,
                num_outputs=config["num_outputs"],
                shots=config["shots"]
            )
            
            # Decode outputs to concepts
            output_concepts = [qam.decode_vector_to_concept(vec) for vec in output_vecs]
            
            # Calculate semantic distances
            semantic_distances = [
                calculate_semantic_distance(prompt_vec, vec) for vec in output_vecs
            ]
            
            # AI scoring
            if scorer and scorer.initialized:
                ai_scores = {}
                for i, concept in enumerate(output_concepts):
                    scores = await scorer.score_output(
                        theme, concept,
                        f"Network has concepts: {', '.join(network.concepts[:5])}..."
                    )
                    ai_scores[f"output_{i}"] = scores
            else:
                # Mock scoring
                ai_scores = {}
                for i, concept in enumerate(output_concepts):
                    ai_scores[f"output_{i}"] = {
                        "openai": {"novelty": 4 + np.random.randn(), 
                                 "relevance": 4 + np.random.randn(),
                                 "surprise": 4 + np.random.randn()},
                        "anthropic": {"novelty": 4 + np.random.randn(),
                                    "relevance": 4 + np.random.randn(),
                                    "surprise": 4 + np.random.randn()},
                        "google": {"novelty": 4 + np.random.randn(),
                                 "relevance": 4 + np.random.randn(),
                                 "surprise": 4 + np.random.randn()}
                    }
            
            # Calculate average AI scores
            score_values = []
            for output_key in ai_scores:
                for judge in ["openai", "anthropic", "google"]:
                    if judge in ai_scores[output_key]:
                        judge_scores = ai_scores[output_key][judge]
                        if isinstance(judge_scores, dict):
                            score_values.extend([
                                judge_scores.get("novelty", 4.0),
                                judge_scores.get("relevance", 4.0),
                                judge_scores.get("surprise", 4.0)
                            ])
            
            if score_values:
                avg_novelty = np.mean([s for i, s in enumerate(score_values) if i % 3 == 0])
                avg_relevance = np.mean([s for i, s in enumerate(score_values) if i % 3 == 1])
                avg_surprise = np.mean([s for i, s in enumerate(score_values) if i % 3 == 2])
            else:
                avg_novelty = avg_relevance = avg_surprise = 4.0
            
            # Convert to bitstrings for complexity analysis
            prompt_bits = ''.join(['1' if x > 0 else '0' for x in prompt_vec])
            output_bits = [''.join(['1' if x > 0 else '0' for x in vec]) for vec in output_vecs]
            
            # Calculate complexity metrics
            prompt_lz = lz_complexity(prompt_bits)
            prompt_ent = shannon_entropy(prompt_bits)
            
            output_lz_avg = np.mean([lz_complexity(bits) for bits in output_bits])
            output_ent_avg = np.mean([shannon_entropy(bits) for bits in output_bits])
            
            # Detect emergence
            emergence = detect_creative_emergence(
                theme, output_concepts, ai_scores,
                semantic_distances, network
            )
            
            print(f" -> Creativity: {emergence['creativity_score']:.2f}, "
                  f"Emergent: {emergence['is_emergent']}")
            
            # Store results
            results.append([
                f"QAM_creative_{theme}_{run_id}",
                datetime.now().isoformat(),
                NUM_QUBITS,
                theme, theme,  # prompt theme and concept
                output_concepts[0], output_concepts[1], output_concepts[2],
                avg_novelty, avg_relevance, avg_surprise,
                semantic_distances[0], semantic_distances[1], semantic_distances[2],
                confidences[0], confidences[1], confidences[2],
                energies[0], energies[1], energies[2],
                prompt_lz, prompt_ent,
                output_lz_avg, output_ent_avg,
                emergence["is_emergent"],
                emergence["emergence_type"],
                emergence["emergence_score"],
                emergence["creativity_score"],
                emergence["avg_semantic_distance"],
                config["shots"],
                config.get("temperature", 0.5)
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
    
    total_emergent = sum(1 for r in results if r[24])  # is_emergent column
    print(f"Total runs: {len(results)}")
    print(f"Emergent cases: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    
    # Calculate average creativity scores
    avg_creativity = np.mean([r[27] for r in results])  # creativity_score column
    print(f"Average creativity score: {avg_creativity:.2f}/7.0")
    
    # Breakdown by emergence type
    emergence_types = {}
    for r in results:
        if r[24]:  # is_emergent
            etype = r[25]  # emergence_type
            emergence_types[etype] = emergence_types.get(etype, 0) + 1
    
    if emergence_types:
        print("\nEmergence types:")
        for etype, count in emergence_types.items():
            print(f"  {etype}: {count} ({count/total_emergent*100:.1f}% of emergent)")
    
    print(f"\nQuantum configuration used:")
    print(f"  Qubits: {NUM_QUBITS}")
    print(f"  Shots: {config['shots']}")
    print(f"  State space: {2**NUM_QUBITS} dimensions")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADAPTIVE QUANTUM CREATIVE THINKING EXPERIMENT")
    print("="*60)
    print("\nThis experiment uses a Quantum Associative Memory network")
    print("to generate creative conceptual associations.")
    print("\nHigher qubit counts enable:")
    print("  • Richer semantic representations")
    print("  • More complex association patterns")
    print("  • Greater creative potential")
    print("  • Increased computational requirements")
    
    # Configuration for experiments
    config = {
        "num_prompts_per_theme": 5,  # 5 prompts per theme
        "num_outputs": 3,  # Generate 3 associations per prompt
        "shots": SHOTS,  # Quantum measurement shots (dynamically set)
        "temperature": 0.5,  # Creativity temperature
        "output_dir": "results",
        "use_ai_scoring": True  # Set to True to use real AI scoring (if keys are configured)
    }
    
    # Run async experiments
    asyncio.run(run_experiments(config))
    
    print(f"\n✅ {NUM_QUBITS}-qubit creative thinking experiment complete!")
    print("=" * 60)