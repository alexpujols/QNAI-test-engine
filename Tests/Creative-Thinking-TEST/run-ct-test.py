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

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Seed management for reproducibility
GLOBAL_SEED = None  # Set to integer for reproducibility, None for true randomness
QUANTUM_NOISE_LEVEL = 0.05  # Quantum noise parameter for measurement uncertainty

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

# Creativity thresholds
CREATIVITY_THRESHOLD = 5.5  # Average score > 5.5 on 1-7 scale
SEMANTIC_DISTANCE_THRESHOLD = 0.4  # Minimum distance for novelty

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
        self.network = self._create_fixed_network()
    
    def _create_fixed_network(self) -> SemanticNetwork:
        """
        Create the fixed semantic network with 10 concepts and 20 bidirectional associations.
        Using sparse one-hot encoding (10-dimensional vectors).
        """
        # Define 10 core concepts across 5 themes
        concepts = [
            "animal",     # Theme 1: Living beings
            "machine",    # Theme 2: Technology
            "emotion",    # Theme 3: Feelings
            "nature",     # Theme 1: Living beings
            "algorithm",  # Theme 2: Technology  
            "joy",        # Theme 3: Feelings
            "forest",     # Theme 4: Environment
            "memory",     # Theme 5: Cognition
            "ocean",      # Theme 4: Environment
            "dream"       # Theme 5: Cognition
        ]
        
        # Create one-hot encoded vectors (10-dimensional sparse encoding)
        concept_vectors = {}
        for i, concept in enumerate(concepts):
            vec = np.zeros(10, dtype=np.float32)
            vec[i] = 1.0
            concept_vectors[concept] = vec
        
        # Define 20 bidirectional associations (meaningful connections)
        associations = [
            ("animal", "nature"),
            ("animal", "forest"),
            ("machine", "algorithm"),
            ("machine", "memory"),
            ("emotion", "joy"),
            ("emotion", "dream"),
            ("nature", "forest"),
            ("nature", "ocean"),
            ("algorithm", "memory"),
            ("joy", "dream"),
            ("forest", "ocean"),
            ("memory", "dream"),
            ("animal", "emotion"),  # Cross-theme associations
            ("machine", "dream"),
            ("nature", "joy"),
            ("algorithm", "emotion"),
            ("forest", "dream"),
            ("ocean", "emotion"),
            ("memory", "nature"),
            ("joy", "animal")
        ]
        
        # Create association matrix (10x10)
        association_matrix = np.zeros((10, 10), dtype=np.float32)
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        
        for c1, c2 in associations:
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
        """Get the 5 core themes for prompt generation."""
        return ["animal", "machine", "emotion", "nature", "algorithm"]
    
    def generate_prompt_vector(self, theme: str, add_noise: bool = True) -> np.ndarray:
        """
        Generate a prompt vector based on a theme.
        
        Args:
            theme: Core theme for the prompt
            add_noise: Add slight noise for variation
            
        Returns:
            10-dimensional prompt vector
        """
        if theme not in self.network.concept_vectors:
            # If theme not in concepts, use random concept
            theme = self.rng.choice(self.network.concepts)
        
        prompt_vec = self.network.concept_vectors[theme].copy()
        
        if add_noise:
            # Add small noise while maintaining sparsity
            noise = self.rng.normal(0, 0.1, size=10)
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
    
    Implements:
    - One-shot Hebbian learning for association storage
    - Quantum superposition for parallel search
    - Probabilistic output generation
    - Energy-based retrieval dynamics
    """
    
    def __init__(self, num_dimensions: int = 10, temperature: float = 0.5):
        """
        Initialize QAM.
        
        Args:
            num_dimensions: Vector dimensionality (10 for one-hot encoding)
            temperature: Temperature for probabilistic dynamics
        """
        self.num_dimensions = num_dimensions
        self.num_qubits = num_dimensions  # Direct mapping
        self.temperature = temperature
        self.weights = np.zeros((num_dimensions, num_dimensions), dtype=np.float32)
        self.stored_associations = []
        self.semantic_network = None
        self.rng = np.random.RandomState()
    
    def store_semantic_network(self, network: SemanticNetwork):
        """
        Store the semantic network using one-shot Hebbian learning.
        
        Args:
            network: SemanticNetwork object to store
        """
        self.semantic_network = network
        
        # Build weight matrix from associations
        self.weights = network.association_matrix.copy()
        
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
                                  shots: int = 100) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Quantum search for associated concepts.
        
        Args:
            prompt: Input prompt vector
            num_outputs: Number of top outputs to return
            shots: Number of quantum measurements
            
        Returns:
            Tuple of (output vectors, confidence scores, energies)
        """
        if PENNYLANE_AVAILABLE and shots > 10:
            return self._quantum_search(prompt, num_outputs, shots)
        else:
            return self._classical_search(prompt, num_outputs)
    
    def _quantum_search(self, prompt: np.ndarray, num_outputs: int, 
                       shots: int) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        GPU-accelerated quantum circuit for association search.
        
        Args:
            prompt: Input prompt vector
            num_outputs: Number of outputs to generate
            shots: Number of measurements
            
        Returns:
            Tuple of (output vectors, confidence scores, energies)
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
            # Use amplitude encoding for the normalized prompt
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
            
            # Prepare the quantum state using StatePrep (updated from QubitStateVector)
            qml.StatePrep(state_vector, wires=range(self.num_qubits))
            
            # Apply associative memory dynamics via controlled rotations
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    if abs(self.weights[i, j]) > 0.01:
                        # Entangling gates based on association strength
                        qml.CRZ(2.0 * self.weights[i, j] * np.pi, wires=[i, j])
                        qml.CNOT(wires=[i, j])
            
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
            similarity = np.dot(vector, concept_vec) / (np.linalg.norm(vector) * np.linalg.norm(concept_vec))
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = concept
        
        return best_concept

# ============================================================================
# AI SCORING MODULE
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
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
                
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
            # Using the new OpenAI client interface
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
            # Using updated Claude model name
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Updated to available model
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
            # Using synchronous API (not async)
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
    # Use cosine distance
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    distance = 1 - abs(similarity)
    return distance

# ============================================================================
# EMERGENCE DETECTION (updated with better error handling)
# ============================================================================

def detect_creative_emergence(prompt: str, outputs: List[str], 
                            ai_scores: Dict[str, Dict[str, float]],
                            semantic_distances: List[float],
                            network: SemanticNetwork) -> Dict[str, Any]:
    """
    Detect emergent creative behavior.
    
    Args:
        prompt: Input prompt
        outputs: Generated concepts
        ai_scores: Creativity scores from AI judges
        semantic_distances: Distances from prompt
        network: Semantic network for context
        
    Returns:
        Dictionary with emergence assessment
    """
    # Calculate average scores across judges and outputs with better error handling
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
    
    # High creativity threshold (>5.5 on 1-7 scale)
    if overall_creativity > CREATIVITY_THRESHOLD:
        # Check if outputs are not directly encoded
        novel_outputs = 0
        for output in outputs:
            if output not in network.concepts:
                novel_outputs += 1
            elif avg_semantic_distance > SEMANTIC_DISTANCE_THRESHOLD:
                novel_outputs += 0.5
        
        if novel_outputs >= 2:  # At least 2 novel outputs
            is_emergent = True
            emergence_type = "creative_association"
            emergence_score = min(1.0, (overall_creativity - 4) / 3 * 
                                 (avg_semantic_distance / SEMANTIC_DISTANCE_THRESHOLD))
    
    # Alternative emergence: High semantic distance with relevance
    elif avg_semantic_distance > 0.6 and overall_creativity > 4.5:
        is_emergent = True
        emergence_type = "semantic_leap"
        emergence_score = min(1.0, avg_semantic_distance * (overall_creativity / 7))
    
    return {
        "is_emergent": is_emergent,
        "emergence_type": emergence_type,
        "emergence_score": emergence_score,
        "creativity_score": overall_creativity,
        "avg_semantic_distance": avg_semantic_distance
    }

# ============================================================================
# MAIN EXPERIMENTAL PIPELINE (updated with better error handling)
# ============================================================================

async def run_experiments(config: Optional[Dict] = None):
    """
    Execute creative thinking experiments.
    
    Args:
        config: Experimental configuration dictionary
        
    Returns:
        Results and validation metrics
    """
    if config is None:
        config = {
            "num_prompts_per_theme": 3,
            "num_outputs": 3,
            "shots": 100,
            "temperature": 0.5,
            "output_dir": "results",
            "use_ai_scoring": True
        }
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    print("=" * 60)
    print("QUANTUM ASSOCIATIVE MEMORY CREATIVE THINKING EXPERIMENTS")
    print("=" * 60)
    print(f"Temperature: {config.get('temperature', 0.5)}")
    print(f"Quantum noise: {QUANTUM_NOISE_LEVEL}")
    print(f"AI Scoring: {'Enabled' if config['use_ai_scoring'] else 'Mock'}")
    print(f"Seed: {GLOBAL_SEED or 'True random'}")
    print(f".env file support: {'Enabled' if DOTENV_AVAILABLE else 'Disabled'}")
    print()
    
    # Initialize components
    network_gen = SemanticNetworkGenerator(seed=GLOBAL_SEED)
    network = network_gen.network
    
    qam = QuantumAssociativeMemory(
        num_dimensions=10,
        temperature=config.get("temperature", 0.5)
    )
    qam.store_semantic_network(network)
    
    # Initialize AI scorer
    scorer = AICreativityScorer() if config["use_ai_scoring"] else None
    
    # Output setup
    os.makedirs(config["output_dir"], exist_ok=True)
    output_file = os.path.join(config["output_dir"], f"qam_creative_results_{timestamp}.csv")
    
    headers = [
        "run_id", "timestamp", "prompt_theme", "prompt_concept",
        "output_1", "output_2", "output_3",
        "ai_score_novelty_avg", "ai_score_relevance_avg", "ai_score_surprise_avg",
        "semantic_distance_1", "semantic_distance_2", "semantic_distance_3",
        "confidence_1", "confidence_2", "confidence_3",
        "energy_1", "energy_2", "energy_3",
        "prompt_lz_complexity", "prompt_shannon_entropy",
        "output_lz_complexity_avg", "output_shannon_entropy_avg",
        "is_emergent", "emergence_type", "emergence_score",
        "creativity_score", "avg_semantic_distance"
    ]
    
    results = []
    run_id = 0
    
    print("Running experiments...\n")
    
    # Test each theme
    themes = network_gen.get_prompt_themes()
    
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
            
            # Calculate average AI scores with better error handling
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
                emergence["avg_semantic_distance"]
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
    
    total_emergent = sum(1 for r in results if r[23])  # is_emergent column
    print(f"Total runs: {len(results)}")
    print(f"Emergent cases: {total_emergent} ({total_emergent/len(results)*100:.1f}%)")
    
    # Calculate average creativity scores
    avg_creativity = np.mean([r[26] for r in results])  # creativity_score column
    print(f"Average creativity score: {avg_creativity:.2f}/7.0")
    
    # Breakdown by emergence type
    emergence_types = {}
    for r in results:
        if r[23]:  # is_emergent
            etype = r[24]  # emergence_type
            emergence_types[etype] = emergence_types.get(etype, 0) + 1
    
    if emergence_types:
        print("\nEmergence types:")
        for etype, count in emergence_types.items():
            print(f"  {etype}: {count} ({count/total_emergent*100:.1f}% of emergent)")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration for experiments
    config = {
        "num_prompts_per_theme": 5,  # 5 prompts per theme
        "num_outputs": 3,  # Generate 3 associations per prompt
        "shots": 100,  # Quantum measurement shots
        "temperature": 0.5,  # Creativity temperature
        "output_dir": "results",
        "use_ai_scoring": True  # Set to True to use real AI scoring (if keys are configured)
    }
    
    # Run async experiments
    asyncio.run(run_experiments(config))
    
    print("\n✅ Creative thinking experiment complete!")