import numpy as np
import random

# Function to generate synthetic data for a Quantum Hopfield Neural Network (QHNN) suitable for pattern matching
def sydge_generate_qhnn_data(
    pattern_size=(5, 5),
    num_patterns=3,
    noise_level=0.1,
    incompleteness_level=0.2
):
    """
    Generates pattern data for a Quantum Hopfield Network (QHN).

    This function creates a set of fundamental binary patterns and their
    corresponding noisy and incomplete versions for QHN-based associative memory tasks.

    Args:
        pattern_size (tuple): The dimensions (height, width) of the 2D patterns.
        num_patterns (int): The number of fundamental patterns to generate.
        noise_level (float): The fraction of bits to flip to create noisy patterns.
        incompleteness_level (float): The fraction of bits to mask (set to 0) to
                                     create incomplete patterns.

    Returns:
        dict: A dictionary containing three lists of numpy arrays:
              'fundamental': The original, ideal patterns.
              'noisy': The noisy versions of the fundamental patterns.
              'incomplete': The incomplete versions of the fundamental patterns.
    """
    print("Generating synthetic data for Quantum Hopfield Neural Network (QHNN) suitable for pattern matching...")
    if not (0 <= noise_level <= 1 and 0 <= incompleteness_level <= 1):
        raise ValueError("Noise and incompleteness levels must be between 0 and 1.")

    # Generate fundamental patterns
    num_neurons = pattern_size[0] * pattern_size[1]
    fundamental_patterns = []
    noisy_patterns = []
    incomplete_patterns = []

    for i in range(num_patterns):
        # Generate a unique fundamental pattern
        pattern = np.random.choice([-1, 1], size=num_neurons)
        fundamental_patterns.append(pattern.reshape(pattern_size))

        # --- Generate Noisy Version ---
        noisy_pattern = pattern.copy()
        num_flips = int(noise_level * num_neurons)
        flip_indices = np.random.choice(num_neurons, num_flips, replace=False)
        noisy_pattern[flip_indices] *= -1
        noisy_patterns.append(noisy_pattern.reshape(pattern_size))

        # --- Generate Incomplete Version ---
        incomplete_pattern = pattern.copy()
        num_masked = int(incompleteness_level * num_neurons)
        mask_indices = np.random.choice(num_neurons, num_masked, replace=False)
        incomplete_pattern[mask_indices] = 0  # Using 0 to represent a masked/unknown state
        incomplete_patterns.append(incomplete_pattern.reshape(pattern_size))

    # Return the generated patterns as a dictionary
    return {
        "fundamental": fundamental_patterns,
        "noisy": noisy_patterns,
        "incomplete": incomplete_patterns
    }