##!/usr/bin/env python
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
__version__ = "1.0"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title	      :	{Quantum Neuromorphic Artificial Intelligence Test Engine}
Date		  :	{05-18-2025}
Description   :	{A Python/Cirq test engine that simulates multiple quantum and classical neural networks for testing/simulation purposes}
Options	      :	{TBD}
Notes	      :	{Available at Github at https://github.com/alexpujols/QNAI-test-engine}
'''

## Import modules declarations
import cirq
import numpy as np

## Class declarations

# Class for TBD

# Class for TBD

# Class for TBD


## Function declarations

# Function to test for valid input and convert to int for further processing
def input_int_validate():
    while True:
        try:
            validate = int(input(": "))
            break
        except:
            print ("\nIncorrect value! Please make a new selection\n")
    return validate
# Function to create a sample quantum circuit that applies a square root of NOT gate to a qubit and measures it
def square_root_not_gate():
    # Pick a qubit.
    qubit = cirq.GridQubit(0, 0)
    
    # Create a circuit that applies a square root of NOT gate, then measures the qubit.
    circuit = cirq.Circuit(cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m'))
    print("\nCircuit:")
    print(circuit)
   
    # Simulate the circuit several times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)
    print("\nResults:")
    print(result)
# Function to generate synthetic data for a Quantum Hopfield Neural Network (QHNN) suitable for pattern matching
def sydge_generate_qhn_data(
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
# Function to generate synthetic data for a Variational Quantum Neural Network (VQNN) suitable for problem solving
def sydge_generate_vqn_data():
    print("Generating synthetic data for Variational Quantum Neural Network (VQNN) suitable for problem solving...")
# Function to generate synthetic data for a Quantum Associative Memory Network (QAM) suitable for creative thinking
def sydge_generate_qam_data():
    print("Generating synthetic data for Quantum Associative Memory Network (QAM) suitable for creative thinking...")
# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

# Function for TBD

### Main code begins ###

# Set global vaiables

# Begin main code execution
while True:

    print ("Hi, which quantum simulation scenario would you like to run?")
    print ("1 - _Start Scenario_ Pattern Matching")
    print ("2 - _Start Scenario_ Problem Solving")
    print ("3 - _Start Scenario_ Creative Thinking")
    print ("4 - _Test Data_      Sythetic Data Generation Engine (SyDGE)")
    print ("5 - _Test Circuit_   Sample Quantum Circuit Routine")
    print ("0 - EXIT")

    # Take user input and validate
    main_selection = input_int_validate()

    # Take action based on user selection
    if (main_selection == 1):
        print ("\nYou selected Pattern Matching!\n")
    if (main_selection == 2):
        print ("\nYou selected Problem Solving!\n")
    if (main_selection == 3):
        print ("\nYou selected Creative Thinking!\n")
    if (main_selection == 4):
        print ("\nYou selected Sythetic Data Generation Engine (SyDGE)!\n")
        while True:
            #print ("\n")
            print ("Hi, which randomized sythetic data sample would you like to generate?")
            print ("1 - _Test Data_ QHNN Data for Pattern Matching")
            print ("2 - _Test Data_ VQNN Data for Problem Solving")
            print ("3 - _Test Data_ QAM Data for Creative Thinking")
            print ("0 - EXIT")
            # Take user input and validate
            sydge_selection = input_int_validate()

            # Take action based on user selection
            if (sydge_selection == 1):
                print ("\nYou selected to generate a sample synthetic dataset for a Quantum Hopfield Neural Network (QHNN) suitable for pattern matching.")
                pattern_size_input = int(input("Enter one value to set both the number of rows and columns (e.g., \"5\" for a 5x5 pattern) : "))
                num_patterns_input = int(input("How many patterns would you like to generate? : "))
                noise_level_input = float(input("What noise level would you like to apply to the patterns? (0.0 - 1.0) : "))
                incompleteness_level_input = float(input("What incompleteness level would you like to apply to the patterns? (0.0 - 1.0) : "))
                # Generate synthetic data for QHNN
                patterns_data = sydge_generate_qhn_data(
                    pattern_size=(pattern_size_input, pattern_size_input),
                    num_patterns=num_patterns_input,
                    noise_level=noise_level_input,   # % of the bits will be flipped to create noisy patterns
                    incompleteness_level=incompleteness_level_input    # % of the bits will be masked (set to 0) to create incomplete patterns
                )
                # Print the generated patterns
                for i in range(len(patterns_data["fundamental"])):
                    print(f"\n\n--- Pattern {i+1} ---")
                    print("\nFundamental Pattern:\n", patterns_data["fundamental"][i])
                    #print("\nNoisy Pattern (" + noise_level_input * 10 + "%):\n", patterns_data["noisy"][i])
                    print("\nNoisy Pattern (" + str(noise_level_input * 100) + "%):\n", patterns_data["noisy"][i])
                    #print("\nIncomplete Pattern (30% masked):\n", patterns_data["incomplete"][i])
                    print("\nIncomplete Pattern (" + str(incompleteness_level_input * 100) + "%):\n", patterns_data["incomplete"][i])
                    print("-" * 25 + "\n")            
            if (sydge_selection == 2):
                print ("\nYou selected to generate a sample synthetic dataset for a Variational Quantum Neural Network (VQNN) suitable for problem solving.\n")
            if (sydge_selection == 3):
                print ("\nYou selected to generate a sample synthetic dataset for a Quantum Associative Memory Network (QAM) suitable for creative thinking.\n")
            if (sydge_selection == 0):
                print ("\n You have chosen to leave the program.  Goodbye!\n")
                break
    if (main_selection == 5):
        print ("\nYou selected a Sample Quantum Test Circuit.\n")
        qubit = square_root_not_gate()
    if (main_selection == 0):
        print ("\nYou have chosen to leave the program.  Goodbye!\n")
        break
## Main code ends ###
