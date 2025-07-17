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
__version__ = "1.0"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Prototype"

'''
Title         : {Quantum Neuromorphic Artificial Intelligence Test Engine}
Date          : {05-18-2025}
Description   : {A Python/Cirq test engine that simulates multiple quantum and classical neural networks for testing/simulation purposes}
Options       : {TBD}
Notes         : {Available at Github at https://github.com/alexpujols/QNAI-test-engine}
'''

# Import modules from local files
from utils import input_int_validate
from circuits_quantum import quantum_square_root_not_gate
from data_generation import sydge_generate_qhnn_data, sydge_generate_vqnn_data, sydge_generate_qamnn_data

### Main code begins ###

# Begin main code execution
while True:
    print("\nHi, which quantum simulation scenario would you like to run?")
    print("1 - _Start Scenario_ Pattern Matching")
    print("2 - _Start Scenario_ Problem Solving")
    print("3 - _Start Scenario_ Creative Thinking")
    print("4 - _Test Data_      Sythetic Data Generation Engine (SyDGE)")
    print("5 - _Test Circuit_   Sample Quantum Circuit Routine")
    print("0 - EXIT")

    # Take user input and validate
    main_selection = input_int_validate()

    # Take action based on user selection
    if main_selection == 1:
        print("\nYou selected Pattern Matching!\n")
    elif main_selection == 2:
        print("\nYou selected Problem Solving!\n")
    elif main_selection == 3:
        print("\nYou selected Creative Thinking!\n")
    elif main_selection == 4:
        print("\nYou selected Sythetic Data Generation Engine (SyDGE)!\n")
        while True:
            print("Hi, which randomized sythetic data sample would you like to generate?")
            print("1 - _Test Data_ QHNN Data for Pattern Matching")
            print("2 - _Test Data_ VQNN Data for Problem Solving")
            print("3 - _Test Data_ QAM Data for Creative Thinking")
            print("0 - EXIT")
            # Take user input and validate
            sydge_selection = input_int_validate()

            # Take action based on user selection
            if sydge_selection == 1:
                print("\nYou selected to generate a sample synthetic dataset for a Quantum Hopfield Neural Network (QHNN) suitable for pattern matching.")
                pattern_size_input = int(input("Enter one value to set both the number of rows and columns (e.g., \"5\" for a 5x5 pattern) : "))
                num_patterns_input = int(input("How many patterns would you like to generate? : "))
                noise_level_input = float(input("What noise level would you like to apply to the patterns? (0.0 - 1.0) : "))
                incompleteness_level_input = float(input("What incompleteness level would you like to apply to the patterns? (0.0 - 1.0) : "))
                # Generate synthetic data for QHNN
                patterns_data = sydge_generate_qhnn_data(
                    pattern_size=(pattern_size_input, pattern_size_input),
                    num_patterns=num_patterns_input,
                    noise_level=noise_level_input,
                    incompleteness_level=incompleteness_level_input
                )
                # Print the generated patterns
                for i in range(len(patterns_data["fundamental"])):
                    print(f"\n\n--- Pattern {i+1} ---")
                    print("\nFundamental Pattern:\n", patterns_data["fundamental"][i])
                    print("\nNoisy Pattern (" + str(noise_level_input * 100) + "%):\n", patterns_data["noisy"][i])
                    print("\nIncomplete Pattern (" + str(incompleteness_level_input * 100) + "%):\n", patterns_data["incomplete"][i])
                    print("-" * 25 + "\n")
            elif sydge_selection == 2:
                print("\nYou selected to generate a sample synthetic dataset for a Variational Quantum Neural Network (VQNN) suitable for problem solving.\n")
                # Call the function to generate maze data for the VQNN
                # This will use the default arguments: num_mazes=10, maze_size=(5, 5)
                vqnn_mazes = sydge_generate_vqnn_data()

                # Check if any mazes were generated
                if vqnn_mazes:
                    print("\n--- Generated Maze Details ---")
                    # Loop through the returned list and print the data for each maze
                    for i, maze_data in enumerate(vqnn_mazes):
                        print(f"\n--- Maze {i + 1} ---")
                        print("Layout (0:path, 1:wall, 2:start, 3:goal):")
                        print(maze_data["maze"])
                        print(f"Start Position: {maze_data['start_pos']}")
                        print(f"Goal Position: {maze_data['goal_pos']}")
                        print(f"Complexity: {maze_data['complexity']}")
                    print("\n" + "-" * 28 + "\n")
                else:
                    print("No maze data was generated.")
            elif sydge_selection == 3:
                print("\nYou selected to generate a sample synthetic dataset for a Quantum Associative Memory Network (QAM) suitable for creative thinking.\n")
                sydge_generate_qamnn_data()
            elif sydge_selection == 0:
                print("\n You have chosen to leave the SyDGE program. Goodbye!\n")
                break
    elif main_selection == 5:
        print("\nYou selected a Sample Quantum Test Circuit.\n")
        quantum_square_root_not_gate()
    elif main_selection == 0:
        print("\nYou have chosen to leave the program. Goodbye!\n")
        break