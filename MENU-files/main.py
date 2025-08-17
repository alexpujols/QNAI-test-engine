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
__version__ = "1.03-alpha"
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
from utils import pattern_size_check
from utils import clear_screen
from circuits_quantum_cirq import quantum_square_root_not_gate
from circuits_quantum_pennylane import lightning_gpu_test
from data_generation_QHNN import sydge_generate_qhnn_data
from data_generation_VQNN import sydge_generate_vqnn_data
from data_generation_QAMNN import sydge_generate_qamnn_data
from data_generation_QAMNN_wordbank_generator import generate_dataset
from model_QHNN import QuantumHopfieldNetworkPennyLane
from model_VQNN import VariationalQuantumAgentPennyLane
from model_QAMNN import QuantumAssociativeMemoryPennyLane

### Main code begins ###

# Begin main code execution
while True:
    print("\n--------------------------------------------------------------")
    print("-- Which quantum simulation scenario would you like to run? --")
    print("--------------------------------------------------------------")
    print("1 - _Start Scenario_ Pattern Matching")
    print("2 - _Start Scenario_ Problem Solving")
    print("3 - _Start Scenario_ Creative Thinking")
    print("4 - _Test SyDGE_     Synthetic Data Generation Engine")
    print("5 - _Test Circuit_   Sample Quantum Circuit Routine")
    print("0 - EXIT")

    # Take user input and validate
    main_selection = input_int_validate()

    # Take action based on user selection
    if main_selection == 1:
        print("\n--- Scenario: Pattern Matching with QHNN ---")
        # 1. Get parameters and generate data
        size = pattern_size_check()
        num_p = int(input("Enter number of patterns to store: "))
        noise = float(input("Enter noise level for retrieval test (0.0 to 1.0): "))
        
        patterns_data = sydge_generate_qhnn_data(
            pattern_size=(size, size),
            num_patterns=num_p,
            noise_level=noise
        )
        
        # 2. Instantiate and configure the QHNN
        num_neurons = size * size
        qknn_model = QuantumHopfieldNetworkPennyLane(num_neurons=num_neurons)
        
        # Flatten patterns for the model
        flat_fundamental_patterns = [p.flatten() for p in patterns_data["fundamental"]]
        qknn_model.store_patterns(flat_fundamental_patterns)

        # 3. Perform retrieval on the first noisy pattern
        print("\v   - Testing retrieval on the first generated noisy pattern...")
        noisy_pattern_to_test = patterns_data["noisy"][0]
        retrieved_pattern = qknn_model.retrieve(noisy_pattern_to_test)

        # 4. Display results
        print("\n\n--- Retrieval Results ---\n")
        print("Original Fundamental Pattern:\n", patterns_data["fundamental"][0])
        print("\nNoisy Input Pattern:\n", noisy_pattern_to_test)
        print("\nRetrieved Pattern:\n", retrieved_pattern)
        print("\n" + "-" * 25 + "\n")
    elif main_selection == 2:
        print("\n--- Scenario: Problem Solving with VQNN ---")
        # 1. Get parameters and generate a maze
        while True:
            size = pattern_size_check()
            if size % 2 == 1:
                break
        episodes = int(input("Enter number of training episodes: "))
        
        maze_data = sydge_generate_vqnn_data(num_mazes=1, maze_size=(size, size))[0]
        maze = maze_data["maze"]
        start_pos = maze_data["start_pos"]
        goal_pos = maze_data["goal_pos"]
        
        # 2. Instantiate the agent
        agent = VariationalQuantumAgentPennyLane(maze_size=(size, size))

        # 3. Run the training loop
        print("\n   - Starting VQNN agent training...")
        max_steps_per_episode = size * size * 2  # Set a step limit

        for episode in range(episodes):
            agent_pos = start_pos
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                # The 'state' is the full maze with the agent's position marked
                current_state = maze.copy()
                current_state[agent_pos] = 4 # Use '4' to mark agent position
                
                # Agent chooses an action
                action = agent.choose_action(current_state)
                
                # Simple environment logic
                next_pos = list(agent_pos)
                if action == 0: next_pos[0] -= 1 # Up
                elif action == 1: next_pos[0] += 1 # Down
                elif action == 2: next_pos[1] -= 1 # Left
                elif action == 3: next_pos[1] += 1 # Right

                reward = -1 # Cost for each step
                if tuple(next_pos) == goal_pos:
                    reward = 100
                    done = True
                elif not (0 <= next_pos[0] < size and 0 <= next_pos[1] < size and maze[tuple(next_pos)] != 1):
                    reward = -10 # Penalty for hitting a wall
                    next_pos = agent_pos # Stay in place
                    done = False
                else:
                    done = False

                next_state = maze.copy()
                next_state[tuple(next_pos)] = 4
                
                # Agent learns from the experience
                agent.train(current_state, action, reward, next_state)
                
                agent_pos = tuple(next_pos)
                total_reward += reward

                if done:
                    print(f"   - Episode {episode + 1}: Goal reached in {step + 1} steps! Total Reward: {total_reward}")
                    break
            
            if not done:
                print(f"   - Episode {episode + 1}: Max steps reached. Total Reward: {total_reward}")
        print("\n--- Training Complete ---")
    elif main_selection == 3:
        print("\n--- Scenario: Creative Thinking with QAMNN ---")
        # 1. Load the semantic network data
        qam_data = sydge_generate_qamnn_data()
        if not qam_data:
            print("   - Could not load QAM data. Exiting scenario.")
        else:
            # 2. Instantiate and configure the QAM
            num_concepts = len(qam_data["concept_map"])
            qam_model = QuantumAssociativeMemoryPennyLane(num_concepts=num_concepts)
            qam_model.store_memories(qam_data["memory_vectors"])

            # 3. Prompt user to select a theme
            print("\n   - Please select a creative prompt (theme):")
            themes = list(qam_data["prompts"].keys())
            for i, theme in enumerate(themes):
                print(f"{i + 1} - {theme}")
            
            choice = -1
            while choice < 1 or choice > len(themes):
                choice = int(input(f"Enter your choice (1-{len(themes)}): "))
            
            selected_theme = themes[choice - 1]
            prompt_vector = qam_data["prompts"][selected_theme]

            # 4. Query the QAM with the selected prompt
            output_vector = qam_model.query(prompt_vector)

            # 5. Decode and display the results
            index_to_concept = {v: k for k, v in qam_data["concept_map"].items()}
            prompt_concepts = [index_to_concept[i] for i, bit in enumerate(prompt_vector) if bit == 1]
            output_concepts = [index_to_concept[i] for i, bit in enumerate(output_vector) if bit == 1]
            
            print("\n--- Creative Output ---")
            print(f"Prompt Concepts: {' + '.join(prompt_concepts)}")
            print(f"Generated Output Concepts: {' & '.join(output_concepts)}")
            print("-" * 25 + "\n")
    elif main_selection == 4:
        print("\n   - You selected Sythetic Data Generation Engine (SyDGE)!")
        while True:
            print("\nHi, which randomized sythetic data sample would you like to generate?")
            print("1 - _Test Data_ QHNN Data for Pattern Matching")
            print("2 - _Test Data_ VQNN Data for Problem Solving")
            print("3 - _Test Data_ QAM Data for Creative Thinking")
            print("0 - EXIT")
            
            # Take user input and validate
            sydge_selection = input_int_validate()

            # Take action based on user selection
            if sydge_selection == 1:
                print("\n   - You selected to generate a sample synthetic dataset for a Quantum Hopfield Neural Network (QHNN) suitable for pattern matching.")
                pattern_size_input = int(input("Enter one value to set both the number of rows and columns (e.g., \"5\" for a 5x5 pattern) : "))
                num_patterns_input = int(input("How many patterns would you like to generate? : "))
                noise_level_input = float(input("What noise level would you like to apply to the patterns? (0.0 - 1.0) : "))
                incompleteness_level_input = float(input("What incompleteness level would you like to apply to the patterns? (0.0 - 1.0) : "))
                
                # Generate synthetic data for QHNN
                clear_screen()
                patterns_data = sydge_generate_qhnn_data(
                    pattern_size=(pattern_size_input, pattern_size_input),
                    num_patterns=num_patterns_input,
                    noise_level=noise_level_input,
                    incompleteness_level=incompleteness_level_input
                )
                
                # Print the generated patterns
                for i in range(len(patterns_data["fundamental"])):
                    print(f"\n--- Pattern {i+1} ---")
                    print("\nFundamental Pattern:\n", patterns_data["fundamental"][i])
                    print("\nNoisy Pattern (" + str(noise_level_input * 100) + "%):\n", patterns_data["noisy"][i])
                    print("\nIncomplete Pattern (" + str(incompleteness_level_input * 100) + "%):\n", patterns_data["incomplete"][i])
                    print("\n" + "-" * 17)
            elif sydge_selection == 2:           
                # Call the function to generate maze data for the VQNN
                # Take action based on user selection
                print("\n   - You selected to generate a sample synthetic dataset for a Variational Quantum Neural Network (VQNN) suitable for problem solving.\n")
                num_mazes_input = int(input("How many mazes would you like to generate? : "))
                while True:
                    maze_size_input = int(input("Enter one value (odd number) to set both the number of rows and columns (e.g., \"5\" for a 5x5 maze) : "))
                    if maze_size_input % 2 == 1:
                        break
                    else:
                        print("   - Please enter an odd number for the maze size to ensure a valid maze structure.")

                # Generate synthetic data for VQNN
                clear_screen()
                vqnn_mazes = sydge_generate_vqnn_data(
                    num_mazes=num_mazes_input,
                    maze_size=(maze_size_input, maze_size_input)
                )

                # Print maze data
                if vqnn_mazes:
                    print("\n--- Generated Maze Details ---")
                    # Loop through the returned list and print the data for each maze
                    for i, maze_data in enumerate(vqnn_mazes):
                        print(f"\n--- Maze {i + 1} ---\n")
                        print("Layout (0:path, 1:wall, 2:start, 3:goal):\n")
                        print(maze_data["maze"])
                        print(f"\n   - Start Position: ({maze_data['start_pos'][1]}, {maze_data['start_pos'][0]})")
                        print(f"   - Goal Position: ({maze_data['goal_pos'][1]}, {maze_data['goal_pos'][0]})")
                        print(f"   - Complexity: {maze_data['complexity']}")
                    print("\n" + "-" * 14 + "\n")
                else:
                    print("   - No maze data was generated.")
            elif sydge_selection == 3:
                print("\n   - You selected to generate a sample synthetic dataset for a Quantum Associative Memory Network (QAM) suitable for creative thinking.\n")
                
                # Ask the user if they want to generate a new dataset or use existing files
                while True:
                    gen_new_data = str(input("Would you like to generate a brand new data set (Y/N)? (If 'N,' exiting JSON data is used) : "))
                    if gen_new_data.lower() == 'y':
                        # Call the function to generate a new dataset
                        generate_dataset()
                        break
                    elif gen_new_data.lower() == 'n':
                        print("\n   - Using existing dataset files for QAM data generation...")
                        break
                    else:
                        print("   - Invalid input. Please select Y or N")

                # Call the function to generate QAM data
                clear_screen()
                qam_data = sydge_generate_qamnn_data()

                # Print the generated data in a readable format
                if qam_data:
                    # Create a reverse map for easy lookup from index to concept name
                    index_to_concept = {v: k for k, v in qam_data["concept_map"].items()}

                    print("--- Generated QAM Semantic Network ---")
                    print("\nCore Concepts Map (Concept: Vector Index):")
                    print(qam_data["concept_map"])

                    print("\n--- Stored Memory Associations (Binary Vectors): ---")
                    for i, vector in enumerate(qam_data["memory_vectors"]):
                        # Find the concepts that are 'on' in the vector
                        active_concepts = [index_to_concept[idx] for idx, val in enumerate(vector) if val == 1]
                        print(f"\n  Memory {i+1:>2}: {vector} -> ({' & '.join(active_concepts)})")

                    print("\n--- Creative Prompts (Binary Vectors): ---")
                    for theme, vector in qam_data["prompts"].items():
                        active_concepts = [index_to_concept[idx] for idx, val in enumerate(vector) if val == 1]
                        print(f"\n  Theme '{theme}': {vector} -> ({' + '.join(active_concepts)})")

                    print("\n" + "-" * 40)
                else:
                    print("   - No QAM data was generated.")
            elif sydge_selection == 0:
                print("\n   - You have chosen to leave the SyDGE program. Goodbye!\n")
                break
    elif main_selection == 5:
        print("\n   - You selected a Sample Quantum Test Circuit")
        # Prompt the user to select a quantum simulation test
        # This will loop until the user selects a valid option or exits
        while True:
            print("\nHi, which quantum simulation test would you like to run?")
            print("1 - _Simulation_ Quantum square root NOT gate (Hadamard gate) using Cirq")
            print("2 - _Simulation_ Quantum Hadamard gate to Bell state using Pennylane")
            print("0 - EXIT")
            # Take user input and validate
            select_quantum_test = input_int_validate()

            # Take action based on user selection
            if select_quantum_test == 1:
                # Call the function to run the quantum square root NOT gate simulation
                quantum_square_root_not_gate()
            elif select_quantum_test == 2:
                # Call the function to run the Pennylane quantum simulation
                lightning_gpu_test()
            elif select_quantum_test == 0:
                break
            else:
                print("   - Invalid input. Please select 1 or 2 to run a quantum simulation test, or 0 to exit.")
    elif main_selection == 0:
        print("\n   - You have chosen to leave the program. Goodbye!\n")
        break