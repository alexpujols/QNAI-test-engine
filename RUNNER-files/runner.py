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
Title         : {Quantum Neuromorphic Artificial Intelligence Test Script}
Date          : {05-18-2025}
Description   : {A Python/PennyLane test engine that simulates multiple quantum and classical neural networks for testing/simulation purposes}
Options       : {Options located in config.json configurationf file}
Dependencies  : {PennyLane, NumPy, TensorFlow, SciPy, Matplotlib, Pandas, etc.}
Requirements  : {Python 3.8+}
Usage         : {python runner.py}
Notes         : {Available at Github at https://github.com/alexpujols/QNAI-test-engine}
'''

import json
import csv
import itertools
import os
import time
from datetime import datetime

# Import your existing modules
from data_generation_QHNN import sydge_generate_qhnn_data
from data_generation_VQNN import sydge_generate_vqnn_data
from data_generation_QAMNN import sydge_generate_qamnn_data
from model_QHNN import QuantumHopfieldNetworkPennyLane
from model_VQNN import VariationalQuantumAgentPennyLane
from model_QAMNN import QuantumAssociativeMemoryPennyLane

def setup_csv(filepath, headers):
    """Creates a new CSV file with headers if it doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def run_pattern_matching_experiments(config):
    """Runs all configured pattern matching simulations."""
    print("--- Starting Pattern Matching (QHNN) Experiments ---")
    filepath = config["output_csv"]
    headers = [
        "run_id", "timestamp", "pattern_size", "num_patterns_stored", 
        "noise_level", "prediction_accuracy", "raw_output_bitstring"
    ]
    setup_csv(filepath, headers)

    param_combinations = list(itertools.product(
        config["pattern_sizes"],
        config["num_patterns_to_store"],
        config["noise_levels"]
    ))

    for i, (size, num_p, noise) in enumerate(param_combinations):
        run_id = f"QHNN_{i+1}"
        print(f"Running {run_id}: size={size}x{size}, patterns={num_p}, noise={noise}")

        # 1. Generate data for this specific run
        patterns_data = sydge_generate_qhnn_data(
            pattern_size=(size, size),
            num_patterns=num_p,
            noise_level=noise
        )
        
        # 2. Instantiate and run the model
        qknn_model = QuantumHopfieldNetworkPennyLane(num_neurons=size*size)
        flat_patterns = [p.flatten() for p in patterns_data["fundamental"]]
        qknn_model.store_patterns(flat_patterns)
        
        # Test retrieval on the first noisy pattern
        noisy_pattern = patterns_data["noisy"][0]
        original_pattern = patterns_data["fundamental"][0]
        
        results = qknn_model.retrieve(noisy_pattern, original_pattern)

        # 3. Log results to CSV
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id, datetime.now().isoformat(), size, num_p, noise,
                results["prediction_accuracy"], ''.join(map(str, results["raw_output_bitstring"]))
            ])
    print("--- QHNN Experiments Complete ---\n")

def run_problem_solving_experiments(config):
    """Runs all configured problem solving simulations."""
    print("--- Starting Problem Solving (VQNN) Experiments ---")
    filepath = config["output_csv"]
    headers = [
        "run_id", "timestamp", "maze_size", "episodes", "steps_to_goal",
        "time_to_goal", "final_reward", "solution_path"
    ]
    setup_csv(filepath, headers)

    param_combinations = list(itertools.product(
        config["maze_sizes"],
        config["episodes"]
    ))

    for i, (size, episodes) in enumerate(param_combinations):
        run_id = f"VQNN_{i+1}"
        print(f"Running {run_id}: maze_size={size}x{size}, episodes={episodes}")
        
        # 1. Generate a maze
        maze_data = sydge_generate_vqnn_data(num_mazes=1, maze_size=(size, size))[0]
        
        # 2. Instantiate and run the agent
        agent = VariationalQuantumAgentPennyLane(maze_size=(size, size))
        results = agent.train_and_evaluate(
            maze=maze_data["maze"],
            start_pos=maze_data["start_pos"],
            goal_pos=maze_data["goal_pos"],
            episodes=episodes
        )
        
        # 3. Log results to CSV
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id, datetime.now().isoformat(), size, episodes,
                results["steps_to_goal"], results["time_to_goal_s"],
                results["final_reward"], str(results["solution_path"])
            ])
    print("--- VQNN Experiments Complete ---\n")

def run_creative_thinking_experiments(config):
    """Runs all configured creative thinking simulations."""
    print("--- Starting Creative Thinking (QAM) Experiments ---")
    filepath = config["output_csv"]
    headers = ["run_id", "timestamp", "prompt_theme", "prompt_concepts", "output_concepts", "raw_output_vector"]
    setup_csv(filepath, headers)

    # 1. Load the semantic network once
    qam_data = sydge_generate_qamnn_data()
    if not qam_data:
        print("Could not load QAM data. Aborting creative thinking experiments.")
        return
        
    num_concepts = len(qam_data["concept_map"])
    qam_model = QuantumAssociativeMemoryPennyLane(num_concepts=num_concepts)
    qam_model.store_memories(qam_data["memory_vectors"])

    for i in range(config["num_runs"]):
        run_id = f"QAM_{i+1}"
        
        # 2. Randomly select a prompt for each run
        selected_theme = random.choice(list(qam_data["prompts"].keys()))
        prompt_vector = qam_data["prompts"][selected_theme]
        
        print(f"Running {run_id}: theme='{selected_theme}'")
        
        # 3. Query the model
        results = qam_model.query(prompt_vector, qam_data["concept_map"])
        
        # 4. Log results
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id, datetime.now().isoformat(), selected_theme,
                ' + '.join(results["prompt_concepts"]),
                ' & '.join(results["output_concepts"]),
                ''.join(map(str, results["raw_output_vector"]))
            ])
    print("--- QAM Experiments Complete ---\n")


if __name__ == "__main__":
    print("==============================================")
    print("==    QNAI Test Engine Automated Runner     ==")
    print("==============================================")
    
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create it before running.")
        exit()

    if config["pattern_matching"]["enabled"]:
        run_pattern_matching_experiments(config["pattern_matching"])
        
    if config["problem_solving"]["enabled"]:
        run_problem_solving_experiments(config["problem_solving"])
        
    if config["creative_thinking"]["enabled"]:
        run_creative_thinking_experiments(config["creative_thinking"])
        
    print("All configured experiments have been completed.")