import numpy as np
import random
import json

# Function to generate synthetic data for a Quantum Associative Memory Network (QAM) suitable for creative thinking
def sydge_generate_qamnn_data():
    """
    Generates a semantic network for a QAM network by loading data from JSON files.
    """
    # Load data from JSON files
    print("\nLoading QAM data from JSON files...")


    try:
        # 1. Load data from external JSON files
        with open('data_generation_QAMNN_concepts.json', 'r') as f:
            concepts = json.load(f)
        
        with open('data_generation_QAMNN_associations.json', 'r') as f:
            associations = json.load(f)

        with open('data_generation_QAMNN_themes.json', 'r') as f:
            themes = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading QAM data files: {e}")
        return None # Return nothing if files can't be loaded or are invalid

    # --- The rest of the logic remains largely the same ---

    print("Generating synthetic data for Quantum Associative Memory Network (QAM)...")
    
    # 2. Create the concept map
    concept_map = {concept: i for i, concept in enumerate(concepts)}
    num_concepts = len(concepts)

    # 3. Generate binary memory vectors from associations
    memory_vectors = []
    for concept1, concept2 in associations:
        vector = np.zeros(num_concepts, dtype=int)
        if concept1 in concept_map and concept2 in concept_map:
            vector[concept_map[concept1]] = 1
            vector[concept_map[concept2]] = 1
            memory_vectors.append(vector)

    # 4. Generate prompt vectors from themes
    prompts = {}
    for theme_name, related_concepts in themes.items():
        prompt_vector = np.zeros(num_concepts, dtype=int)
        for concept in related_concepts:
            if concept in concept_map:
                prompt_vector[concept_map[concept]] = 1
        prompts[theme_name] = prompt_vector

    print(f"Successfully generated {len(memory_vectors)} memory vectors and {len(prompts)} prompts.\n")

    return {
        "concept_map": concept_map,
        "memory_vectors": memory_vectors,
        "prompts": prompts
    }