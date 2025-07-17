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

# Function to generate synthetic data for a Variational Quantum Neural Network (VQNN) suitable for problem solving
def sydge_generate_vqnn_data(num_mazes=10, maze_size=(5, 5)):
    """
    Generates environments for a Variational Quantum Neural Network (VQNN)
    for adaptive problem-solving scenarios.

    This function procedurally generates a specified number of unique grid mazes
    with varying complexities, including distinct start and goal positions,
    dead ends, and branching factors. The mazes are programmatically
    represented as NumPy arrays, suitable for subsequent amplitude encoding
    for VQNNs.

    Args:
        num_mazes (int): The number of unique mazes to generate. Defaults to 10.
        maze_size (tuple): The dimensions (height, width) of the mazes.
                           Both dimensions must be odd. Defaults to (5, 5).

    Returns:
        list: A list of dictionaries, where each dictionary represents a maze
              and contains:
              'maze': A 2D NumPy array representing the maze layout
                      (0: path, 1: wall, 2: start, 3: goal).
              'start_pos': A tuple (row, col) for the start position.
              'goal_pos': A tuple (row, col) for the goal position.
              'complexity': A dictionary with 'dead_ends' (int) and
                            'branching_factor' (float).
    """
    print(f"\nGenerating {num_mazes} synthetic {maze_size[0]}x{maze_size[1]} mazes for VQNN...")
    # This algorithm works best with odd dimensions to create a clear grid of paths and walls
    if maze_size[0] % 2 == 0 or maze_size[1] % 2 == 0:
        raise ValueError("Maze dimensions must be odd integers to ensure proper grid generation.")

    mazes_data = []

    for i in range(num_mazes):
        # Initialize grid with walls (1)
        maze = np.ones(maze_size, dtype=int)
        
        # Use randomized Depth-First Search (DFS) to carve paths
        # Start at a random "cell" (must have odd coordinates)
        start_node = (random.randrange(0, maze_size[0], 2), random.randrange(0, maze_size[1], 2))
        stack = [start_node]
        maze[start_node] = 0

        while stack:
            current_r, current_c = stack[-1]
            neighbors = []

            # Find unvisited neighbors 2 cells away
            for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nr, nc = current_r + dr, current_c + dc
                if 0 <= nr < maze_size[0] and 0 <= nc < maze_size[1] and maze[nr, nc] == 1:
                    neighbors.append((nr, nc))
            
            if neighbors:
                next_r, next_c = random.choice(neighbors)
                
                # Carve path to the neighbor by removing the wall in between
                wall_r, wall_c = current_r + (next_r - current_r) // 2, current_c + (next_c - current_c) // 2
                maze[wall_r, wall_c] = 0
                maze[next_r, next_c] = 0
                
                stack.append((next_r, next_c))
            else:
                # No unvisited neighbors, backtrack
                stack.pop()

        # --- Calculate complexity metrics (dead ends and branching factor) ---
        path_cells = np.argwhere(maze == 0)
        dead_ends = 0
        junctions = 0
        total_choices_at_junctions = 0

        for r, c in path_cells:
            path_neighbors = 0
            # Check cardinal neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze_size[0] and 0 <= nc < maze_size[1] and maze[nr, nc] != 1: # Path, start, or goal
                    path_neighbors += 1
            
            if path_neighbors == 1:
                dead_ends += 1
            elif path_neighbors > 2:
                junctions += 1
                total_choices_at_junctions += path_neighbors
        
        branching_factor = (total_choices_at_junctions / junctions) if junctions > 0 else 0.0

        # --- Select distinct start (2) and goal (3) positions ---
        path_indices = np.where(maze == 0)
        num_path_cells = len(path_indices[0])
        # Ensure there are at least two path cells to pick from
        if num_path_cells < 2:
            continue # Skip this maze if it's too small to have a start and goal
            
        start_idx, goal_idx = random.sample(range(num_path_cells), 2)
        
        start_pos = (int(path_indices[0][start_idx]), int(path_indices[1][start_idx]))
        goal_pos = (int(path_indices[0][goal_idx]), int(path_indices[1][goal_idx]))
        
        maze[start_pos] = 2
        maze[goal_pos] = 3

        mazes_data.append({
            "maze": maze,
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "complexity": {
                "dead_ends": dead_ends,
                "branching_factor": round(branching_factor, 2)
            }
        })
    print(f"Successfully generated {len(mazes_data)} mazes.")
    return mazes_data

# Function to generate synthetic data for a Quantum Associative Memory Network (QAM) suitable for creative thinking
def sydge_generate_qamnn_data():
    print("\nGenerating synthetic data for Quantum Associative Memory Network (QAM) suitable for creative thinking...\n")
    # TBD: Add implementation for QAM data generation
    pass

