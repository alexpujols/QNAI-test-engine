import numpy as np
import random

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
    print(f"\n   - Generating {num_mazes} synthetic {maze_size[0]}x{maze_size[1]} mazes for VQNN...")

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
    print(f"   - Successfully generated {len(mazes_data)} mazes.")
    return mazes_data