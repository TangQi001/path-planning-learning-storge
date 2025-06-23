# Placeholder for A* planner module
def find_astar_path():
    pass 

import heapq
import numpy as np
# Assuming GridEnvironment is in a file named environment.py in the same directory or accessible via PYTHONPATH
# from .environment import GridEnvironment # For package structure if run as module
# For direct script running if files are in the same folder:
# from environment import GridEnvironment


class AStarNode:
    """Helper class for A* nodes to store parent and scores."""
    def __init__(self, position, parent=None, g_score=0, h_score=0):
        self.position = tuple(map(int, position)) # Ensure integer coordinates
        self.parent = parent
        self.g_score = g_score # Cost from start to this node
        self.h_score = h_score # Heuristic cost from this node to goal

    def f_score(self):
        return self.g_score + self.h_score

    # For heapq comparison
    # We need a unique tie-breaker if f_scores are equal. Using position as a secondary key.
    def __lt__(self, other):
        if not isinstance(other, AStarNode):
            return NotImplemented
        return (self.f_score(), self.position) < (other.f_score(), other.position)

    def __eq__(self, other):
        if not isinstance(other, AStarNode):
            return False
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

def euclidean_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D points."""
    p1 = np.array(pos1)
    p2 = np.array(pos2)
    return np.linalg.norm(p1 - p2)

def get_neighbors(position, environment):
    """
    Get valid neighbors for a position in the 3D grid (26-connectivity).
    A neighbor is valid if it's within bounds and not an obstacle.
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the current position itself

                neighbor_pos = (position[0] + dx, position[1] + dy, position[2] + dz)
                
                if environment.is_valid_point(*neighbor_pos):
                    neighbors.append(neighbor_pos)
    return neighbors

def reconstruct_path(current_node):
    """Reconstruct path from goal node back to start node using parent pointers."""
    path = []
    while current_node:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1] # Reverse to get path from start to goal

def find_astar_path(environment, start_coords, goal_coords):
    """
    Finds a path from start_coords to goal_coords using A* algorithm.
    Includes post-search validation to ensure path does not go through obstacles.

    Args:
        environment (GridEnvironment): The grid environment.
        start_coords (tuple): (x, y, z) for the start.
        goal_coords (tuple): (x, y, z) for the goal.

    Returns:
        list: A list of (x, y, z) tuples representing the path, or None if no path found.
    """
    start_pos = tuple(map(int, start_coords))
    goal_pos = tuple(map(int, goal_coords))

    if not environment.is_valid_point(*start_pos):
        print(f"A* Error: Start position {start_pos} is invalid (obstacle or out of bounds).")
        return None
    if not environment.is_valid_point(*goal_pos):
        print(f"A* Error: Goal position {goal_pos} is invalid (obstacle or out of bounds).")
        return None
    if start_pos == goal_pos:
        return [start_pos]

    start_node = AStarNode(start_pos, h_score=euclidean_distance(start_pos, goal_pos))
    
    open_set = []  # Priority queue (min-heap)
    heapq.heappush(open_set, start_node) # Stores AStarNode objects directly due to __lt__
    
    # closed_set stores positions that have been fully evaluated.
    closed_set = set() 
    
    # g_scores dictionary: maps position to the lowest g_score found so far to reach it
    g_scores = {start_pos: 0}

    path_found_node = None
    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.position == goal_pos:
            path_found_node = current_node
            break # Path found

        if current_node.position in closed_set:
            continue # Already processed this node via a shorter or equal path

        closed_set.add(current_node.position)

        for neighbor_pos_tuple in get_neighbors(current_node.position, environment):
            neighbor_pos = tuple(map(int, neighbor_pos_tuple))

            # Cost to move from current to neighbor
            move_cost = euclidean_distance(current_node.position, neighbor_pos)
            tentative_g_score = current_node.g_score + move_cost

            if tentative_g_score < g_scores.get(neighbor_pos, float('inf')):
                g_scores[neighbor_pos] = tentative_g_score
                h_score = euclidean_distance(neighbor_pos, goal_pos)
                neighbor_node = AStarNode(neighbor_pos, parent=current_node, g_score=tentative_g_score, h_score=h_score)
                
                # No need to check if neighbor_node is in open_set to update it.
                # heapq allows duplicate items; the one with the lower f_score (due to __lt__) will be popped first.
                heapq.heappush(open_set, neighbor_node)
                
    if path_found_node:
        reconstructed_path = reconstruct_path(path_found_node)
        
        # *** START OF A* PATH VALIDATION ***
        for i, point_in_path in enumerate(reconstructed_path):
            # Start and goal points are already checked for validity by is_valid_point at the beginning.
            # We are interested if any intermediate point is an obstacle.
            # However, is_obstacle also checks bounds, so a point being an obstacle covers both cases.
            if environment.is_obstacle(point_in_path[0], point_in_path[1], point_in_path[2]):
                # Allow start/goal to be on edge cases if they were deemed valid initially,
                # but intermediate points must not be obstacles.
                # This check is somewhat redundant if get_neighbors and is_valid_point work perfectly,
                # but serves as a strong assertion.
                is_start_or_goal = (point_in_path == start_pos) or (point_in_path == goal_pos)
                if not is_start_or_goal: # Only fail for intermediate points
                    print(f"CRITICAL A* ERROR: Path point {point_in_path} at index {i} is an OBSTACLE!")
                    print(f"Obstacles set: {environment.obstacles}")
                    # To help debug, print neighbors of the previous point
                    if i > 0:
                        prev_point = reconstructed_path[i-1]
                        print(f"Neighbors of previous point {prev_point} considered by A*: {get_neighbors(prev_point, environment)}")
                    return None # Path is invalid
        # *** END OF A* PATH VALIDATION ***
        return reconstructed_path
        
    print("A* Info: No path found to goal.")
    return None


if __name__ == '__main__':
    # This requires GridEnvironment to be importable
    GridEnvironment_imported = False
    try:
        # Attempt to import from the current directory structure if run as a script
        from environment import GridEnvironment
        GridEnvironment_imported = True
    except ImportError:
        try:
            # Attempt to import if part of a package uav_planning_simulation
            from uav_planning_simulation.environment import GridEnvironment
            GridEnvironment_imported = True
        except ImportError:
            print("CRITICAL: Could not import GridEnvironment. Ensure 'environment.py' is in the same directory or 'uav_planning_simulation' is in PYTHONPATH.")
            # Define a dummy class to prevent NameError if tests are to be run partially
            class GridEnvironment:
                def __init__(self, size): pass
                def add_obstacle(self, x,y,z): pass
                def add_obstacles(self, l): pass
                def is_valid_point(self,x,y,z): return False # Fail safe
                @property
                def obstacles(self): return set()


    if GridEnvironment_imported:
        print("\nTest Case 1: Simple path")
        env1 = GridEnvironment(size=(5, 5, 5))
        start1 = (0, 0, 0)
        goal1 = (4, 4, 4)
        path1 = find_astar_path(env1, start1, goal1)
        print(f"Path from {start1} to {goal1}: {path1}")
        assert path1 is not None, "Test Case 1 Failed: No path found"
        assert path1[0] == start1, "Test Case 1 Failed: Path doesn't start at start_pos"
        assert path1[-1] == goal1, "Test Case 1 Failed: Path doesn't end at goal_pos"

        print("\nTest Case 2: Path with an obstacle")
        env2 = GridEnvironment(size=(5, 5, 5))
        # Make a wall
        for i in range(5): # Wall along y-axis at x=2, z=2
             env2.add_obstacle(2, i, 2)
        start2 = (0, 2, 2)
        goal2 = (4, 2, 2)
        print(f"Obstacles in env2: {env2.obstacles}")
        path2 = find_astar_path(env2, start2, goal2)
        print(f"Path from {start2} to {goal2} (with wall at x=2, z=2): {path2}")
        assert path2 is not None, "Test Case 2 Failed: No path found"
        # Check that no point in the path (except possibly start/end if they are obstacles - though find_astar_path checks this) is an obstacle
        intermediate_path2 = path2[1:-1] if len(path2) > 1 else path2
        assert all(not env2.is_obstacle(*p) for p in intermediate_path2), "Test Case 2 Failed: Path goes through obstacle"


        print("\nTest Case 3: No path")
        env3 = GridEnvironment(size=(3, 3, 3))
        # Create a complete wall at x=1
        for i in range(3): # y-coord
            for j in range(3): # z-coord
                env3.add_obstacle(1, i, j)
        start3 = (0, 1, 1)
        goal3 = (2, 1, 1)
        print(f"Obstacles in env3: {env3.obstacles}")
        path3 = find_astar_path(env3, start3, goal3)
        print(f"Path from {start3} to {goal3} (should be None): {path3}")
        assert path3 is None, "Test Case 3 Failed: Path found when none should exist"

        print("\nTest Case 4: Start or Goal is obstacle")
        env4 = GridEnvironment(size=(3,3,3))
        env4.add_obstacle(0,0,0)
        env4.add_obstacle(2,2,2)
        path4_1 = find_astar_path(env4, (0,0,0), (1,1,1))
        print(f"Path with start as obstacle: {path4_1}")
        assert path4_1 is None, "Test Case 4.1 Failed"
        path4_2 = find_astar_path(env4, (1,1,1), (2,2,2))
        print(f"Path with goal as obstacle: {path4_2}")
        assert path4_2 is None, "Test Case 4.2 Failed"
        
        print("\nTest Case 5: Start equals Goal")
        env5 = GridEnvironment(size=(3,3,3))
        path5 = find_astar_path(env5, (1,1,1), (1,1,1))
        print(f"Path with start == goal: {path5}")
        assert path5 == [(1,1,1)], "Test Case 5 Failed"

        print("\nAll A* local tests seem to pass based on assertions (if no errors shown above).")
    else:
        print("Skipping A* tests because GridEnvironment could not be imported.") 