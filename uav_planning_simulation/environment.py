# Placeholder for environment module
import numpy as np

class GridEnvironment:
    def __init__(self, size=(9, 9, 9)):
        """
        Initializes the 3D grid environment.

        Args:
            size (tuple): A tuple of 3 integers (width, height, depth) for the grid.
        """
        if not (isinstance(size, tuple) and len(size) == 3 and all(isinstance(s, int) and s > 0 for s in size)):
            raise ValueError("Size must be a tuple of 3 positive integers.")
        
        self.size = size
        self.width, self.height, self.depth = size
        # Grid: 0 for free space, 1 for obstacle
        self.grid = np.zeros(size, dtype=np.uint8)
        self.obstacles = set() # Store obstacle coordinates for quick lookup

    def add_obstacle(self, x, y, z):
        """Adds a single obstacle at the given coordinates."""
        if self.is_within_bounds(x, y, z):
            self.grid[int(x), int(y), int(z)] = 1
            self.obstacles.add((int(x), int(y), int(z)))
        else:
            print(f"Warning: Obstacle ({x},{y},{z}) is out of bounds and was not added.")

    def add_obstacles(self, obstacles_list):
        """
        Adds multiple obstacles from a list of coordinates.

        Args:
            obstacles_list (list): A list of tuples, where each tuple is (x, y, z) of an obstacle.
        """
        for obs_x, obs_y, obs_z in obstacles_list:
            self.add_obstacle(obs_x, obs_y, obs_z)

    def is_obstacle(self, x, y, z):
        """Checks if the given coordinates (x, y, z) correspond to an obstacle."""
        # Ensure coordinates are integers for grid indexing and set lookup
        coords = (int(round(x)), int(round(y)), int(round(z)))
        if not self.is_within_bounds(*coords):
            return True # Out of bounds is considered an obstacle for path planning
        return coords in self.obstacles
        # Alternative using grid:
        # return self.grid[coords[0], coords[1], coords[2]] == 1


    def is_within_bounds(self, x, y, z):
        """Checks if the given coordinates are within the grid boundaries."""
        # Allow for floating point inputs that might be slightly outside due to precision,
        # but for indexing, they must be convertible to valid integer indices.
        # For strict check, use int(x), int(y), int(z)
        px, py, pz = int(round(x)), int(round(y)), int(round(z))
        if not (0 <= px < self.width and \
                0 <= py < self.height and \
                0 <= pz < self.depth):
            return False
        return True

    def is_valid_point(self, x, y, z):
        """Checks if a point is within bounds and not an obstacle."""
        return self.is_within_bounds(x, y, z) and not self.is_obstacle(x, y, z)

    def get_grid_for_visualization(self):
        """Returns the grid (useful for plotting)."""
        return self.grid

    def __str__(self):
        return f"GridEnvironment(size={self.size}, num_obstacles={len(self.obstacles)})"

if __name__ == '__main__':
    # Example Usage
    env = GridEnvironment(size=(5, 5, 5))
    print(env)

    # Add some obstacles
    env.add_obstacle(1, 1, 1)
    env.add_obstacles([(2, 2, 2), (3, 3, 3)])
    print(f"Obstacles at: {env.obstacles}")

    print(f"Is (1,1,1) an obstacle? {env.is_obstacle(1, 1, 1)}")
    print(f"Is (0,0,0) an obstacle? {env.is_obstacle(0, 0, 0)}")
    print(f"Is (1.1, 1.1, 1.1) an obstacle (rounded)? {env.is_obstacle(1.1, 1.1, 1.1)}")


    print(f"Is (4,4,4) within bounds? {env.is_within_bounds(4, 4, 4)}")
    print(f"Is (5,5,5) within bounds? {env.is_within_bounds(5, 5, 5)}") # Should be False (0-4)
    print(f"Is (-1,0,0) within bounds? {env.is_within_bounds(-1, 0, 0)}")

    print(f"Is (0,0,0) a valid point? {env.is_valid_point(0,0,0)}")
    print(f"Is (1,1,1) a valid point? {env.is_valid_point(1,1,1)}")
    print(f"Is (5,5,5) a valid point? {env.is_valid_point(5,5,5)}")

    # Test floating point coordinates that are close to obstacles
    env.add_obstacle(2,3,4)
    print(f"Is (2.01, 3.01, 4.01) an obstacle? {env.is_obstacle(2.01, 3.01, 4.01)}")
    print(f"Is (2.49, 3.49, 4.49) an obstacle? {env.is_obstacle(2.49, 3.49, 4.49)}")
    print(f"Is (1.51, 2.51, 3.51) an obstacle? {env.is_obstacle(1.51, 2.51, 3.51)}") # Should be False if no obstacle at (2,3,4) 