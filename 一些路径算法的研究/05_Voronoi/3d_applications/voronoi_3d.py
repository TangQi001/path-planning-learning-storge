
# Fix Chinese font display
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback font configuration
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

#!/usr/bin/env python3
"""
3D Voronoi Diagram Path Planning Implementation

This module extends the 2D Voronoi diagram path planning to 3D space,
enabling path planning for aerial vehicles and 3D environments.

Features:
- 3D Voronoi diagram generation
- 3D obstacle avoidance
- Height-aware path planning
- Terrain following capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import SphericalVoronoi, distance
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from implementation.core_voronoi import VoronoiPlanner
except ImportError:
    print("Warning: Could not import core_voronoi module. Running standalone.")
    VoronoiPlanner = None


class Voronoi3DPlanner:
    """3D Voronoi diagram path planner"""
    
    def __init__(self, bounds, obstacles=None):
        """
        Initialize 3D Voronoi planner
        
        Args:
            bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            obstacles: List of 3D obstacles (spheres or boxes)
        """
        self.bounds = np.array(bounds)
        self.obstacles = obstacles or []
        self.voronoi_points = []
        self.voronoi_graph = {}
        self.path = []
        
    def add_spherical_obstacle(self, center, radius):
        """Add spherical obstacle"""
        obstacle = {
            'type': 'sphere',
            'center': np.array(center),
            'radius': radius
        }
        self.obstacles.append(obstacle)
        
    def add_box_obstacle(self, min_corner, max_corner):
        """Add box-shaped obstacle"""
        obstacle = {
            'type': 'box',
            'min_corner': np.array(min_corner),
            'max_corner': np.array(max_corner)
        }
        self.obstacles.append(obstacle)
        
    def is_point_free(self, point):
        """Check if point is collision-free"""
        point = np.array(point)
        
        # Check bounds
        for i in range(3):
            if point[i] < self.bounds[i,0] or point[i] > self.bounds[i,1]:
                return False
                
        # Check obstacles
        for obstacle in self.obstacles:
            if obstacle['type'] == 'sphere':
                dist = np.linalg.norm(point - obstacle['center'])
                if dist <= obstacle['radius']:
                    return False
            elif obstacle['type'] == 'box':
                if np.all(point >= obstacle['min_corner']) and \
                   np.all(point <= obstacle['max_corner']):
                    return False
                    
        return True
        
    def generate_voronoi_points(self, num_points=100, safety_margin=0.5):
        """Generate Voronoi seed points around obstacles"""
        self.voronoi_points = []
        
        # Add boundary points
        x_range = self.bounds[0,1] - self.bounds[0,0]
        y_range = self.bounds[1,1] - self.bounds[1,0]
        z_range = self.bounds[2,1] - self.bounds[2,0]
        
        # Boundary grid points
        for x in np.linspace(self.bounds[0,0], self.bounds[0,1], 5):
            for y in np.linspace(self.bounds[1,0], self.bounds[1,1], 5):
                for z in [self.bounds[2,0], self.bounds[2,1]]:
                    if self.is_point_free([x, y, z]):
                        self.voronoi_points.append([x, y, z])
                        
        # Points around obstacles
        for obstacle in self.obstacles:
            if obstacle['type'] == 'sphere':
                center = obstacle['center']
                radius = obstacle['radius'] + safety_margin
                
                # Generate points on sphere surface
                phi = np.linspace(0, 2*np.pi, 12, endpoint=False)
                theta = np.linspace(0, np.pi, 6)
                
                for p in phi:
                    for t in theta:
                        x = center[0] + radius * np.sin(t) * np.cos(p)
                        y = center[1] + radius * np.sin(t) * np.sin(p)
                        z = center[2] + radius * np.cos(t)
                        
                        if self.is_point_free([x, y, z]):
                            self.voronoi_points.append([x, y, z])
                            
            elif obstacle['type'] == 'box':
                # Generate points around box faces
                min_c = obstacle['min_corner'] - safety_margin
                max_c = obstacle['max_corner'] + safety_margin
                
                faces = [
                    # Front and back faces
                    ([min_c[0], min_c[1], min_c[2]], [max_c[0], max_c[1], min_c[2]]),
                    ([min_c[0], min_c[1], max_c[2]], [max_c[0], max_c[1], max_c[2]]),
                    # Left and right faces
                    ([min_c[0], min_c[1], min_c[2]], [min_c[0], max_c[1], max_c[2]]),
                    ([max_c[0], min_c[1], min_c[2]], [max_c[0], max_c[1], max_c[2]]),
                    # Top and bottom faces
                    ([min_c[0], min_c[1], min_c[2]], [max_c[0], min_c[1], max_c[2]]),
                    ([min_c[0], max_c[1], min_c[2]], [max_c[0], max_c[1], max_c[2]])
                ]
                
                for face_min, face_max in faces:
                    for i in range(8):
                        point = face_min + np.random.random(3) * (face_max - face_min)
                        if self.is_point_free(point):
                            self.voronoi_points.append(point.tolist())
        
        # Random free space points
        attempts = 0
        while len(self.voronoi_points) < num_points and attempts < num_points * 5:
            point = [
                np.random.uniform(self.bounds[0,0], self.bounds[0,1]),
                np.random.uniform(self.bounds[1,0], self.bounds[1,1]),
                np.random.uniform(self.bounds[2,0], self.bounds[2,1])
            ]
            
            if self.is_point_free(point):
                self.voronoi_points.append(point)
            attempts += 1
            
        self.voronoi_points = np.array(self.voronoi_points)
        print(f"Generated {len(self.voronoi_points)} Voronoi points")
        
    def build_visibility_graph(self, max_distance=None):
        """Build visibility graph between Voronoi points"""
        if max_distance is None:
            max_distance = np.inf
            
        self.voronoi_graph = {i: [] for i in range(len(self.voronoi_points))}
        
        for i in range(len(self.voronoi_points)):
            for j in range(i+1, len(self.voronoi_points)):
                p1, p2 = self.voronoi_points[i], self.voronoi_points[j]
                dist = np.linalg.norm(p2 - p1)
                
                if dist <= max_distance and self.is_line_free(p1, p2):
                    self.voronoi_graph[i].append((j, dist))
                    self.voronoi_graph[j].append((i, dist))
                    
    def is_line_free(self, p1, p2, num_checks=20):
        """Check if line segment is collision-free"""
        for t in np.linspace(0, 1, num_checks):
            point = p1 + t * (p2 - p1)
            if not self.is_point_free(point):
                return False
        return True
        
    def find_path(self, start, goal):
        """Find path using A* on visibility graph"""
        # Add start and goal to graph temporarily
        start = np.array(start)
        goal = np.array(goal)
        
        if not self.is_point_free(start) or not self.is_point_free(goal):
            print("Start or goal position is not collision-free!")
            return []
            
        # Find nearest Voronoi points
        start_neighbors = []
        goal_neighbors = []
        
        for i, point in enumerate(self.voronoi_points):
            if self.is_line_free(start, point):
                dist = np.linalg.norm(point - start)
                start_neighbors.append((i, dist))
            if self.is_line_free(goal, point):
                dist = np.linalg.norm(point - goal)
                goal_neighbors.append((i, dist))
                
        if not start_neighbors or not goal_neighbors:
            print("Cannot connect start or goal to Voronoi graph!")
            return []
            
        # A* search
        start_idx = len(self.voronoi_points)
        goal_idx = len(self.voronoi_points) + 1
        
        # Extended graph including start and goal
        extended_graph = self.voronoi_graph.copy()
        extended_graph[start_idx] = start_neighbors
        extended_graph[goal_idx] = goal_neighbors
        
        # Add reverse connections
        for neighbor_idx, dist in start_neighbors:
            extended_graph[neighbor_idx].append((start_idx, dist))
        for neighbor_idx, dist in goal_neighbors:
            extended_graph[neighbor_idx].append((goal_idx, dist))
            
        # Extended points array
        extended_points = np.vstack([self.voronoi_points, [start], [goal]])
        
        # A* implementation
        open_set = [(0, start_idx, [])]
        closed_set = set()
        
        while open_set:
            f_cost, current, path = min(open_set)
            open_set.remove((f_cost, current, path))
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            new_path = path + [current]
            
            if current == goal_idx:
                # Reconstruct path with actual coordinates
                self.path = [extended_points[idx] for idx in new_path]
                return self.path
                
            for neighbor, edge_cost in extended_graph.get(current, []):
                if neighbor not in closed_set:
                    g_cost = f_cost - self.heuristic_3d(extended_points[current], extended_points[goal_idx]) + edge_cost
                    h_cost = self.heuristic_3d(extended_points[neighbor], extended_points[goal_idx])
                    f_cost_new = g_cost + h_cost
                    
                    open_set.append((f_cost_new, neighbor, new_path))
                    
        print("No path found!")
        return []
        
    def heuristic_3d(self, p1, p2):
        """3D Euclidean distance heuristic"""
        return np.linalg.norm(p2 - p1)
        
    def plan_path(self, start, goal, num_voronoi_points=100):
        """Complete path planning pipeline"""
        print("Generating Voronoi points...")
        self.generate_voronoi_points(num_voronoi_points)
        
        print("Building visibility graph...")
        self.build_visibility_graph()
        
        print("Finding path...")
        path = self.find_path(start, goal)
        
        if path:
            print(f"Path found with {len(path)} waypoints")
            print(f"Path length: {self.calculate_path_length():.2f}")
        
        return path
        
    def calculate_path_length(self):
        """Calculate total path length"""
        if len(self.path) < 2:
            return 0
            
        total_length = 0
        for i in range(len(self.path) - 1):
            total_length += np.linalg.norm(self.path[i+1] - self.path[i])
        return total_length
        
    def visualize_3d(self, start=None, goal=None):
        """Visualize 3D Voronoi diagram and path"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Voronoi points
        if len(self.voronoi_points) > 0:
            ax.scatter(self.voronoi_points[:,0], 
                      self.voronoi_points[:,1], 
                      self.voronoi_points[:,2], 
                      c='blue', alpha=0.6, s=20, label='Voronoi Points')
        
        # Plot obstacles
        for obstacle in self.obstacles:
            if obstacle['type'] == 'sphere':
                self.plot_sphere(ax, obstacle['center'], obstacle['radius'])
            elif obstacle['type'] == 'box':
                self.plot_box(ax, obstacle['min_corner'], obstacle['max_corner'])
                
        # Plot visibility graph
        for i, neighbors in self.voronoi_graph.items():
            for j, _ in neighbors:
                if i < j:  # Avoid duplicate edges
                    p1, p2 = self.voronoi_points[i], self.voronoi_points[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        # Plot path
        if len(self.path) > 1:
            path_array = np.array(self.path)
            ax.plot(path_array[:,0], path_array[:,1], path_array[:,2], 
                   'red', linewidth=3, label='Path')
            
        # Plot start and goal
        if start is not None:
            ax.scatter(*start, c='green', s=100, marker='o', label='Start')
        if goal is not None:
            ax.scatter(*goal, c='red', s=100, marker='*', label='Goal')
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Voronoi Path Planning')
        
        plt.tight_layout()
        plt.show()
        
    def plot_sphere(self, ax, center, radius, alpha=0.3):
        """Plot sphere obstacle"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=alpha, color='red')
        
    def plot_box(self, ax, min_corner, max_corner, alpha=0.3):
        """Plot box obstacle"""
        # Define the vertices of a box
        r = [min_corner[0], max_corner[0]]
        s = [min_corner[1], max_corner[1]]
        t = [min_corner[2], max_corner[2]]
        
        # Generate vertices
        vertices = []
        for x in r:
            for y in s:
                for z in t:
                    vertices.append([x, y, z])
        vertices = np.array(vertices)
        
        # Define the 12 edges of the box
        edges = [
            [0, 1], [2, 3], [4, 5], [6, 7],  # Bottom and top edges
            [0, 2], [1, 3], [4, 6], [5, 7],  # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
        ]
        
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, 'red', alpha=0.6, linewidth=2)


def demo_3d_voronoi():
    """Demonstration of 3D Voronoi path planning"""
    print("3D Voronoi Path Planning Demo")
    print("=" * 40)
    
    # Create environment
    bounds = [[-10, 10], [-10, 10], [0, 20]]
    planner = Voronoi3DPlanner(bounds)
    
    # Add obstacles
    planner.add_spherical_obstacle([0, 0, 10], 3)
    planner.add_spherical_obstacle([-5, 5, 15], 2)
    planner.add_box_obstacle([3, -2, 5], [7, 2, 12])
    
    # Plan path
    start = [-8, -8, 2]
    goal = [8, 8, 18]
    
    path = planner.plan_path(start, goal, num_voronoi_points=150)
    
    if path:
        print("\nPath planning successful!")
        print(f"Number of waypoints: {len(path)}")
        print(f"Total path length: {planner.calculate_path_length():.2f}")
        
        # Visualize
        try:
            planner.visualize_3d(start, goal)
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Path waypoints:")
            for i, waypoint in enumerate(path):
                print(f"  {i}: ({waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f})")
    else:
        print("Path planning failed!")


if __name__ == "__main__":
    try:
        demo_3d_voronoi()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()