
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
Advanced Voronoi Path Planning Features

This module implements advanced features and optimizations for Voronoi diagram-based
path planning, including dynamic obstacle handling, multi-robot coordination,
and real-time path adaptation.

Features:
- Dynamic obstacle handling
- Multi-robot path planning
- Informed Voronoi diagrams
- Risk-aware path planning
- Real-time path adaptation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
import heapq
import time
import sys
import os
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from implementation.core_voronoi import VoronoiPlanner
except ImportError:
    print("Warning: Could not import core_voronoi module. Running standalone.")
    VoronoiPlanner = None


class DynamicVoronoiPlanner:
    """Voronoi planner with dynamic obstacle handling"""
    
    def __init__(self, bounds, initial_obstacles=None):
        """
        Initialize dynamic Voronoi planner
        
        Args:
            bounds: Environment boundaries [x_min, x_max, y_min, y_max]
            initial_obstacles: Initial static obstacles
        """
        self.bounds = bounds
        self.static_obstacles = initial_obstacles or []
        self.dynamic_obstacles = []
        self.voronoi_points = []
        self.voronoi_diagram = None
        self.visibility_graph = {}
        self.current_path = []
        self.last_update_time = 0
        self.update_threshold = 0.1  # seconds
        
    def add_dynamic_obstacle(self, center, radius, velocity=None):
        """Add moving obstacle"""
        obstacle = {
            'center': np.array(center),
            'radius': radius,
            'velocity': np.array(velocity) if velocity else np.zeros(2),
            'timestamp': time.time()
        }
        self.dynamic_obstacles.append(obstacle)
        
    def update_dynamic_obstacles(self, dt=None):
        """Update positions of dynamic obstacles"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update_time
            
        for obstacle in self.dynamic_obstacles:
            obstacle['center'] += obstacle['velocity'] * dt
            obstacle['timestamp'] = current_time
            
        self.last_update_time = current_time
        
    def get_current_obstacles(self):
        """Get all current obstacles (static + dynamic)"""
        return self.static_obstacles + self.dynamic_obstacles
        
    def predict_obstacle_positions(self, prediction_time=1.0):
        """Predict future obstacle positions"""
        predicted_obstacles = self.static_obstacles.copy()
        
        for obstacle in self.dynamic_obstacles:
            predicted_center = obstacle['center'] + obstacle['velocity'] * prediction_time
            predicted_obstacle = obstacle.copy()
            predicted_obstacle['center'] = predicted_center
            predicted_obstacles.append(predicted_obstacle)
            
        return predicted_obstacles
        
    def generate_adaptive_voronoi_points(self, num_points=100, safety_margin=1.0):
        """Generate Voronoi points with adaptive density"""
        self.voronoi_points = []
        current_obstacles = self.get_current_obstacles()
        
        # Higher density around obstacles
        for obstacle in current_obstacles:
            center = obstacle['center']
            radius = obstacle['radius'] + safety_margin
            
            # Number of points proportional to obstacle importance
            num_obstacle_points = max(8, int(12 * radius))
            
            for i in range(num_obstacle_points):
                angle = 2 * np.pi * i / num_obstacle_points
                point = center + radius * np.array([np.cos(angle), np.sin(angle)])
                
                if self.is_point_in_bounds(point):
                    self.voronoi_points.append(point)
                    
        # Random points in free space
        attempts = 0
        while len(self.voronoi_points) < num_points and attempts < num_points * 3:
            point = np.array([
                np.random.uniform(self.bounds[0], self.bounds[1]),
                np.random.uniform(self.bounds[2], self.bounds[3])
            ])
            
            if self.is_point_free(point, current_obstacles):
                self.voronoi_points.append(point)
            attempts += 1
            
        self.voronoi_points = np.array(self.voronoi_points)
        
    def replan_if_needed(self, robot_position, goal, replan_threshold=2.0):
        """Check if replanning is needed and replan if necessary"""
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_update_time < self.update_threshold:
            return self.current_path
            
        # Update dynamic obstacles
        self.update_dynamic_obstacles()
        
        # Check if current path is still valid
        if not self.is_path_valid(self.current_path):
            print("Path invalid, replanning...")
            return self.plan_path(robot_position, goal)
            
        # Check if robot is too far from planned path
        if len(self.current_path) > 1:
            distance_to_path = self.distance_to_path(robot_position, self.current_path)
            if distance_to_path > replan_threshold:
                print("Robot too far from path, replanning...")
                return self.plan_path(robot_position, goal)
                
        return self.current_path
        
    def is_path_valid(self, path, safety_margin=0.5):
        """Check if path is still collision-free"""
        if len(path) < 2:
            return True
            
        current_obstacles = self.get_current_obstacles()
        
        for i in range(len(path) - 1):
            if not self.is_line_free(path[i], path[i+1], current_obstacles, safety_margin):
                return False
        return True
        
    def distance_to_path(self, point, path):
        """Calculate minimum distance from point to path"""
        if len(path) < 2:
            return float('inf')
            
        min_distance = float('inf')
        point = np.array(point)
        
        for i in range(len(path) - 1):
            seg_start = np.array(path[i])
            seg_end = np.array(path[i+1])
            
            # Distance to line segment
            segment_length = np.linalg.norm(seg_end - seg_start)
            if segment_length == 0:
                distance = np.linalg.norm(point - seg_start)
            else:
                t = max(0, min(1, np.dot(point - seg_start, seg_end - seg_start) / segment_length**2))
                projection = seg_start + t * (seg_end - seg_start)
                distance = np.linalg.norm(point - projection)
                
            min_distance = min(min_distance, distance)
            
        return min_distance
        
    def plan_path(self, start, goal, num_points=100):
        """Plan path with current obstacle configuration"""
        self.generate_adaptive_voronoi_points(num_points)
        
        # Generate Voronoi diagram
        if len(self.voronoi_points) < 4:
            print("Not enough Voronoi points generated")
            return []
            
        self.voronoi_diagram = Voronoi(self.voronoi_points)
        
        # Build visibility graph
        self.build_visibility_graph()
        
        # Find path
        path = self.find_path_astar(start, goal)
        self.current_path = path
        return path
        
    def build_visibility_graph(self):
        """Build visibility graph from Voronoi vertices"""
        self.visibility_graph = defaultdict(list)
        vertices = self.voronoi_diagram.vertices
        current_obstacles = self.get_current_obstacles()
        
        # Add valid Voronoi vertices
        valid_vertices = []
        for vertex in vertices:
            if (self.is_point_in_bounds(vertex) and 
                self.is_point_free(vertex, current_obstacles)):
                valid_vertices.append(vertex)
                
        # Connect visible vertices
        for i, v1 in enumerate(valid_vertices):
            for j, v2 in enumerate(valid_vertices):
                if i != j and self.is_line_free(v1, v2, current_obstacles):
                    distance = np.linalg.norm(v2 - v1)
                    self.visibility_graph[i].append((j, distance, v2))
                    
        self.valid_vertices = valid_vertices
        
    def find_path_astar(self, start, goal):
        """Find path using A* on visibility graph"""
        start = np.array(start)
        goal = np.array(goal)
        current_obstacles = self.get_current_obstacles()
        
        if not (self.is_point_free(start, current_obstacles) and 
                self.is_point_free(goal, current_obstacles)):
            print("Start or goal is not collision-free")
            return []
            
        # Find connections to start and goal
        start_connections = []
        goal_connections = []
        
        for i, vertex in enumerate(self.valid_vertices):
            if self.is_line_free(start, vertex, current_obstacles):
                dist = np.linalg.norm(vertex - start)
                start_connections.append((i, dist))
            if self.is_line_free(goal, vertex, current_obstacles):
                dist = np.linalg.norm(vertex - goal)
                goal_connections.append((i, dist))
                
        if not start_connections or not goal_connections:
            print("Cannot connect start or goal to visibility graph")
            return []
            
        # A* search
        open_set = []
        start_idx = -1
        goal_idx = -2
        
        # Add start connections to open set
        for vertex_idx, dist in start_connections:
            g_cost = dist
            h_cost = np.linalg.norm(self.valid_vertices[vertex_idx] - goal)
            f_cost = g_cost + h_cost
            heapq.heappush(open_set, (f_cost, g_cost, vertex_idx, [start, self.valid_vertices[vertex_idx]]))
            
        closed_set = set()
        
        while open_set:
            f_cost, g_cost, current_idx, path = heapq.heappop(open_set)
            
            if current_idx in closed_set:
                continue
                
            closed_set.add(current_idx)
            
            # Check if we can reach goal
            current_vertex = self.valid_vertices[current_idx]
            if self.is_line_free(current_vertex, goal, current_obstacles):
                return path + [goal]
                
            # Expand neighbors
            for neighbor_idx, edge_dist, neighbor_pos in self.visibility_graph[current_idx]:
                if neighbor_idx not in closed_set:
                    new_g_cost = g_cost + edge_dist
                    h_cost = np.linalg.norm(neighbor_pos - goal)
                    new_f_cost = new_g_cost + h_cost
                    new_path = path + [neighbor_pos]
                    
                    heapq.heappush(open_set, (new_f_cost, new_g_cost, neighbor_idx, new_path))
                    
        print("No path found")
        return []
        
    def is_point_free(self, point, obstacles=None, safety_margin=0.1):
        """Check if point is collision-free"""
        if obstacles is None:
            obstacles = self.get_current_obstacles()
            
        for obstacle in obstacles:
            distance = np.linalg.norm(point - obstacle['center'])
            if distance <= obstacle['radius'] + safety_margin:
                return False
        return True
        
    def is_point_in_bounds(self, point):
        """Check if point is within environment bounds"""
        return (self.bounds[0] <= point[0] <= self.bounds[1] and
                self.bounds[2] <= point[1] <= self.bounds[3])
                
    def is_line_free(self, p1, p2, obstacles=None, safety_margin=0.1, num_checks=20):
        """Check if line segment is collision-free"""
        if obstacles is None:
            obstacles = self.get_current_obstacles()
            
        for t in np.linspace(0, 1, num_checks):
            point = p1 + t * (p2 - p1)
            if not self.is_point_free(point, obstacles, safety_margin):
                return False
        return True
        
    def visualize(self, start=None, goal=None, robot_position=None):
        """Visualize the dynamic environment and path"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot static obstacles
        for obstacle in self.static_obstacles:
            circle = plt.Circle(obstacle['center'], obstacle['radius'], 
                              color='red', alpha=0.5, label='Static Obstacle')
            ax.add_patch(circle)
            
        # Plot dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            circle = plt.Circle(obstacle['center'], obstacle['radius'], 
                              color='orange', alpha=0.5, label='Dynamic Obstacle')
            ax.add_patch(circle)
            
            # Plot velocity vector
            if np.linalg.norm(obstacle['velocity']) > 0:
                ax.arrow(obstacle['center'][0], obstacle['center'][1],
                        obstacle['velocity'][0], obstacle['velocity'][1],
                        head_width=0.3, head_length=0.2, fc='orange', ec='orange')
                        
        # Plot Voronoi diagram
        if self.voronoi_diagram is not None and len(self.voronoi_points) > 0:
            voronoi_plot_2d(self.voronoi_diagram, ax=ax, show_vertices=True, 
                           line_colors='blue', line_width=1, line_alpha=0.3,
                           point_size=4)
                           
        # Plot visibility graph
        for i, connections in self.visibility_graph.items():
            v1 = self.valid_vertices[i]
            for j, dist, v2 in connections:
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 
                       'gray', alpha=0.3, linewidth=0.5)
                       
        # Plot current path
        if len(self.current_path) > 1:
            path_array = np.array(self.current_path)
            ax.plot(path_array[:,0], path_array[:,1], 
                   'green', linewidth=3, label='Current Path')
                   
        # Plot points
        if start is not None:
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        if goal is not None:
            ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
        if robot_position is not None:
            ax.plot(robot_position[0], robot_position[1], 'bo', markersize=8, label='Robot')
            
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Dynamic Voronoi Path Planning')
        
        plt.tight_layout()
        plt.show()


class MultiRobotVoronoiPlanner:
    """Multi-robot coordination using Voronoi diagrams"""
    
    def __init__(self, bounds, obstacles=None):
        self.bounds = bounds
        self.obstacles = obstacles or []
        self.robots = {}
        self.robot_paths = {}
        
    def add_robot(self, robot_id, position, goal, radius=0.5):
        """Add robot to the system"""
        self.robots[robot_id] = {
            'position': np.array(position),
            'goal': np.array(goal),
            'radius': radius,
            'path': [],
            'last_update': time.time()
        }
        
    def plan_coordinated_paths(self):
        """Plan paths for all robots with collision avoidance"""
        robot_ids = list(self.robots.keys())
        
        # Treat other robots as dynamic obstacles
        for robot_id in robot_ids:
            other_robots = [r for r in robot_ids if r != robot_id]
            
            # Create dynamic planner for this robot
            planner = DynamicVoronoiPlanner(self.bounds, self.obstacles)
            
            # Add other robots as dynamic obstacles
            for other_id in other_robots:
                other_robot = self.robots[other_id]
                planner.add_dynamic_obstacle(
                    other_robot['position'], 
                    other_robot['radius'] * 2,  # Safety margin
                    velocity=np.zeros(2)  # Assume stationary for planning
                )
                
            # Plan path
            path = planner.plan_path(
                self.robots[robot_id]['position'],
                self.robots[robot_id]['goal']
            )
            
            self.robot_paths[robot_id] = path
            self.robots[robot_id]['path'] = path
            
        return self.robot_paths
        
    def visualize_multi_robot(self):
        """Visualize multi-robot scenario"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle['center'], obstacle['radius'], 
                              color='red', alpha=0.5)
            ax.add_patch(circle)
            
        # Plot robots and their paths
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, (robot_id, robot) in enumerate(self.robots.items()):
            color = colors[i % len(colors)]
            
            # Plot robot
            circle = plt.Circle(robot['position'], robot['radius'], 
                              color=color, alpha=0.7, label=f'Robot {robot_id}')
            ax.add_patch(circle)
            
            # Plot goal
            ax.plot(robot['goal'][0], robot['goal'][1], 
                   '*', color=color, markersize=15)
                   
            # Plot path
            if robot_id in self.robot_paths and len(self.robot_paths[robot_id]) > 1:
                path = np.array(self.robot_paths[robot_id])
                ax.plot(path[:,0], path[:,1], color=color, linewidth=2, alpha=0.8)
                
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Multi-Robot Voronoi Path Planning')
        
        plt.tight_layout()
        plt.show()


def demo_dynamic_voronoi():
    """Demonstration of dynamic Voronoi path planning"""
    print("Dynamic Voronoi Path Planning Demo")
    print("=" * 40)
    
    # Create environment
    bounds = [0, 20, 0, 20]
    static_obstacles = [
        {'center': np.array([5, 5]), 'radius': 2},
        {'center': np.array([15, 15]), 'radius': 1.5}
    ]
    
    planner = DynamicVoronoiPlanner(bounds, static_obstacles)
    
    # Add dynamic obstacles
    planner.add_dynamic_obstacle([10, 2], 1, velocity=[0, 1])
    planner.add_dynamic_obstacle([2, 18], 0.8, velocity=[1, -0.5])
    
    # Plan initial path
    start = [1, 1]
    goal = [19, 19]
    
    print("Planning initial path...")
    path = planner.plan_path(start, goal)
    
    if path:
        print(f"Initial path found with {len(path)} waypoints")
        
        # Simulate robot movement and replanning
        robot_pos = np.array(start)
        simulation_steps = 10
        
        for step in range(simulation_steps):
            print(f"\nSimulation step {step + 1}")
            
            # Update robot position (move along path)
            if len(path) > 1:
                direction = path[1] - robot_pos
                direction = direction / np.linalg.norm(direction)
                robot_pos += direction * 0.5
                
            # Check if replanning is needed
            new_path = planner.replan_if_needed(robot_pos, goal)
            
            if len(new_path) != len(path):
                print("Path updated due to dynamic obstacles")
                path = new_path
                
            time.sleep(0.1)  # Simulate time passing
            
        print("\nFinal robot position:", robot_pos)
        
        # Visualize final state
        try:
            planner.visualize(start, goal, robot_pos)
        except Exception as e:
            print(f"Visualization error: {e}")
            
    else:
        print("No path found!")


def demo_multi_robot():
    """Demonstration of multi-robot coordination"""
    print("\nMulti-Robot Voronoi Planning Demo")
    print("=" * 40)
    
    bounds = [0, 15, 0, 15]
    obstacles = [
        {'center': np.array([7, 7]), 'radius': 2}
    ]
    
    planner = MultiRobotVoronoiPlanner(bounds, obstacles)
    
    # Add robots
    planner.add_robot(1, [1, 1], [13, 13])
    planner.add_robot(2, [13, 1], [1, 13])
    planner.add_robot(3, [7, 1], [7, 13])
    
    print("Planning coordinated paths...")
    paths = planner.plan_coordinated_paths()
    
    for robot_id, path in paths.items():
        if path:
            print(f"Robot {robot_id}: Path with {len(path)} waypoints")
        else:
            print(f"Robot {robot_id}: No path found")
            
    # Visualize
    try:
        planner.visualize_multi_robot()
    except Exception as e:
        print(f"Visualization error: {e}")


if __name__ == "__main__":
    try:
        demo_dynamic_voronoi()
        demo_multi_robot()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()