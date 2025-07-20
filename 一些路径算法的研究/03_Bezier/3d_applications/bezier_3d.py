
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
3D Bezier Curve Path Planning Implementation

This module extends Bezier curve path planning to 3D space, enabling smooth
trajectory generation for aerial vehicles and 3D robotic applications.

Features:
- 3D Bezier curve generation
- Altitude-aware path planning
- Smooth 3D trajectory interpolation
- Constraint handling for 3D motion
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from implementation.core_algorithm import BezierPlanner
except ImportError:
    print("Warning: Could not import core_algorithm module. Running standalone.")
    BezierPlanner = None


class Bezier3DPlanner:
    """3D Bezier curve path planner"""
    
    def __init__(self, bounds=None, obstacles=None):
        """
        Initialize 3D Bezier planner
        
        Args:
            bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            obstacles: List of 3D obstacles
        """
        self.bounds = bounds or [[-10, 10], [-10, 10], [0, 20]]
        self.obstacles = obstacles or []
        self.control_points = []
        self.curve_points = []
        self.path_length = 0
        
    def add_spherical_obstacle(self, center, radius):
        """Add spherical obstacle"""
        self.obstacles.append({
            'type': 'sphere',
            'center': np.array(center),
            'radius': radius
        })
        
    def add_cylindrical_obstacle(self, center, radius, height):
        """Add cylindrical obstacle"""
        self.obstacles.append({
            'type': 'cylinder',
            'center': np.array(center),
            'radius': radius,
            'height': height
        })
        
    def bezier_curve_3d(self, control_points, t):
        """
        Evaluate 3D Bezier curve at parameter t
        
        Args:
            control_points: Array of 3D control points
            t: Parameter value (0 to 1)
            
        Returns:
            3D point on the curve
        """
        n = len(control_points) - 1
        point = np.zeros(3)
        
        for i, cp in enumerate(control_points):
            bernstein = self.bernstein_polynomial(n, i, t)
            point += bernstein * cp
            
        return point
        
    def bernstein_polynomial(self, n, i, t):
        """Compute Bernstein polynomial"""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        
    def bezier_derivative_3d(self, control_points, t):
        """
        Compute first derivative of 3D Bezier curve
        
        Args:
            control_points: Array of 3D control points
            t: Parameter value (0 to 1)
            
        Returns:
            3D tangent vector
        """
        n = len(control_points) - 1
        if n == 0:
            return np.zeros(3)
            
        derivative_points = []
        for i in range(n):
            derivative_points.append(n * (control_points[i+1] - control_points[i]))
            
        return self.bezier_curve_3d(derivative_points, t)
        
    def bezier_second_derivative_3d(self, control_points, t):
        """
        Compute second derivative of 3D Bezier curve
        
        Args:
            control_points: Array of 3D control points
            t: Parameter value (0 to 1)
            
        Returns:
            3D curvature vector
        """
        n = len(control_points) - 1
        if n <= 1:
            return np.zeros(3)
            
        # First derivative control points
        first_deriv_points = []
        for i in range(n):
            first_deriv_points.append(n * (control_points[i+1] - control_points[i]))
            
        # Second derivative control points
        m = len(first_deriv_points) - 1
        second_deriv_points = []
        for i in range(m):
            second_deriv_points.append(m * (first_deriv_points[i+1] - first_deriv_points[i]))
            
        return self.bezier_curve_3d(second_deriv_points, t)
        
    def curvature_3d(self, control_points, t):
        """
        Compute curvature at parameter t
        
        Args:
            control_points: Array of 3D control points
            t: Parameter value (0 to 1)
            
        Returns:
            Curvature value
        """
        first_deriv = self.bezier_derivative_3d(control_points, t)
        second_deriv = self.bezier_second_derivative_3d(control_points, t)
        
        cross_product = np.cross(first_deriv, second_deriv)
        cross_magnitude = np.linalg.norm(cross_product)
        first_deriv_magnitude = np.linalg.norm(first_deriv)
        
        if first_deriv_magnitude < 1e-10:
            return 0
            
        return cross_magnitude / (first_deriv_magnitude ** 3)
        
    def generate_initial_control_points(self, start, goal, num_intermediate=2):
        """
        Generate initial control points for optimization
        
        Args:
            start: Starting 3D point
            goal: Goal 3D point
            num_intermediate: Number of intermediate control points
            
        Returns:
            Array of initial control points
        """
        start = np.array(start)
        goal = np.array(goal)
        
        control_points = [start]
        
        # Generate intermediate points along straight line with altitude variation
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            
            # Linear interpolation with altitude boost
            point = start + t * (goal - start)
            
            # Add altitude variation for smoother 3D curves
            altitude_boost = 2 * np.sin(np.pi * t)  # Arc shape in altitude
            point[2] += altitude_boost
            
            # Add some randomness to avoid local minima
            noise = np.random.normal(0, 0.5, 3)
            noise[2] *= 0.5  # Less noise in altitude
            point += noise
            
            # Ensure point is within bounds
            for j in range(3):
                point[j] = np.clip(point[j], self.bounds[j][0], self.bounds[j][1])
                
            control_points.append(point)
            
        control_points.append(goal)
        return np.array(control_points)
        
    def is_point_collision_free(self, point, safety_margin=0.5):
        """Check if point is collision-free"""
        point = np.array(point)
        
        for obstacle in self.obstacles:
            if obstacle['type'] == 'sphere':
                distance = np.linalg.norm(point - obstacle['center'])
                if distance <= obstacle['radius'] + safety_margin:
                    return False
                    
            elif obstacle['type'] == 'cylinder':
                # Check horizontal distance
                center_2d = obstacle['center'][:2]
                point_2d = point[:2]
                horizontal_distance = np.linalg.norm(point_2d - center_2d)
                
                # Check if within cylinder height
                if (horizontal_distance <= obstacle['radius'] + safety_margin and
                    obstacle['center'][2] <= point[2] <= obstacle['center'][2] + obstacle['height']):
                    return False
                    
        return True
        
    def is_curve_collision_free(self, control_points, num_samples=100):
        """Check if entire curve is collision-free"""
        for i in range(num_samples + 1):
            t = i / num_samples
            point = self.bezier_curve_3d(control_points, t)
            
            if not self.is_point_collision_free(point):
                return False
                
        return True
        
    def objective_function(self, flat_control_points, start, goal, num_intermediate):
        """
        Objective function for control point optimization
        
        Args:
            flat_control_points: Flattened array of intermediate control points
            start: Starting point
            goal: Goal point
            num_intermediate: Number of intermediate points
            
        Returns:
            Cost value
        """
        # Reshape flat array to 3D points
        intermediate_points = flat_control_points.reshape(num_intermediate, 3)
        control_points = np.vstack([start, intermediate_points, goal])
        
        # Check collision
        if not self.is_curve_collision_free(control_points):
            return 1e6  # High penalty for collision
            
        # Path length cost
        length_cost = self.compute_curve_length(control_points)
        
        # Curvature cost (smoothness)
        curvature_cost = 0
        num_samples = 20
        for i in range(1, num_samples):
            t = i / num_samples
            curvature = self.curvature_3d(control_points, t)
            curvature_cost += curvature ** 2
            
        curvature_cost /= num_samples
        
        # Altitude variation cost (prefer smoother altitude changes)
        altitude_cost = 0
        for i in range(1, num_samples):
            t1 = (i-1) / num_samples
            t2 = i / num_samples
            point1 = self.bezier_curve_3d(control_points, t1)
            point2 = self.bezier_curve_3d(control_points, t2)
            altitude_change = abs(point2[2] - point1[2])
            altitude_cost += altitude_change ** 2
            
        altitude_cost /= num_samples
        
        # Total cost
        total_cost = length_cost + 0.5 * curvature_cost + 0.3 * altitude_cost
        
        return total_cost
        
    def compute_curve_length(self, control_points, num_samples=100):
        """Compute approximate curve length"""
        total_length = 0
        
        for i in range(num_samples):
            t1 = i / num_samples
            t2 = (i + 1) / num_samples
            
            point1 = self.bezier_curve_3d(control_points, t1)
            point2 = self.bezier_curve_3d(control_points, t2)
            
            segment_length = np.linalg.norm(point2 - point1)
            total_length += segment_length
            
        return total_length
        
    def optimize_path(self, start, goal, num_intermediate=3, max_iterations=100):
        """
        Optimize 3D Bezier path using control point optimization
        
        Args:
            start: Starting 3D point
            goal: Goal 3D point
            num_intermediate: Number of intermediate control points
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized control points
        """
        start = np.array(start)
        goal = np.array(goal)
        
        # Generate initial control points
        initial_control_points = self.generate_initial_control_points(start, goal, num_intermediate)
        initial_intermediate = initial_control_points[1:-1]  # Exclude start and goal
        
        # Flatten intermediate points for optimization
        x0 = initial_intermediate.flatten()
        
        # Bounds for intermediate control points
        bounds = []
        for _ in range(num_intermediate):
            for j in range(3):
                bounds.append((self.bounds[j][0], self.bounds[j][1]))
                
        # Optimization
        result = minimize(
            self.objective_function,
            x0,
            args=(start, goal, num_intermediate),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )
        
        if result.success:
            # Reconstruct control points
            optimized_intermediate = result.x.reshape(num_intermediate, 3)
            self.control_points = np.vstack([start, optimized_intermediate, goal])
            
            print(f"Optimization successful. Cost: {result.fun:.2f}")
            return self.control_points
        else:
            print(f"Optimization failed: {result.message}")
            self.control_points = initial_control_points
            return self.control_points
            
    def generate_trajectory(self, control_points=None, num_points=100):
        """
        Generate trajectory points from control points
        
        Args:
            control_points: Control points (uses stored ones if None)
            num_points: Number of trajectory points
            
        Returns:
            Array of trajectory points
        """
        if control_points is None:
            control_points = self.control_points
            
        if len(control_points) == 0:
            print("No control points available")
            return np.array([])
            
        trajectory = []
        
        for i in range(num_points + 1):
            t = i / num_points
            point = self.bezier_curve_3d(control_points, t)
            trajectory.append(point)
            
        self.curve_points = np.array(trajectory)
        self.path_length = self.compute_curve_length(control_points)
        
        return self.curve_points
        
    def plan_path(self, start, goal, num_intermediate=3, num_trajectory_points=100):
        """
        Complete 3D path planning pipeline
        
        Args:
            start: Starting 3D point
            goal: Goal 3D point
            num_intermediate: Number of intermediate control points
            num_trajectory_points: Number of final trajectory points
            
        Returns:
            Trajectory points
        """
        print("Starting 3D Bezier path planning...")
        
        # Optimize control points
        control_points = self.optimize_path(start, goal, num_intermediate)
        
        # Generate trajectory
        trajectory = self.generate_trajectory(control_points, num_trajectory_points)
        
        if len(trajectory) > 0:
            print(f"Path planning successful!")
            print(f"Path length: {self.path_length:.2f}")
            print(f"Number of trajectory points: {len(trajectory)}")
            
        return trajectory
        
    def visualize_3d(self, start=None, goal=None, show_control_points=True):
        """Visualize 3D Bezier curve and environment"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot obstacles
        for obstacle in self.obstacles:
            if obstacle['type'] == 'sphere':
                self.plot_sphere(ax, obstacle['center'], obstacle['radius'])
            elif obstacle['type'] == 'cylinder':
                self.plot_cylinder(ax, obstacle['center'], obstacle['radius'], obstacle['height'])
                
        # Plot control points
        if show_control_points and len(self.control_points) > 0:
            cp = self.control_points
            ax.scatter(cp[:,0], cp[:,1], cp[:,2], 
                      c='blue', s=100, alpha=0.7, label='Control Points')
            
            # Connect control points
            ax.plot(cp[:,0], cp[:,1], cp[:,2], 
                   'b--', alpha=0.5, linewidth=1, label='Control Polygon')
                   
        # Plot trajectory
        if len(self.curve_points) > 0:
            traj = self.curve_points
            ax.plot(traj[:,0], traj[:,1], traj[:,2], 
                   'red', linewidth=3, label='Bezier Trajectory')
                   
        # Plot start and goal
        if start is not None:
            ax.scatter(*start, c='green', s=150, marker='o', label='Start')
        if goal is not None:
            ax.scatter(*goal, c='red', s=150, marker='*', label='Goal')
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Bezier Path Planning')
        
        # Set equal aspect ratio
        max_range = max([
            self.bounds[0][1] - self.bounds[0][0],
            self.bounds[1][1] - self.bounds[1][0],
            self.bounds[2][1] - self.bounds[2][0]
        ]) / 2
        
        mid_x = (self.bounds[0][0] + self.bounds[0][1]) / 2
        mid_y = (self.bounds[1][0] + self.bounds[1][1]) / 2
        mid_z = (self.bounds[2][0] + self.bounds[2][1]) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
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
        
    def plot_cylinder(self, ax, center, radius, height, alpha=0.3):
        """Plot cylindrical obstacle"""
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(center[2], center[2] + height, 20)
        
        # Cylinder surface
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = center[0] + radius * np.cos(theta_mesh)
        y_mesh = center[1] + radius * np.sin(theta_mesh)
        
        ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=alpha, color='red')
        
        # Top and bottom circles
        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)
        z_bottom = np.full_like(x_circle, center[2])
        z_top = np.full_like(x_circle, center[2] + height)
        
        ax.plot(x_circle, y_circle, z_bottom, 'r-', linewidth=2)
        ax.plot(x_circle, y_circle, z_top, 'r-', linewidth=2)


def demo_3d_bezier():
    """Demonstration of 3D Bezier path planning"""
    print("3D Bezier Path Planning Demo")
    print("=" * 40)
    
    # Create environment
    bounds = [[-10, 10], [-10, 10], [0, 15]]
    planner = Bezier3DPlanner(bounds)
    
    # Add obstacles
    planner.add_spherical_obstacle([0, 0, 7], 3)
    planner.add_cylindrical_obstacle([-4, 4, 0], 1.5, 8)
    planner.add_spherical_obstacle([5, -3, 12], 2)
    
    # Plan path
    start = [-8, -8, 2]
    goal = [8, 8, 13]
    
    trajectory = planner.plan_path(start, goal, num_intermediate=4)
    
    if len(trajectory) > 0:
        print("\nPath planning results:")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Path length: {planner.path_length:.2f}")
        print(f"Number of waypoints: {len(trajectory)}")
        
        # Analyze path properties
        max_curvature = 0
        for i in range(1, len(trajectory)-1):
            t = i / (len(trajectory) - 1)
            curvature = planner.curvature_3d(planner.control_points, t)
            max_curvature = max(max_curvature, curvature)
            
        print(f"Maximum curvature: {max_curvature:.4f}")
        
        # Visualize
        try:
            planner.visualize_3d(start, goal)
        except Exception as e:
            print(f"Visualization error: {e}")
            print("First 5 trajectory points:")
            for i, point in enumerate(trajectory[:5]):
                print(f"  {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
                
    else:
        print("Path planning failed!")


if __name__ == "__main__":
    try:
        demo_3d_bezier()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()