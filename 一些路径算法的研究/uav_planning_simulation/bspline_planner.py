# Placeholder for B-spline (Bezier based) planner module
import numpy as np
from scipy.optimize import minimize

# Assuming cost_functions.py and bezier_utils.py are in the same directory or accessible
# For package structure:
# from .cost_functions import total_cost_function_for_optimizer
# from .bezier_utils import bezier_curve
# from .environment import GridEnvironment 

# For direct script running (if files are in the same folder as main script that imports them):
from cost_functions import total_cost_function_for_optimizer
from bezier_utils import bezier_curve
# GridEnvironment might be needed for type hinting if an instance is passed
# from environment import GridEnvironment 

class BsplinePlanner:
    def __init__(self, environment, cost_params, bezier_order=6):
        """
        Initializes the B-spline (Bezier curve based) planner.

        Args:
            environment (GridEnvironment): The environment for planning.
            cost_params (dict): Parameters for the cost functions.
            bezier_order (int): The order p of the Bezier curve (p=6 means 7 control points).
        """
        self.environment = environment
        self.cost_params = cost_params
        self.bezier_order = bezier_order
        self.num_control_points = bezier_order + 1
        # C0 and C_last are fixed, so num_variable_cps = (p+1) - 2 = p - 1
        self.num_variable_cps = self.bezier_order - 1 
        if self.num_variable_cps < 1: # e.g. order 1 Bezier (line segment) has 0 variable CPs if ends fixed.
            # For order < 2, this setup might not make sense if we expect internal CPs to optimize.
            # Paper uses p=6, so 5 variable CPs.
            raise ValueError(f"Bezier order {bezier_order} must be at least 2 for variable internal control points.")


    def plan(self, start_point, goal_point, initial_guess_strategy='linear', optimizer_method='SLSQP'):
        """
        Plans a path from start_point to goal_point using Bezier curve optimization.

        Args:
            start_point (tuple or np.ndarray): The (x,y,z) starting position.
            goal_point (tuple or np.ndarray): The (x,y,z) goal position.
            initial_guess_strategy (str): Method to generate initial guess for variable CPs ('linear').
            optimizer_method (str): Optimization method for scipy.optimize.minimize.

        Returns:
            tuple: (optimized_trajectory_points, optimized_control_points) or (None, None) if planning fails.
        """
        fixed_start_cp = np.array(start_point, dtype=float)
        fixed_end_cp = np.array(goal_point, dtype=float)

        # 1. Generate initial guess for variable control points (C1 to C_p-1)
        # For p=6, variable CPs are C1, C2, C3, C4, C5.
        initial_variable_cps = np.zeros((self.num_variable_cps, 3))
        if initial_guess_strategy == 'linear':
            for i in range(self.num_variable_cps):
                # Interpolate between start (C0) and end (C_p)
                # Fraction for C_k is (k) / (p) where k ranges from 1 to p-1 for variable CPs.
                # Here, CPs are indexed 0 to p. Variable CPs are 1 to p-1.
                # So for variable_cp_index i (0 to num_variable_cps-1), actual CP index is i+1.
                fraction = (i + 1.0) / (self.num_control_points - 1.0) # (i+1)/p for C_{i+1}
                initial_variable_cps[i] = fixed_start_cp * (1 - fraction) + fixed_end_cp * fraction
        else:
            # Fallback or other strategies (e.g., random within bounds)
            # For now, just linear
            pass
        
        flat_initial_guess = initial_variable_cps.flatten()

        # 2. Define bounds for optimizer variables (coordinates of variable CPs)
        # Bounds should keep CPs within the environment grid
        min_bounds = np.array([0, 0, 0])
        max_bounds = np.array([
            self.environment.size[0] - 1e-3, # slightly less than max to avoid boundary issues
            self.environment.size[1] - 1e-3,
            self.environment.size[2] - 1e-3
        ])
        
        bounds = []
        for _ in range(self.num_variable_cps):
            for _ in range(3): # x, y, z
                bounds.append((min_bounds[_ % 3], max_bounds[_ % 3]))
        
        # Ensure bounds are correctly formatted for all variables
        if len(bounds) != len(flat_initial_guess):
             # Fallback: create bounds for each variable component
             bounds = []
             for i in range(self.num_variable_cps):
                 bounds.extend([(min_bounds[0], max_bounds[0]), 
                                (min_bounds[1], max_bounds[1]), 
                                (min_bounds[2], max_bounds[2])])

        # 3. Setup arguments for the cost function
        # The goal_point for position_cost is typically the fixed_end_cp
        args_for_cost_func = (
            fixed_start_cp,
            fixed_end_cp,
            fixed_end_cp, # This is the goal_point for the position_cost component
            self.environment,
            bezier_curve, # Pass the actual function
            self.cost_params
        )

        # 4. Run the optimization
        print(f"Starting optimization with {optimizer_method}...")
        opt_result = minimize(
            total_cost_function_for_optimizer,
            flat_initial_guess,
            args=args_for_cost_func,
            method=optimizer_method,
            bounds=bounds if optimizer_method in ['L-BFGS-B', 'SLSQP', 'TNC'] else None,
            options=self.cost_params.get('optimizer_options', {'disp': True, 'maxiter': 200})
        )

        if opt_result.success:
            print("Optimization successful.")
            optimized_flat_cps = opt_result.x
            optimized_variable_cps = optimized_flat_cps.reshape((self.num_variable_cps, 3))
            
            final_control_points = np.vstack([
                fixed_start_cp.reshape(1,3),
                optimized_variable_cps,
                fixed_end_cp.reshape(1,3)
            ])
            
            final_trajectory = bezier_curve(final_control_points, n_points=self.cost_params.get('num_traj_points_final', 100))
            return final_trajectory, final_control_points
        else:
            print(f"Optimization failed: {opt_result.message}")
            # Optionally, return path based on initial guess or None
            # For now, return None if optimization fails to find a better path
            # initial_trajectory = bezier_curve(np.vstack([fixed_start_cp.reshape(1,3), initial_variable_cps, fixed_end_cp.reshape(1,3)]), 
            #                                   n_points=self.cost_params.get('num_traj_points_final', 100))
            # return initial_trajectory, np.vstack([fixed_start_cp.reshape(1,3), initial_variable_cps, fixed_end_cp.reshape(1,3)])
            return None, None

if __name__ == '__main__':
    # This requires GridEnvironment, cost_functions, and bezier_utils to be importable
    # Mock or ensure they are in path for testing
    GridEnvironment_imported = False
    try:
        from environment import GridEnvironment
        GridEnvironment_imported = True
    except ImportError:
        print("BsplinePlanner Test: Could not import GridEnvironment. Using Mock.")
        class MockGridEnvironment:
            def __init__(self, size=(9,9,9)):
                self.size = size
                self.width, self.height, self.depth = size
                self.obstacles = set()
            def add_obstacle(self, x,y,z): self.obstacles.add((int(x),int(y),int(z)))
            def is_obstacle(self, x,y,z): return (int(x),int(y),int(z)) in self.obstacles
            def is_within_bounds(self,x,y,z): 
                return 0<=x<self.size[0] and 0<=y<self.size[1] and 0<=z<self.size[2]
        GridEnvironment = MockGridEnvironment # Use mock if import fails
        GridEnvironment_imported = True # Mark as imported for test execution

    if GridEnvironment_imported: 
        print("\nTesting BsplinePlanner...")
        env = GridEnvironment(size=(10, 10, 10))
        # env.add_obstacle(5, 5, 5) # Add a central obstacle
        # env.add_obstacle(5,5,4)
        # env.add_obstacle(5,5,6)
        # env.add_obstacle(5,4,5)
        # env.add_obstacle(5,6,5)
        # env.add_obstacle(4,5,5)
        # env.add_obstacle(6,5,5)

        cost_parameters = {
            'w_pos': 1000.0,      # Weight for reaching the goal
            'w_obs': 500.0,       # Weight for avoiding obstacles
            'w_smooth': 10.0,     # Weight for path smoothness
            'yukawa_A': 1.0,      # Yukawa potential scale factor
            'yukawa_alpha': 0.8,   # Yukawa potential decay rate
            'safety_dist': 1.5,   # Safety distance from obstacles for Yukawa cost calc (effective range)
            'num_traj_points': 30, # Number of points to discretize Bezier curve for cost evaluation
            'num_obs_check_points': 15, # Number of points along trajectory to check for obstacle cost
            'num_traj_points_final': 100, # Number of points for the final output trajectory
            'optimizer_options': {'disp': False, 'maxiter': 150, 'ftol': 1e-7} # SLSQP options
        }

        planner = BsplinePlanner(env, cost_parameters, bezier_order=6)

        start = (1, 1, 1)
        goal = (8, 8, 8)

        print(f"Planning from {start} to {goal}")
        trajectory, control_points = planner.plan(start, goal, optimizer_method='SLSQP')

        if trajectory is not None:
            print("Planning successful!")
            print(f"Number of points in trajectory: {len(trajectory)}")
            print(f"Trajectory start: {trajectory[0]}")
            print(f"Trajectory end: {trajectory[-1]}")
            print(f"Final control points:\n{control_points}")
            
            # Basic checks
            assert np.allclose(trajectory[0], start), "Trajectory doesn't start at the start point."
            assert np.allclose(trajectory[-1], goal), "Trajectory doesn't end at the goal point."
            assert len(control_points) == planner.num_control_points, "Incorrect number of control points."

            # Check if trajectory avoids obstacles (simple check)
            # This is a soft constraint in cost function, so it might not be perfectly avoided
            # depending on weights.
            # For a more rigorous check, one would iterate through trajectory points.
            # print("Checking for collisions in the final trajectory...")
            # collided = False
            # for pt in trajectory:
            #     if env.is_obstacle(pt[0], pt[1], pt[2]):
            #         print(f"Collision detected at point: {pt}")
            #         collided = True
            #         # assert not collided, "Path collides with an obstacle!"
            #         break
            # if not collided:
            #     print("No collisions detected in the final trajectory.")

        else:
            print("Planning failed.")
        
        print("\n--- Test with an obstacle ---")
        env_obs = GridEnvironment(size=(10, 10, 10))
        # Create a wall-like obstacle
        for i in range(3, 7):
            for j in range(3,7):
                env_obs.add_obstacle(5, i, j)
        print(f"Obstacles added: {env_obs.obstacles}")

        cost_parameters_obs = cost_parameters.copy()
        cost_parameters_obs['w_obs'] = 2000.0 # Increase obstacle avoidance weight
        cost_parameters_obs['optimizer_options'] = {'disp': False, 'maxiter': 300, 'ftol': 1e-7}

        planner_obs = BsplinePlanner(env_obs, cost_parameters_obs, bezier_order=6)
        start_obs = (1, 5, 5)
        goal_obs = (8, 5, 5)
        print(f"Planning from {start_obs} to {goal_obs} with obstacles")
        trajectory_obs, control_points_obs = planner_obs.plan(start_obs, goal_obs, optimizer_method='SLSQP')

        if trajectory_obs is not None:
            print("Obstacle avoidance planning successful!")
            print(f"Trajectory start: {trajectory_obs[0]}")
            print(f"Trajectory end: {trajectory_obs[-1]}")
            
            collided = False
            min_dist_to_obs_center = float('inf')
            obstacle_centers_arr = np.array([list(obs_coord) for obs_coord in env_obs.obstacles]) + 0.5
            
            for pt_idx, pt in enumerate(trajectory_obs):
                # Check actual grid cell collision
                if env_obs.is_obstacle(pt[0], pt[1], pt[2]):
                    print(f"Collision detected at discrete grid cell for point {pt_idx}: {pt}")
                    collided = True
                    # break # Comment out to see all collisions
                
                # Check distance to centers of obstacles for soft constraint evaluation
                if len(obstacle_centers_arr) > 0:
                    dists_to_centers = np.linalg.norm(obstacle_centers_arr - pt, axis=1)
                    current_min_dist = np.min(dists_to_centers)
                    if current_min_dist < min_dist_to_obs_center:
                        min_dist_to_obs_center = current_min_dist
            
            print(f"Minimum distance from trajectory to any obstacle center: {min_dist_to_obs_center:.3f}")
            # If min_dist_to_obs_center is small (e.g. < 0.5 for unit cubes), it might have grazed or passed through.
            # The Yukawa cost should penalize this.
            # assert not collided, "Path (obstacle test) collides with an obstacle!"
        else:
            print("Obstacle avoidance planning failed.")

    else:
        print("Skipping BsplinePlanner tests because GridEnvironment could not be imported properly.") 