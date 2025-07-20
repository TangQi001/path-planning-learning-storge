# Placeholder for cost functions
def calculate_total_cost():
    pass 

import numpy as np
# from .bezier_utils import bezier_curve # For package structure
# from bezier_utils import bezier_curve # For direct script running

def position_cost(trajectory_points, goal_point, weight):
    """
    Calculates the cost associated with the distance of the trajectory's endpoint to the goal.
    Args:
        trajectory_points (np.ndarray): Array of points (N, 3) representing the trajectory.
        goal_point (np.ndarray): A 3D array for the goal position.
        weight (float): Weight for this cost component.
    Returns:
        float: The weighted position cost.
    """
    if trajectory_points is None or len(trajectory_points) == 0:
        return weight * 1e9 # High cost if trajectory is invalid
    end_point = trajectory_points[-1]
    cost = np.sum((end_point - goal_point)**2) # Squared Euclidean distance
    return weight * cost

def obstacle_cost(trajectory_points, environment, yukawa_A, yukawa_alpha, safety_distance, weight, num_obstacle_points_to_check=10):
    """
    Calculates the cost associated with obstacle proximity using Yukawa potential.
    Args:
        trajectory_points (np.ndarray): Array of points (N, 3) representing the trajectory.
        environment (GridEnvironment): The environment object with obstacle information.
        yukawa_A (float): Scaling factor for Yukawa potential.
        yukawa_alpha (float): Decay rate for Yukawa potential.
        safety_distance (float): An influence radius. Obstacles within this distance to a point contribute to cost.
        weight (float): Weight for this cost component.
        num_obstacle_points_to_check (int): Number of points along the trajectory to check against obstacles.
    Returns:
        float: The weighted obstacle cost.
    """
    if trajectory_points is None or len(trajectory_points) == 0:
        return weight * 1e12 # Very high cost if trajectory is invalid
    
    total_obstacle_cost = 0.0
    
    if len(trajectory_points) > num_obstacle_points_to_check:
        indices = np.linspace(0, len(trajectory_points) - 1, num_obstacle_points_to_check, dtype=int)
        points_to_check = trajectory_points[indices]
    else:
        points_to_check = trajectory_points

    if not environment.obstacles: # No obstacles, no cost
        return 0.0

    # Obstacle centers are assumed to be the center of the grid cell
    obstacle_centers = np.array([list(obs) for obs in environment.obstacles]) + 0.5 

    min_dist_to_any_obstacle = float('inf') # For debugging or adaptive cost

    for point in points_to_check:
        if not environment.is_within_bounds(point[0], point[1], point[2]):
            # Penalize points outside bounds heavily as if they hit an implicit boundary obstacle
            # This assumes bounds act like hard constraints enforced by a penalty
            total_obstacle_cost += yukawa_A * 1000 # Large penalty for out of bounds
            min_dist_to_any_obstacle = 0 # Effectively a collision with boundary
            continue
            
        distances_to_centers = np.linalg.norm(obstacle_centers - point, axis=1)
        
        for d_ob in distances_to_centers:
            if d_ob < safety_distance: # Only consider obstacles within the safety_distance (influence radius)
                # Effective distance for Yukawa potential: avoid d_ob being zero for the denominator.
                effective_d_ob = max(d_ob, 1e-6) # Prevent division by zero or issues with d_ob=0
                
                # Yukawa potential: A * exp(-alpha * d) / d
                # This cost should increase sharply as effective_d_ob approaches 0.
                cost_single_obstacle = yukawa_A * np.exp(-yukawa_alpha * effective_d_ob) / effective_d_ob
                total_obstacle_cost += cost_single_obstacle
            
            if d_ob < min_dist_to_any_obstacle:
                min_dist_to_any_obstacle = d_ob
                
    # print(f"Min dist to obs center for this trajectory: {min_dist_to_any_obstacle:.3f}, Obs cost component: {total_obstacle_cost:.3f}")
    return weight * total_obstacle_cost

def smoothness_cost(control_points, weight):
    """
    Calculates the smoothness cost. 
    This version penalizes the sum of squared magnitudes of the second-order differences 
    of the control points (approximates sum of squared second derivatives).
    Args:
        control_points (np.ndarray): Array of control points (M, 3).
        weight (float): Weight for this cost component.
    Returns:
        float: The weighted smoothness cost.
    """
    if control_points is None or len(control_points) < 3:
        return 0.0 # Not enough points to calculate second-order differences
    
    # Second-order differences (approximates acceleration)
    # C[i+2] - 2*C[i+1] + C[i]
    cost = 0.0
    for i in range(len(control_points) - 2):
        accel_approx = control_points[i+2] - 2 * control_points[i+1] + control_points[i]
        cost += np.sum(accel_approx**2)
    return weight * cost

def total_cost_function_for_optimizer(
    flat_variable_control_points,
    fixed_start_control_point,
    fixed_end_control_point,
    goal_point,
    environment,
    bezier_curve_func, # Function to generate Bezier curve points
    cost_params):
    """
    Calculates the total cost for the optimizer.
    Args:
        flat_variable_control_points (np.ndarray): 1D array of the variable control points' coordinates.
        fixed_start_control_point (np.ndarray): The first control point (start of the Bezier curve).
        fixed_end_control_point (np.ndarray): The last control point (end of the Bezier curve).
        goal_point (np.ndarray): The target goal position.
        environment (GridEnvironment): The environment.
        bezier_curve_func (callable): Function like bezier_utils.bezier_curve.
        cost_params (dict): Dictionary containing weights and parameters for cost functions.
                            e.g., {'w_pos': 1.0, 'w_obs': 10.0, 'w_smooth': 0.1,
                                   'yukawa_A': 1.0, 'yukawa_alpha': 1.0, 'safety_dist': 1.5,
                                   'num_traj_points': 50, 'num_obs_check_points': 20}
    Returns:
        float: The total combined cost.
    """
    # Reshape flat_variable_control_points into (num_variable_points, 3)
    # For p=6 Bezier (7 CPs), C0 and C6 are fixed. C1-C5 are variable (5 CPs).
    num_variable_cps = len(flat_variable_control_points) // 3
    variable_control_points = flat_variable_control_points.reshape((num_variable_cps, 3))
    
    # Combine fixed and variable control points
    # Order: C0, C1, ..., C5, C6 (assuming C1-C5 are variable)
    control_points = np.vstack([
        fixed_start_control_point.reshape(1,3),
        variable_control_points,
        fixed_end_control_point.reshape(1,3)
    ])

    # Generate trajectory points from the Bezier curve
    try:
        trajectory_points = bezier_curve_func(control_points, n_points=cost_params.get('num_traj_points', 50))
    except Exception as e:
        # print(f"Error generating Bezier curve: {e}")
        # print(f"Control points causing error: {control_points}")
        return 1e12 # Very high cost if curve generation fails

    # 1. Position Cost
    pos_c = position_cost(trajectory_points, goal_point, cost_params['w_pos'])

    # 2. Obstacle Cost
    obs_c = obstacle_cost(trajectory_points, environment, 
                          cost_params['yukawa_A'], cost_params['yukawa_alpha'], 
                          cost_params['safety_dist'], cost_params['w_obs'],
                          num_obstacle_points_to_check=cost_params.get('num_obs_check_points', 20))

    # 3. Smoothness Cost
    smooth_c = smoothness_cost(control_points, cost_params['w_smooth'])
    
    total_c = pos_c + obs_c + smooth_c
    # print(f"Costs: Pos={pos_c:.2f}, Obs={obs_c:.2f}, Smooth={smooth_c:.2f}, Total={total_c:.2f}")
    return total_c


if __name__ == '__main__':
    # Example Usage and Basic Tests
    # Need to mock GridEnvironment and bezier_curve for standalone testing here
    class MockGridEnvironment:
        def __init__(self, size=(9,9,9)):
            self.size = size
            self.obstacles = set()
        def add_obstacle(self, x,y,z): self.obstacles.add((x,y,z))
        def is_obstacle(self, x,y,z): return (x,y,z) in self.obstacles
        def is_within_bounds(self,x,y,z): 
            return 0<=x<self.size[0] and 0<=y<self.size[1] and 0<=z<self.size[2]

    def mock_bezier_curve(control_points, n_points=10):
        # Simple linear interpolation between control points for mock
        if len(control_points) < 2:
            return np.array([control_points[0]] * n_points) if len(control_points) == 1 else np.empty((0,3))
        
        total_segments = len(control_points) - 1
        points_per_segment = n_points // total_segments if total_segments > 0 else n_points
        
        curve = []
        for i in range(total_segments):
            start_cp = control_points[i]
            end_cp = control_points[i+1]
            for j in range(points_per_segment):
                t = j / float(points_per_segment -1 if points_per_segment > 1 else 1)
                point = start_cp * (1-t) + end_cp * t
                curve.append(point)
        # Ensure n_points by potentially adding last point or trimming
        if not curve and len(control_points) > 0:
             curve = [control_points[-1]] * n_points # if only one CP or weird n_points
        elif len(curve) < n_points and len(control_points) > 0:
            curve.extend([control_points[-1]] * (n_points - len(curve)))
        return np.array(curve)[:n_points]

    print("Testing cost functions...")

    # --- Test Position Cost ---
    traj_test_pos = np.array([[0,0,0], [1,1,1], [2,2,2]])
    goal_test_pos = np.array([2,2,2])
    cost_p = position_cost(traj_test_pos, goal_test_pos, weight=1.0)
    assert np.isclose(cost_p, 0.0), f"Position cost failed: expected 0, got {cost_p}"
    goal_test_pos_2 = np.array([3,2,2])
    cost_p_2 = position_cost(traj_test_pos, goal_test_pos_2, weight=1.0)
    assert np.isclose(cost_p_2, 1.0), f"Position cost failed: expected 1.0, got {cost_p_2}"
    print("Position cost tests passed.")

    # --- Test Obstacle Cost ---
    env_test_obs = MockGridEnvironment()
    env_test_obs.add_obstacle(1,1,1) # Obstacle at (1,1,1)
    
    # Trajectory far from obstacle
    traj_far = np.array([[5,5,5], [6,6,6]])
    cost_o_far = obstacle_cost(traj_far, env_test_obs, yukawa_A=1.0, yukawa_alpha=1.0, safety_distance=2.0, weight=1.0, num_obstacle_points_to_check=2)
    assert np.isclose(cost_o_far, 0.0), f"Obstacle cost (far) failed: expected 0, got {cost_o_far}"

    # Trajectory passing through obstacle center (using obstacle point itself for test)
    # Obstacle center is (1.5, 1.5, 1.5) if (1,1,1) is bottom-left. Let's assume obstacle is cell (1,1,1)
    # And obstacle_centers in function is (1.5,1.5,1.5)
    traj_on_obs_center = np.array([[1.5,1.5,1.5]]) 
    cost_o_on = obstacle_cost(traj_on_obs_center, env_test_obs, yukawa_A=1.0, yukawa_alpha=1.0, safety_distance=0.1, weight=1.0, num_obstacle_points_to_check=1)
    # d_ob will be 0, so should be very high cost due to 1e-6 protection
    assert cost_o_on > 1000, f"Obstacle cost (on obstacle) failed: expected very high, got {cost_o_on}"

    # Trajectory close to obstacle
    traj_close = np.array([[1.5,1.5,1.5-0.2]]) # Point very close to obstacle center at (1.5,1.5,1.5)
    # d_ob = 0.2
    # Expected: 1.0 * exp(-1.0 * 0.2) / 0.2 = exp(-0.2)/0.2 = 0.8187 / 0.2 = 4.093
    cost_o_close = obstacle_cost(traj_close, env_test_obs, yukawa_A=1.0, yukawa_alpha=1.0, safety_distance=1.0, weight=1.0, num_obstacle_points_to_check=1)
    expected_close_cost = 1.0 * np.exp(-1.0 * 0.2) / 0.2
    assert np.isclose(cost_o_close, expected_close_cost), f"Obstacle cost (close) failed: expected {expected_close_cost}, got {cost_o_close}"
    print("Obstacle cost tests passed.")

    # --- Test Smoothness Cost ---
    cps_smooth = np.array([[0,0,0], [1,1,1], [2,2,2]]) # Linear, accel=0
    cost_s_linear = smoothness_cost(cps_smooth, weight=1.0)
    assert np.isclose(cost_s_linear, 0.0), f"Smoothness cost (linear) failed: {cost_s_linear}"
    
    cps_curve = np.array([[0,0,0], [1,2,1], [2,0,2]]) # C1=(0,0,0), C2=(1,2,1), C3=(2,0,2)
    # accel_approx = C3 - 2*C2 + C1 = [2,0,2] - 2*[1,2,1] + [0,0,0] = [2,0,2] - [2,4,2] + [0,0,0] = [0, -4, 0]
    # cost = sum([0,-4,0]**2) = 0 + 16 + 0 = 16
    cost_s_curve = smoothness_cost(cps_curve, weight=1.0)
    assert np.isclose(cost_s_curve, 16.0), f"Smoothness cost (curve) failed: {cost_s_curve}"
    print("Smoothness cost tests passed.")

    # --- Test Total Cost Function ---
    start_cp = np.array([0,0,0])
    end_cp = np.array([8,8,8])
    goal_p = np.array([8,8,8])
    
    # 5 variable CPs, 3 coords each = 15 vars
    # C1, C2, C3, C4, C5
    # Let them be linearly interpolated for a simple test case
    var_cps_flat = np.array([
        1.6, 1.6, 1.6, #C1
        3.2, 3.2, 3.2, #C2
        4.8, 4.8, 4.8, #C3
        6.4, 6.4, 6.4, #C4
        8.0, 8.0, 8.0  #C5 - this makes C5=C6, not ideal. Let C5 be (7,7,7)
    ])
    var_cps_flat_better = np.array([
        1.5, 1.5, 1.5, #C1
        3.0, 3.0, 3.0, #C2
        4.5, 4.5, 4.5, #C3
        6.0, 6.0, 6.0, #C4
        7.5, 7.5, 7.5  #C5
    ])

    test_env = MockGridEnvironment(size=(10,10,10))
    # No obstacles for this first total_cost test

    cost_params_test = {
        'w_pos': 100.0, 'w_obs': 1.0, 'w_smooth': 0.1,
        'yukawa_A': 1.0, 'yukawa_alpha': 0.5, 'safety_dist': 1.0, # safety_dist influences effective range
        'num_traj_points': 10, 'num_obs_check_points': 5
    }

    total_c = total_cost_function_for_optimizer(
        var_cps_flat_better,
        start_cp, end_cp, goal_p,
        test_env, mock_bezier_curve, 
        cost_params_test
    )
    # Expected costs with mock_bezier_curve (linear interpolation between CPs)
    # Trajectory end point will be end_cp = [8,8,8]. Goal is [8,8,8]. So pos_cost = 0.
    # No obstacles, so obs_cost = 0.
    # Smoothness: CPs are [0,0,0], [1.5,1.5,1.5] ... [7.5,7.5,7.5], [8,8,8]
    # All second diffs are 0 for this perfect linear spacing. So smooth_cost = 0.
    # Total cost should be 0.
    assert np.isclose(total_c, 0.0), f"Total cost (linear, no obs) failed: {total_c}"

    # Add an obstacle that the linear path would hit
    test_env.add_obstacle(4,4,4) # Obstacle at (4,4,4)
    # Our mock_bezier_curve will generate points like (4.5,4.5,4.5) which is close to (4.5,4.5,4.5) center of obs.
    # Distance from (4.5,4.5,4.5) to obs center (4.5,4.5,4.5) is 0.
    # This should trigger high obstacle cost.
    total_c_obs = total_cost_function_for_optimizer(
        var_cps_flat_better,
        start_cp, end_cp, goal_p,
        test_env, mock_bezier_curve, 
        cost_params_test
    )
    assert total_c_obs > cost_params_test['w_obs'] * 100, f"Total cost (linear, with obs) failed: {total_c_obs}" # Expect high obs_c

    print("Total cost function tests (basic) passed.")
    print("All cost function tests completed.") 