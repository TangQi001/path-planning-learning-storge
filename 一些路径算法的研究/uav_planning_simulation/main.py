# Main script to run the UAV path planning simulation

import numpy as np

# Assuming all module files are in the same directory (uav_planning_simulation)
# and this script is also in that directory or the directory is in PYTHONPATH.
from environment import GridEnvironment
from astar_planner import find_astar_path
from bspline_planner import BsplinePlanner
from visualization import static_visualization, dynamic_visualization

# Fix Chinese font display
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback font configuration
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# bezier_utils are used by bspline_planner and cost_functions, so not directly needed here usually
# from bezier_utils import bezier_curve 
# cost_functions are used by bspline_planner, so not directly needed here usually
# from cost_functions import total_cost_function_for_optimizer

def run_simulation():
    print("Starting UAV Path Planning Simulation...")

    # 1. Setup Environment
    grid_size = (9, 9, 9)
    env = GridEnvironment(size=grid_size)

    # Define Start and Goal points
    start_point = (1.0, 1.0, 1.0) # Using float for consistency, A* will convert to int for grid
    goal_point = (7.0, 7.0, 7.0)

    # Add a central obstacle block
    # Obstacle from (3,3,3) to (5,5,5) (exclusive of upper bound for cells)
    # So cells (3,y,z), (4,y,z) for x, etc.
    obstacle_coords = []
    for x in range(3, 6): # Cells 3, 4, 5
        for y in range(3, 6): # Cells 3, 4, 5
            for z in range(3, 6): # Cells 3, 4, 5
                obstacle_coords.append((x,y,z))
    env.add_obstacles(obstacle_coords)
    # env.add_obstacle(4,4,4) # Simpler single central obstacle cell
    
    print(f"Environment: {grid_size} grid.")
    print(f"Start: {start_point}, Goal: {goal_point}")
    print(f"Number of obstacle cells: {len(env.obstacles)}")
    if len(env.obstacles) < 10: 
        print(f"Obstacle coordinates: {env.obstacles}")

    # 2. Run A* Planner
    print("\nRunning A* Planner...")
    path_astar = find_astar_path(env, start_point, goal_point)
    if path_astar:
        print(f"A* Path found with {len(path_astar)} points.")
        # print(f"A* Path: {path_astar}")
    else:
        print("A* Path not found.")

    # 3. Setup and Run B-spline (Bezier) Planner
    print("\nSetting up B-spline (Bezier) Planner...")
    # Cost parameters for B-spline planner (crucial for performance and behavior)
    cost_params_bspline = {
        'w_pos': 1000.0,          # High weight to ensure endpoint meets goal
        'w_obs': 20000.0,         # SIGNIFICANTLY INCREASED: Weight for obstacle avoidance
        'w_smooth': 30.0,         # Weight for path smoothness (control point acceleration)
        'yukawa_A': 50.0,          # INCREASED: Yukawa potential scale factor (strength of repulsion)
        'yukawa_alpha': 0.8,      # SLIGHTLY DECREASED: Yukawa potential decay rate (wider influence)
        'safety_dist': 1.8,       # SLIGHTLY INCREASED: Effective distance for Yukawa potential calculation
        'num_traj_points': 50,    # Points on Bezier curve for cost evaluation during optimization
        'num_obs_check_points': 50, # INCREASED: Points along trajectory to check for obstacle cost
        'num_traj_points_final': 100, # Points for the final generated trajectory for visualization
        'optimizer_options': {    # Options for scipy.optimize.minimize
            'disp': False,        # Display convergence messages
            'maxiter': 300,      # Max iterations (increase if optimization fails prematurely or for complex scenarios)
            'ftol': 1e-8,        # Precision goal for the value of f in the stopping criterion
            'eps': 1.4901161193847656e-08 # Default step size for SLSQP numerical differentiation
        }
    }

    bspline_planner = BsplinePlanner(env, cost_params_bspline, bezier_order=6) # p=6 -> 7 CPs
    
    print("Running B-spline (Bezier) Planner Optimization...")
    # This can take some time depending on complexity and optimizer settings
    path_bspline, control_points_bspline = bspline_planner.plan(
        start_point, 
        goal_point, 
        optimizer_method='SLSQP' # SLSQP supports bounds and is generally good for this type of problem
                                  # L-BFGS-B also an option if jacobian is provided or approximated
    )

    if path_bspline is not None:
        print(f"B-spline Path found with {len(path_bspline)} points.")
        # print(f"B-spline Control Points:\n{control_points_bspline}")
    else:
        print("B-spline Path not found or optimization failed.")

    # 4. Visualization
    print("\nPreparing visualization...")
    
    # You can choose static or dynamic visualization
    # print("Showing static visualization first...")
    # static_visualization(env, start_point, goal_point, 
    #                      path_astar, path_bspline, control_points_bspline,
    #                      title="UAV Path Planning Comparison")

    print("Showing dynamic visualization...")
    # The animation object needs to be kept alive if not blocking with plt.show() or saving.
    # dynamic_visualization itself calls plt.show() if not saving.
    _ = dynamic_visualization(env, start_point, goal_point, 
                              path_astar, path_bspline, control_points_bspline,
                              interval=70,  # milliseconds per frame
                              save_gif=False, # Changed to False for interactive plot
                              gif_filename='uav_planning_simulation.gif')
    
    print("\nSimulation run complete.")
    # print("If a GIF was saved, it should be in the 'uav_planning_simulation' directory.") # Commented out as GIF is not saved by default now
    print("An interactive 3D plot window should have opened.")

if __name__ == "__main__":
    # This structure assumes that if you run `python main.py` from within the `uav_planning_simulation` directory,
    # the sibling .py files (environment.py, etc.) will be directly importable.
    
    # To make imports more robust if running from outside, you might need to adjust PYTHONPATH
    # or use relative imports if uav_planning_simulation is treated as a package, e.g.
    # if uav_planning_simulation had an __init__.py and you ran a script outside it like:
    # from uav_planning_simulation.main import run_simulation
    # run_simulation()
    # But for this project structure, direct imports should work when main.py is in the same dir as others.
    
    run_simulation() 