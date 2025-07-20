
# Fix Chinese font display
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback font configuration
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# Placeholder for 3D visualization module
def plot_simulation():
    pass 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.animation as animation

# Assuming GridEnvironment is importable
# from .environment import GridEnvironment

def setup_3d_plot(title='UAV Path Planning Simulation'):
    """Sets up a 3D plot for visualization."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    return fig, ax

def plot_grid_bounds(ax, grid_size):
    """Plots the boundaries of the grid."""
    x, y, z = grid_size
    # Define the 8 corners of the cube
    corners = np.array([
        [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],  # Bottom face
        [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]   # Top face
    ])
    # Define the 12 edges
    edges = [
        [corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]], # Bottom
        [corners[4], corners[5]], [corners[5], corners[6]], [corners[6], corners[7]], [corners[7], corners[4]], # Top
        [corners[0], corners[4]], [corners[1], corners[5]], [corners[2], corners[6]], [corners[3], corners[7]]  # Sides
    ]
    for edge in edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], 'k--', alpha=0.5)
    ax.set_xlim([0, x])
    ax.set_ylim([0, y])
    ax.set_zlim([0, z])

def plot_obstacles(ax, environment, color='gray', alpha=0.7):
    """Plots the obstacles in the environment as 3D cubes."""
    if not environment.obstacles:
        return

    cubes = []
    for obs_x, obs_y, obs_z in environment.obstacles:
        # Define the 8 vertices of the cube for this obstacle
        # Obstacle is at (obs_x, obs_y, obs_z), extends to (obs_x+1, obs_y+1, obs_z+1)
        v = np.array([
            [obs_x, obs_y, obs_z], [obs_x+1, obs_y, obs_z], [obs_x+1, obs_y+1, obs_z], [obs_x, obs_y+1, obs_z], # Bottom face
            [obs_x, obs_y, obs_z+1], [obs_x+1, obs_y, obs_z+1], [obs_x+1, obs_y+1, obs_z+1], [obs_x, obs_y+1, obs_z+1]  # Top face
        ])
        # Define the 6 faces of the cube
        faces = [
            [v[0], v[1], v[2], v[3]], # Bottom
            [v[4], v[5], v[6], v[7]], # Top
            [v[0], v[1], v[5], v[4]], # Front
            [v[2], v[3], v[7], v[6]], # Back
            [v[1], v[2], v[6], v[5]], # Right
            [v[0], v[3], v[7], v[4]]  # Left
        ]
        cubes.append(faces)
    
    # Create a Poly3DCollection
    # Note: Poly3DCollection expects a list of lists of (x,y,z) tuples or arrays for each polygon.
    # The way `faces` is constructed, each element of `cubes` is a list of 6 faces,
    # and each face is a list of 4 vertices. This might need to be flattened or handled carefully.
    
    # For voxel-like plotting, it's often easier to use ax.voxels if available and appropriate,
    # or draw each face separately if Poly3DCollection is tricky with many cubes.
    # A simpler way for filled cubes with solid color:
    for obs_x, obs_y, obs_z in environment.obstacles:
        ax.bar3d(obs_x, obs_y, obs_z, 1, 1, 1, color=color, alpha=alpha, shade=True, edgecolor='black', linewidth=0.5)

def plot_points(ax, points, color='r', marker='o', size=50, label=None, depthshade=True):
    """Plots one or more 3D points. Label is passed directly to scatter."""
    points_arr = np.array(points)
    if points_arr.ndim == 1:
        points_arr = points_arr.reshape(1, -1)
    if points_arr.shape[1] != 3:
        print(f"Warning: Points for plotting are not 3D: {points_arr}")
        return
    ax.scatter(points_arr[:,0], points_arr[:,1], points_arr[:,2], 
               c=color, marker=marker, s=size, label=label, depthshade=depthshade, alpha=0.9)

def plot_path(ax, path, color='b', linestyle='-', linewidth=2, label=None, marker=None, markersize=5):
    """Plots a 3D path. Label is passed directly to plot."""
    if path is None or len(path) == 0:
        return
    path_arr = np.array(path)
    ax.plot(path_arr[:,0], path_arr[:,1], path_arr[:,2], 
            color=color, linestyle=linestyle, linewidth=linewidth, label=label, 
            marker=marker if marker else '', markersize=markersize if marker else 0)

def plot_control_points(ax, control_points, color='g', marker='x', size=40, label='Control Points', linestyle=':', connect_lines=True):
    """Plots Bezier control points. If connect_lines, the line gets the label."""
    if control_points is None or len(control_points) == 0:
        return
    cp_arr = np.array(control_points)
    
    # Scatter CPs. Label only if lines are not connected and labeled.
    scatter_actual_label = label if not connect_lines else None 
    ax.scatter(cp_arr[:,0], cp_arr[:,1], cp_arr[:,2], color=color, marker=marker, s=size, label=scatter_actual_label, alpha=0.7)

    if connect_lines:
        # Connected line gets the primary label for the CPs in the legend.
        ax.plot(cp_arr[:,0], cp_arr[:,1], cp_arr[:,2], 
                color=color, linestyle=linestyle, linewidth=1, label=label, marker=marker, markersize=size/2)

def static_visualization(environment, start_point, goal_point, 
                         path_astar=None, path_bspline=None, 
                         bspline_control_points=None, title='Path Planning Simulation'):
    """Creates a static 3D visualization of the environment, paths, and points."""
    fig, ax = setup_3d_plot(title=title)
    handles = [] # For legend

    plot_grid_bounds(ax, environment.size)
    plot_obstacles(ax, environment) # Obstacles don't have a legend entry by default
    
    h = plot_points(ax, start_point, color='lime', marker='o', size=100, label='Start')
    if h: handles.append(h)
    h = plot_points(ax, goal_point, color='red', marker='*', size=150, label='Goal')
    if h: handles.append(h)

    if path_astar:
        h = plot_path(ax, path_astar, color='blue', linestyle='--', linewidth=2, label='A* Path', marker='.')
        if h: handles.append(h)
    
    if path_bspline:
        h = plot_path(ax, path_bspline, color='orange', linestyle='-', linewidth=2.5, label='B-spline Path')
        if h: handles.append(h)
    
    if bspline_control_points is not None:
        _, h_line = plot_control_points(ax, bspline_control_points, color='magenta', marker='x', size=50, connect_lines=True, label='B-spline CPs')
        if h_line: handles.append(h_line)
    
    # Create legend from collected handles
    # Filter out None handles just in case
    valid_handles = [h for h in handles if h is not None]
    labels = [h.get_label() for h in valid_handles]
    # Remove underscore prefixed labels (like _nolegend_)
    final_handles_labels = [(h, l) for h, l in zip(valid_handles, labels) if not l.startswith('_')]
    final_handles = [item[0] for item in final_handles_labels]
    final_labels = [item[1] for item in final_handles_labels]

    if final_handles:
        ax.legend(final_handles, final_labels)
    plt.show()


# --- Animation Functions (to be developed further) ---

uav_marker_astar, = None, # UAV marker for A* path
uav_marker_bspline, = None, # UAV marker for B-spline path

def init_animation(ax_ref, environment, start_point, goal_point, path_astar_ref, path_bspline_ref, bspline_cps_ref):
    """Initialize the animation plot elements. Ensures legend is clean."""
    global uav_marker_astar, uav_marker_bspline
    
    ax_ref.clear() # Clear everything: data, artists, legend, title, labels etc.

    ax_ref.set_xlabel('X coordinate')
    ax_ref.set_ylabel('Y coordinate')
    ax_ref.set_zlabel('Z coordinate')
    ax_ref.set_title('UAV Path Planning Animation')
    plot_grid_bounds(ax_ref, environment.size)

    plot_obstacles(ax_ref, environment, alpha=0.5)
    plot_points(ax_ref, start_point, color='lime', marker='o', size=100, label='Start')
    plot_points(ax_ref, goal_point, color='red', marker='*', size=150, label='Goal')

    if path_astar_ref is not None and len(path_astar_ref) > 0:
        plot_path(ax_ref, path_astar_ref, color='blue', linestyle=':', linewidth=1.5, label='A* Path')
    
    if path_bspline_ref is not None and len(path_bspline_ref) > 0:
        plot_path(ax_ref, path_bspline_ref, color='orange', linestyle='-', linewidth=2, label='B-spline Path')
        if bspline_cps_ref is not None:
            plot_control_points(ax_ref, bspline_cps_ref, color='magenta', marker='x', size=40, connect_lines=True, label='B-spline CPs')

    astar_initial_pos = path_astar_ref[0] if (path_astar_ref is not None and len(path_astar_ref) > 0) else (np.nan, np.nan, np.nan)
    uav_marker_astar, = ax_ref.plot([astar_initial_pos[0]], [astar_initial_pos[1]], [astar_initial_pos[2]], 
                                    marker='o', color='darkblue', markersize=8, label='UAV (A*)')

    bspline_initial_pos = path_bspline_ref[0] if (path_bspline_ref is not None and len(path_bspline_ref) > 0) else (np.nan, np.nan, np.nan)
    uav_marker_bspline, = ax_ref.plot([bspline_initial_pos[0]], [bspline_initial_pos[1]], [bspline_initial_pos[2]], 
                                      marker='s', color='darkorange', markersize=8, label='UAV (B-spline)')
    
    # Explicitly create legend from unique handles and labels
    handles, labels = ax_ref.get_legend_handles_labels()
    # Filter out underscore-prefixed labels (often used for no-legend items)
    # and create a dictionary to ensure unique labels (last one wins for a given label string)
    by_label = {label: handle for handle, label in zip(handles, labels) if not label.startswith('_')}
    if by_label: # Ensure there's something to make a legend from
        ax_ref.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    artists_to_return = []
    if uav_marker_astar is not None: artists_to_return.append(uav_marker_astar)
    if uav_marker_bspline is not None: artists_to_return.append(uav_marker_bspline)
    return tuple(artists_to_return)

def update_animation(frame_num, path_astar_ref, path_bspline_ref):
    """Update function for animation frames."""
    global uav_marker_astar, uav_marker_bspline
    artists_to_return = []

    if path_astar_ref is not None and len(path_astar_ref) > 0:
        idx_astar = min(frame_num, len(path_astar_ref) - 1)
        x_a, y_a, z_a = path_astar_ref[idx_astar]
        uav_marker_astar.set_data_3d([x_a], [y_a], [z_a]) # Wrap coordinates in lists
        artists_to_return.append(uav_marker_astar)

    if path_bspline_ref is not None and len(path_bspline_ref) > 0:
        idx_bspline = min(frame_num, len(path_bspline_ref) - 1)
        x_b, y_b, z_b = path_bspline_ref[idx_bspline]
        uav_marker_bspline.set_data_3d([x_b], [y_b], [z_b]) # Wrap coordinates in lists
        artists_to_return.append(uav_marker_bspline)

    return tuple(artists_to_return)

def dynamic_visualization(environment, start_point, goal_point, 
                          path_astar, path_bspline, 
                          bspline_control_points=None, 
                          interval=50, save_gif=False, gif_filename='uav_simulation.gif'):
    """Creates a dynamic 3D animation of UAV movement along paths."""
    fig, ax = setup_3d_plot(title='UAV Path Planning Animation')
    
    # Determine the number of frames based on the longest path
    num_frames_astar = len(path_astar) if path_astar is not None else 0
    num_frames_bspline = len(path_bspline) if path_bspline is not None else 0
    max_frames = max(num_frames_astar, num_frames_bspline, 1) # Ensure at least 1 frame

    # Use partial to pass static arguments to init_animation and update_animation
    # This is not strictly necessary for init_func if it takes no arguments other than the ones it gets
    # But for update, we need to pass path_astar and path_bspline
    
    # Global markers are used, so init_animation will set them up.
    # The animation function itself.
    ani = animation.FuncAnimation(fig, 
                                  update_animation, 
                                  frames=max_frames, 
                                  init_func=lambda: init_animation(ax, environment, start_point, goal_point, path_astar, path_bspline, bspline_control_points),
                                  fargs=(path_astar, path_bspline), 
                                  blit=True, # blit=True requires init_func and update_func to return an iterable of artists to be redrawn
                                  interval=interval, 
                                  repeat=True, repeat_delay=1000)

    if save_gif:
        try:
            print(f"Saving animation to {gif_filename}...")
            # Ensure imagemagick is installed and configured for matplotlib
            # On Windows, you might need: plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-VERSION\magick.exe'
            # or for Pillow: writer = animation.PillowWriter(fps=15)
            writer = animation.PillowWriter(fps=int(1000/interval))
            ani.save(gif_filename, writer=writer)
            print(f"Animation saved to {gif_filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}. Ensure a GIF writer (like Pillow or ImageMagick) is installed and configured.")
            print("Showing plot instead.")
            plt.show()
    else:
        plt.show()

    return ani # Return animation object to keep it alive if not saving


if __name__ == '__main__':
    # Example Usage (requires other modules or mocks)
    GridEnvironment_imported = False
    try:
        from environment import GridEnvironment # Assumes environment.py is in the same dir
        GridEnvironment_imported = True
    except ImportError:
        print("Visualization Test: Could not import GridEnvironment. Using Mock.")
        class MockGridEnvironment:
            def __init__(self, size=(9,9,9)):
                self.size = size; self.width, self.height, self.depth = size
                self.obstacles = set()
            def add_obstacle(self,x,y,z): self.obstacles.add((int(x),int(y),int(z)))
        GridEnvironment = MockGridEnvironment
        GridEnvironment_imported = True

    if GridEnvironment_imported:
        env = GridEnvironment(size=(10, 10, 10))
        env.add_obstacle(5, 5, 5)
        env.add_obstacle(5, 4, 5)
        env.add_obstacle(4, 5, 5)
        env.add_obstacle(6, 5, 5)
        env.add_obstacle(5, 6, 5)
        env.add_obstacle(5,5,4)
        env.add_obstacle(5,5,6)


        start = (1, 1, 1)
        goal = (8, 8, 8)

        # Mock paths
        path_a = np.array([
            [1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,4,4], 
            [6,5,5], [7,6,6], [7,7,7], [8,8,8]
        ]) * 1.0 # Ensure float for plotting
        path_b = np.array([
            [1,1,1], [1.5,2,2.5], [2.5,3,4], [4,4.5,5], [5.5,5,5.5], 
            [6.5,6,6.5], [7,7,7.5], [8,8,8]
        ]) * 1.0
        cps_b = np.array([
            [1,1,1], [1,2,3], [3,3,5], [5,5,5], [6,7,6], [7,7,8], [8,8,8]
        ]) * 1.0

        print("Showing static visualization example...")
        # static_visualization(env, start, goal, path_astar=path_a, path_bspline=path_b, bspline_control_points=cps_b)
        
        print("\nShowing dynamic visualization example...")
        # Note: Animation object needs to be kept alive, e.g., by assigning to a variable in a script,
        # or by blocking with plt.show() which happens inside dynamic_visualization if not saving.
        # To run this test and see the animation, uncomment the line below.
        # This will also attempt to save a GIF if save_gif is True.
        # _ = dynamic_visualization(env, start, goal, path_a, path_b, cps_b, save_gif=False, interval=100)
        # print("If you don't see an animation, ensure matplotlib backend supports it (e.g., TkAgg, Qt5Agg).")
        print("Dynamic visualization test setup. Uncomment the call to `dynamic_visualization` to run.")

        # Test with None paths for animation robustness
        print("\nTesting dynamic visualization with one path missing...")
        # _ = dynamic_visualization(env, start, goal, path_a, None, None, save_gif=False, interval=100)
        print("Dynamic visualization test with missing B-spline path setup.")
        
        # _ = dynamic_visualization(env, start, goal, None, path_b, cps_b, save_gif=False, interval=100)
        print("Dynamic visualization test with missing A* path setup.")

        print("\nTesting dynamic visualization with all paths missing...")
        _ = dynamic_visualization(env, start, goal, None, None, None, save_gif=False, interval=200)
        print("Dynamic visualization test with all paths missing completed (should show empty animation).") 